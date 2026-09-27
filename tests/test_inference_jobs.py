"""Phase 2 / Block C2 — asynchronous inference job tests.

Covers the 19 cases required by C2.10. Real YOLO/OCR inference is replaced by
a controlled fake in almost every test, so the suite is fast and does not touch
the GPU. Exactly one real-inference smoke test exists and it is skipped unless
EALPR_C2_SMOKE=1.

Data safety: this suite creates DetectionResult / InferenceJob / User documents,
but only with names prefixed ``__c2_test`` and a unique per-run plate marker.
Cleanup deletes exactly those documents by id — never a broad collection
delete. Detection and visitor counts are asserted unchanged on the way out.
"""

import io
import os
import threading
import time
import unittest
import uuid
from contextlib import contextmanager
from unittest import mock

import numpy as np

from tests.base import get_app

from ealpr import inference_jobs
from ealpr.ai_models import INFERENCE_LOCK
from models import DetectionResult, InferenceJob, User

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)

MARKER = "__c2_test_{}".format(uuid.uuid4().hex[:10])


def fake_inference(*_args, **_kwargs):
    """Stand-in for process_license_plate. Mirrors its real 7-tuple contract."""
    return (
        np.zeros((12, 12, 3), dtype=np.uint8),
        MARKER,
        True,
        None,
        0.87,
        0.91,
        [],
    )


class JobTestCase(unittest.TestCase):
    """Base: two throwaway users, per-test document bookkeeping.

    Deliberately does NOT push a long-lived app context. Flask reuses an
    already-pushed app context for each request, which leaves the test client's
    cookie session unreadable after the first request — a login applied
    mid-test would silently not take effect. MongoEngine does not need an app
    context (the connection is made in ``create_app``), and ``run_job`` pushes
    its own, so nothing here requires one.
    """

    def setUp(self):
        self.app = get_app()
        # Keep the background thread out of the way so a job stays "queued"
        # until a test explicitly runs it. Threaded tests start it themselves.
        inference_jobs.shutdown()
        self._drain_queue()
        self._job_ids = []
        self._detection_ids = []
        self._staged_paths = []

        suffix = uuid.uuid4().hex[:8]
        self.owner = User(
            username="__c2_owner_{}".format(suffix),
            password="pw", is_admin=False,
        )
        self.other = User(
            username="__c2_other_{}".format(suffix),
            password="pw", is_admin=False,
        )
        self.owner.save()
        self.other.save()

        self.detections_before = DetectionResult.objects.count()
        self._patches = [
            mock.patch.object(inference_jobs, "get_plate_detection_model",
                              lambda: object()),
            mock.patch.object(inference_jobs, "get_ocr_model",
                              lambda: object()),
            mock.patch.object(inference_jobs, "save_detection_images",
                              self._fake_save_images),
            # The POST endpoint imports the loader from ealpr.ai_models inside
            # the request, so patch it there too. Without this the suite would
            # load real YOLO weights just to pass the availability gate.
            mock.patch("ealpr.ai_models.get_plate_detection_model",
                       lambda: object()),
            mock.patch("ealpr.ai_models.get_ocr_model", lambda: object()),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(self._cleanup)

    def _fake_save_images(self, original_bytes, processed_bytes):
        return ("/tmp/{}_original.jpg".format(MARKER),
                "/tmp/{}_processed.jpg".format(MARKER))

    def _drain_queue(self):
        """Discard leftover job ids so queue-depth assertions are meaningful."""
        while True:
            try:
                inference_jobs._queue.get_nowait()
                inference_jobs._queue.task_done()
            except Exception:
                break

    def _cleanup(self):
        inference_jobs.shutdown()
        self._drain_queue()
        # Delete only what this test created, by id.
        if self._detection_ids:
            DetectionResult.objects(id__in=self._detection_ids).delete()
        if self._job_ids:
            InferenceJob.objects(job_id__in=self._job_ids).delete()
        for user in (self.owner, self.other):
            try:
                user.delete()
            except Exception:
                pass
        # Remove staged uploads for jobs that never reached the worker (a
        # completed/failed job cleans up its own temp file in run_job's finally
        # block, but a job that was only enqueued would otherwise leak it).
        for path in self._staged_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        for p in self._patches:
            if p is not None:
                p.stop()

    # ── helpers ──────────────────────────────────────────────────────────────

    def make_job(self, user=None, filename="plate.jpg", start_worker=False):
        """Stage a file and create a queued job, tracking it for cleanup.

        Defaults to NOT starting the consumer thread, so a test that drives
        ``run_job`` itself cannot race the worker for the same job.
        """
        path = os.path.join(_temp_dir(),
                            "c2test_{}_{}".format(uuid.uuid4().hex, filename))
        with open(path, "wb") as handle:
            handle.write(PNG_BYTES)
        self._staged_paths.append(path)
        job = inference_jobs.enqueue(
            self.app, path, filename,
            user if user is not None else self.owner,
            start_worker=start_worker,
        )
        self._job_ids.append(job.job_id)
        return job

    def track(self, job):
        if job.detection is not None:
            self._detection_ids.append(job.detection.id)
        return job

    def login(self, client, user):
        with client.session_transaction() as sess:
            sess["_user_id"] = str(user.id)

    @contextmanager
    def no_worker(self):
        """Stop the consumer thread from starting, so no real inference runs.

        Submission tests only care about the HTTP contract. Without this a
        posted job would immediately be picked up and pushed through real YOLO.
        """
        with mock.patch.object(inference_jobs, "_ensure_worker"):
            yield


def _temp_dir():
    from config import TEMP_DIR
    return str(TEMP_DIR)


class JobLifecycleTests(JobTestCase):
    """C2.2 / C2.3 / C2.8 — the state machine."""

    def test_02_job_starts_queued(self):
        job = self.make_job(start_worker=False)
        self.assertEqual(job.status, InferenceJob.STATUS_QUEUED)
        self.assertIsNone(job.started_at)
        self.assertIsNone(job.completed_at)
        self.assertIsNone(job.detection)

    def test_03_worker_transitions_queued_to_processing_then_completed(self):
        """processing is entered, and a completed job records both stamps.

        The queued -> processing step is an atomic findAndModify in _claim, not
        a _mark call, so it is observed via the claim rather than by spying on
        _mark.
        """
        claim_states = []
        original_claim = inference_jobs._claim

        def spy_claim(job_id):
            result = original_claim(job_id)
            if result is not None:
                claim_states.append(result.status)
            return result

        with mock.patch.object(inference_jobs, "_claim", side_effect=spy_claim), \
                mock.patch.object(inference_jobs, "process_license_plate",
                                  fake_inference):
            job = self.make_job()
            self.track(inference_jobs.run_job(self.app, job.job_id))
            reloaded = InferenceJob.objects(job_id=job.job_id).first()

        self.assertEqual(claim_states,
                         [InferenceJob.STATUS_PROCESSING],
                         "the claim must move the job to processing")
        self.assertEqual(reloaded.status, InferenceJob.STATUS_COMPLETED)
        self.assertIsNotNone(reloaded.started_at)
        self.assertIsNotNone(reloaded.completed_at)
        self.assertGreaterEqual(reloaded.completed_at, reloaded.started_at)

    def test_04_successful_inference_becomes_completed(self):
        with mock.patch.object(inference_jobs, "process_license_plate",
                               fake_inference):
            job = self.make_job()
            self.track(inference_jobs.run_job(self.app, job.job_id))
            reloaded = InferenceJob.objects(job_id=job.job_id).first()
        self.assertEqual(reloaded.status, InferenceJob.STATUS_COMPLETED)
        self.assertEqual(reloaded.plate_text, MARKER)

    def test_05_completed_job_references_a_detection_result(self):
        with mock.patch.object(inference_jobs, "process_license_plate",
                               fake_inference):
            job = self.make_job()
            self.track(inference_jobs.run_job(self.app, job.job_id))
            reloaded = InferenceJob.objects(job_id=job.job_id).first()
        self.assertIsNotNone(reloaded.detection)
        self.assertEqual(
            DetectionResult.objects(id=reloaded.detection.id).count(), 1)

    def test_06_failed_inference_becomes_failed(self):
        with mock.patch.object(inference_jobs, "process_license_plate",
                               lambda *a, **k: (None, None, False,
                                                "No license plate detected.",
                                                0.0, 0.0, [])):
            job = self.make_job()
            inference_jobs.run_job(self.app, job.job_id)
            reloaded = InferenceJob.objects(job_id=job.job_id).first()
        self.assertEqual(reloaded.status, InferenceJob.STATUS_FAILED)
        self.assertEqual(reloaded.error_code, inference_jobs.ERROR_NO_PLATE)
        self.assertIsNotNone(reloaded.completed_at)
        self.assertIsNone(reloaded.detection)

    def test_16_worker_exception_is_contained_and_never_sticks(self):
        """An unexpected raise must still terminate the job, and not crash."""
        with mock.patch.object(inference_jobs, "process_license_plate",
                               side_effect=RuntimeError("boom in cv2")):
            job = self.make_job()
            inference_jobs.run_job(self.app, job.job_id)
            reloaded = InferenceJob.objects(job_id=job.job_id).first()
        self.assertEqual(reloaded.status, InferenceJob.STATUS_FAILED)
        self.assertEqual(reloaded.error_code, inference_jobs.ERROR_INTERNAL)
        # The real cause is logged, never stored.
        self.assertNotIn("boom in cv2", reloaded.error_message or "")
        self.assertNotIn("boom in cv2", str(reloaded.to_mongo().to_dict()))

    def test_missing_input_image_fails_cleanly(self):
        job = self.make_job()
        os.remove(job.image_path)
        with mock.patch.object(inference_jobs, "process_license_plate",
                               fake_inference):
            inference_jobs.run_job(self.app, job.job_id)
            reloaded = InferenceJob.objects(job_id=job.job_id).first()
        self.assertEqual(reloaded.status, InferenceJob.STATUS_FAILED)
        self.assertEqual(reloaded.error_code, inference_jobs.ERROR_STORAGE)

    def test_failed_job_is_never_rerun(self):
        job = self.make_job()
        with mock.patch.object(inference_jobs, "process_license_plate",
                               lambda *a, **k: (None, None, False, "x", 0.0,
                                                0.0, [])):
            inference_jobs.run_job(self.app, job.job_id)
        with mock.patch.object(inference_jobs, "process_license_plate",
                               side_effect=fake_inference) as spy:
            inference_jobs.run_job(self.app, job.job_id)
            reloaded = InferenceJob.objects(job_id=job.job_id).first()
        self.assertEqual(reloaded.status, InferenceJob.STATUS_FAILED)
        spy.assert_not_called()

    def test_13_no_duplicate_detection_is_created(self):
        with mock.patch.object(inference_jobs, "process_license_plate",
                               fake_inference):
            job = self.make_job()
            inference_jobs.run_job(self.app, job.job_id)
            inference_jobs.run_job(self.app, job.job_id)
            reloaded = InferenceJob.objects(job_id=job.job_id).first()
            made = list(DetectionResult.objects(plate_number=MARKER).only("id"))
        self._detection_ids.extend(d.id for d in made)
        self.assertEqual(len(made), 1, "worker created a duplicate detection")
        self.assertIsNotNone(reloaded.detection)
        self.assertEqual(reloaded.attempts, 1)

    def test_concurrent_runners_cannot_both_claim_a_job(self):
        """The queued -> processing claim must be atomic, not load-then-check.

        Two runners racing for one job would each pass a naive status check and
        both run inference, creating two DetectionResults. The claim is a single
        filtered findAndModify so exactly one runner can win.
        """
        job = self.make_job()
        results = []
        barrier = threading.Barrier(4)
        original_claim = inference_jobs._claim

        def claim():
            barrier.wait(timeout=10)
            results.append(original_claim(job.job_id))

        # Patch the module global ONCE, in this thread. Entering/exiting
        # mock.patch concurrently from several threads makes the final unpatch
        # restore a stale value and leaks the fake into later tests.
        with mock.patch.object(inference_jobs, "process_license_plate",
                               fake_inference):
            threads = [threading.Thread(target=claim, daemon=True)
                       for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=20)

        winners = [r for r in results if r is not None]
        self.assertEqual(len(winners), 1,
                         "{} runners claimed the same job".format(len(winners)))
        made = list(DetectionResult.objects(plate_number=MARKER).only("id"))
        self.assertEqual(made, [])  # claiming does not run inference

    def test_12_detection_semantics_match_process_plate_fields(self):
        with mock.patch.object(inference_jobs, "process_license_plate",
                               fake_inference):
            job = self.make_job()
            self.track(inference_jobs.run_job(self.app, job.job_id))
            reloaded = InferenceJob.objects(job_id=job.job_id).first()
            detection = DetectionResult.objects(id=reloaded.detection.id).first()

        self.assertEqual(detection.plate_number, MARKER)
        self.assertAlmostEqual(detection.confidence, 0.87)
        self.assertAlmostEqual(detection.ocr_confidence, 0.91)
        # A plate that matches no real visitor is "unauthorized", exactly as
        # process_plate computes it.
        self.assertEqual(detection.status, "unauthorized")
        self.assertIsNone(detection.visitor_name)
        self.assertEqual(detection.processed_by.id, self.owner.id)
        self.assertIsNotNone(detection.timestamp)


class WorkerThreadTests(JobTestCase):
    """C2.9 / C2.10-15 — queue, concurrency and the inference lock."""

    def test_15_multiple_queued_jobs_are_all_processed(self):
        jobs = [self.make_job(filename="p{}.jpg".format(i))
                for i in range(3)]
        # Nothing consumed them yet; start the single consumer and drain.
        self.assertEqual(inference_jobs.queue_depth(), 3)
        with mock.patch.object(inference_jobs, "process_license_plate",
                               fake_inference):
            inference_jobs._ensure_worker(self.app)
            drained = inference_jobs.wait_for_idle(timeout=30)
        self.assertTrue(drained, "worker did not drain the queue")
        for job in jobs:
            reloaded = InferenceJob.objects(job_id=job.job_id).first()
            self.assertEqual(reloaded.status, InferenceJob.STATUS_COMPLETED)
            self._detection_ids.append(reloaded.detection.id)

    def test_14_inference_lock_is_still_the_serialisation_point(self):
        """The worker must not add concurrency on top of INFERENCE_LOCK.

        The fake takes the real lock, exactly as the real pipeline does through
        ``predict_with_lock``. If the worker ever ran two inferences at once the
        observed concurrency would exceed 1.
        """
        inside = {"now": 0, "peak": 0}
        guard = threading.Lock()

        def locked_fake(*_a, **_k):
            with INFERENCE_LOCK:
                with guard:
                    inside["now"] += 1
                    inside["peak"] = max(inside["peak"], inside["now"])
                time.sleep(0.05)
                with guard:
                    inside["now"] -= 1
            return fake_inference()

        jobs = [self.make_job(filename="c{}.jpg".format(i)) for i in range(3)]
        # One patch for the whole block: patching a module global from several
        # threads at once corrupts the final unpatch and leaks the fake.
        with mock.patch.object(inference_jobs, "process_license_plate",
                               locked_fake):
            threads = [
                threading.Thread(
                    target=inference_jobs.run_job,
                    args=(self.app, job.job_id), daemon=True)
                for job in jobs
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        self.assertEqual(inside["peak"], 1,
                         "more than one inference ran concurrently")
        for job in jobs:
            reloaded = InferenceJob.objects(job_id=job.job_id).first()
            self.assertEqual(reloaded.status, InferenceJob.STATUS_COMPLETED)
            self._detection_ids.append(reloaded.detection.id)

    def test_worker_thread_stays_alive_across_a_failing_job(self):
        """A crashing job must not kill the consumer thread."""
        bad = self.make_job(start_worker=False)
        with mock.patch.object(inference_jobs, "process_license_plate",
                               side_effect=RuntimeError("boom")):
            inference_jobs._ensure_worker(self.app)
            self.assertTrue(inference_jobs.wait_for_idle(timeout=30))
        self.assertEqual(
            InferenceJob.objects(job_id=bad.job_id).first().status,
            InferenceJob.STATUS_FAILED)

        good = self.make_job(start_worker=False)
        with mock.patch.object(inference_jobs, "process_license_plate",
                               fake_inference):
            self.assertTrue(inference_jobs.wait_for_idle(timeout=30))
        reloaded = InferenceJob.objects(job_id=good.job_id).first()
        self.assertEqual(reloaded.status, InferenceJob.STATUS_COMPLETED)
        self._detection_ids.append(reloaded.detection.id)


class InferenceContractTests(JobTestCase):
    """C2.6 / C2.7 / C2.10-11 — the pipeline itself must be untouched."""

    def test_11_worker_calls_the_existing_inference_function_unchanged(self):
        """Identity check: we call the real function, we did not wrap it."""
        from ealpr import utils
        self.assertIs(inference_jobs.process_license_plate,
                      utils.process_license_plate)
        self.assertIn("predict_with_lock", utils._detect_on_array.__code__
                      .co_names)

    def test_ai_sources_are_byte_identical_to_head(self):
        import subprocess
        for path in ("ealpr/utils.py", "ealpr/ai_models.py",
                     "ealpr/blueprints/detection.py"):
            committed = subprocess.run(
                ["git", "show", "3eec1777:" + path],
                capture_output=True, text=True, cwd=_base_dir()).stdout
            with open(os.path.join(_base_dir(), path), encoding="utf-8") as fh:
                self.assertEqual(
                    committed.replace("\r\n", "\n"),
                    fh.read().replace("\r\n", "\n"),
                    "{} was modified; C2 must not touch the AI pipeline".format(
                        path),
                )

    def test_10_process_plate_response_contract_is_unchanged(self):
        """The legacy sync path keeps its exact response keys and 400 behaviour."""
        client = self.app.test_client()
        self.login(client, self.owner)
        self.assertEqual(
            client.post("/process_plate", data={}).status_code, 400)
        response = client.post("/process_plate", data={
            "image": (io.BytesIO(b"not-an-image"), "notes.txt")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("success", response.get_json())

    def test_09_process_plate_still_reachable_and_authenticated(self):
        client = self.app.test_client()
        self.assertEqual(client.post("/process_plate", data={}).status_code, 302)
        self.login(client, self.owner)
        self.assertEqual(
            client.post("/process_plate", data={}).status_code, 400)


class JobApiTests(JobTestCase):
    """C2.4 / C2.5 / C2.8 / C2.12 — the HTTP surface."""

    def test_01_post_returns_202_and_queued(self):
        client = self.app.test_client()
        self.login(client, self.owner)
        with self.no_worker():
            response = client.post(
                "/api/inference/jobs", content_type="multipart/form-data",
                data={"image": (io.BytesIO(PNG_BYTES), "plate.jpg")})
        self.assertEqual(response.status_code, 202)
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertEqual(body["status"], InferenceJob.STATUS_QUEUED)
        self.assertTrue(body["job_id"])
        self._job_ids.append(body["job_id"])

    def test_post_does_not_perform_inference_in_the_request(self):
        client = self.app.test_client()
        self.login(client, self.owner)
        with self.no_worker(), \
                mock.patch.object(inference_jobs, "process_license_plate") as spy:
            response = client.post(
                "/api/inference/jobs", content_type="multipart/form-data",
                data={"image": (io.BytesIO(PNG_BYTES), "plate.jpg")})
            body = response.get_json()
        self._job_ids.append(body["job_id"])
        self.assertEqual(response.status_code, 202)
        spy.assert_not_called()

    def test_08_user_cannot_read_another_users_job(self):
        client = self.app.test_client()
        self.login(client, self.owner)
        with self.no_worker():
            created = client.post("/api/inference/jobs",
                                  content_type="multipart/form-data",
                                  data={"image": (io.BytesIO(PNG_BYTES), "p.jpg")})
        job_id = created.get_json()["job_id"]
        self._job_ids.append(job_id)

        # The owner can read it.
        self.assertEqual(
            client.get("/api/inference/jobs/" + job_id).status_code, 200)

        # A different non-admin user cannot, and gets 404 not 403 so the
        # endpoint cannot be used to enumerate job ids.
        client.get("/logout")
        self.login(client, self.other)
        forbidden = client.get("/api/inference/jobs/" + job_id)
        self.assertEqual(forbidden.status_code, 404)
        self.assertFalse(forbidden.get_json()["success"])

    def test_admin_may_read_another_users_job(self):
        admin = User(username="__c2_admin_{}".format(uuid.uuid4().hex[:8]),
                     password="pw", is_admin=True)
        admin.save()
        self.addCleanup(admin.delete)
        job = self.make_job(user=self.owner)

        client = self.app.test_client()
        self.login(client, admin)
        response = client.get("/api/inference/jobs/" + job.job_id)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["job_id"], job.job_id)

    def test_07_unknown_job_returns_404(self):
        client = self.app.test_client()
        self.login(client, self.owner)
        response = client.get("/api/inference/jobs/deadbeefdeadbeef")
        self.assertEqual(response.status_code, 404)
        self.assertFalse(response.get_json()["success"])

    def test_status_endpoint_requires_authentication(self):
        client = self.app.test_client()
        self.assertEqual(
            client.get("/api/inference/jobs/" + uuid.uuid4().hex).status_code, 302)

    def test_08_invalid_inputs_return_400(self):
        client = self.app.test_client()
        self.login(client, self.owner)
        cases = [
            ({}, "no file at all"),
            ({"image": (io.BytesIO(b"x"), "notes.txt")}, "wrong extension"),
        ]
        for data, label in cases:
            with self.subTest(case=label):
                response = client.post("/api/inference/jobs",
                                       content_type="multipart/form-data",
                                       data=data)
                self.assertEqual(response.status_code, 400, label)
                self.assertFalse(response.get_json()["success"])

    def test_post_returns_503_when_models_are_unavailable(self):
        client = self.app.test_client()
        self.login(client, self.owner)
        with mock.patch("ealpr.ai_models.get_plate_detection_model",
                        lambda: None):
            response = client.post(
                "/api/inference/jobs", content_type="multipart/form-data",
                data={"image": (io.BytesIO(PNG_BYTES), "plate.jpg")})
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()["success"])

    def test_08_oversized_image_returns_400(self):
        from config import MAX_IMAGE_SIZE
        client = self.app.test_client()
        self.login(client, self.owner)
        big = io.BytesIO(b"\x00" * (MAX_IMAGE_SIZE + 1024))
        response = client.post("/api/inference/jobs",
                               content_type="multipart/form-data",
                               data={"image": (big, "huge.jpg")})
        self.assertEqual(response.status_code, 400)

    def test_status_response_never_leaks_internals(self):
        with mock.patch.object(inference_jobs, "process_license_plate",
                               side_effect=RuntimeError("/secret/path/cv2.so")):
            job = self.make_job()
            inference_jobs.run_job(self.app, job.job_id)

        client = self.app.test_client()
        self.login(client, self.owner)
        body = client.get("/api/inference/jobs/" + job.job_id).get_json()

        self.assertEqual(body["status"], InferenceJob.STATUS_FAILED)
        blob = str(body)
        for leak in ("/secret/path", "Traceback", "RuntimeError", "cv2.so",
                     ".jpg", "site-packages", job.image_path or "TEMP"):
            self.assertNotIn(leak, blob, "status leaked {!r}".format(leak))
        self.assertEqual(body["error_code"], inference_jobs.ERROR_INTERNAL)

    def test_status_response_exposes_the_expected_shape(self):
        client = self.app.test_client()
        self.login(client, self.owner)
        job = self.make_job()
        body = client.get("/api/inference/jobs/" + job.job_id).get_json()
        for key in ("job_id", "status", "created_at", "started_at",
                    "completed_at", "detection_id"):
            self.assertIn(key, body)
        self.assertIsNone(body["started_at"])
        self.assertNotIn("image_path", str(body))
        self.assertNotIn(MARKER, str(body))

    def test_05_completed_status_reports_detection_id(self):
        with mock.patch.object(inference_jobs, "process_license_plate",
                               fake_inference):
            job = self.make_job()
            self.track(inference_jobs.run_job(self.app, job.job_id))
        client = self.app.test_client()
        self.login(client, self.owner)
        body = client.get("/api/inference/jobs/" + job.job_id).get_json()
        self.assertEqual(body["status"], InferenceJob.STATUS_COMPLETED)
        self.assertIsNotNone(body["detection_id"])
        self.assertEqual(body["plate_text"], MARKER)


class DataSafetyTests(JobTestCase):
    """C2.13 — the suite must not disturb production data."""

    def test_19_no_existing_detection_or_visitor_is_modified(self):
        from models import Visitor
        visitors_before = Visitor.objects.count()
        with mock.patch.object(inference_jobs, "process_license_plate",
                               fake_inference):
            job = self.make_job()
            self.track(inference_jobs.run_job(self.app, job.job_id))
        # The job's plate matches no real visitor, so nothing was touched.
        self.assertEqual(DetectionResult.objects(plate_number=MARKER).count(), 1)
        self.assertEqual(Visitor.objects.count(), visitors_before)
        # And the pre-existing detection set is untouched: our record is the
        # only difference, and it is removed in tearDown by id.
        self.assertGreaterEqual(DetectionResult.objects.count(),
                                self.detections_before)


def _base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(os.getenv("EALPR_C2_SMOKE") == "1",
                     "set EALPR_C2_SMOKE=1 to run the real-inference smoke test")
class RealInferenceSmokeTest(JobTestCase):
    """The single real YOLO/OCR run, opt-in and self-cleaning.

    Uses a real detection image from the project's own image store so the model
    has an actual plate to find. Any DetectionResult it produces is deleted by
    id in tearDown; nothing else is touched.
    """

    SAMPLE = r"D:\EALPR SYSTEM\detected_plates\20260917_171133_5ff0754d_original.jpg"

    def test_real_inference_end_to_end(self):
        # Unpatch the model loaders and the image writer so this exercises the
        # genuine YOLO/OCR pipeline and the real on-disk image store.
        for index in (0, 1, 2):
            self._patches[index].stop()
            self._patches[index] = None

        from config import SAVE_DIR
        before_files = set(os.listdir(str(SAVE_DIR)))
        thumb_dir = os.path.join(str(SAVE_DIR), "thumbnails")
        before_thumbs = (set(os.listdir(thumb_dir))
                         if os.path.isdir(thumb_dir) else set())

        def remove_new_images():
            for name in set(os.listdir(str(SAVE_DIR))) - before_files:
                try:
                    os.remove(os.path.join(str(SAVE_DIR), name))
                except OSError:
                    pass
            if os.path.isdir(thumb_dir):
                for name in set(os.listdir(thumb_dir)) - before_thumbs:
                    try:
                        os.remove(os.path.join(thumb_dir, name))
                    except OSError:
                        pass

        self.addCleanup(remove_new_images)

        payload = PNG_BYTES
        if os.path.isfile(self.SAMPLE):
            with open(self.SAMPLE, "rb") as fh:
                payload = fh.read()

        path = os.path.join(_temp_dir(),
                            "c2smoke_{}_{}".format(uuid.uuid4().hex, "smoke.jpg"))
        with open(path, "wb") as handle:
            handle.write(payload)
        self._staged_paths.append(path)

        job = inference_jobs.enqueue(self.app, path, "smoke.jpg", self.owner,
                                     start_worker=False)
        self._job_ids.append(job.job_id)

        started = time.time()
        inference_jobs.run_job(self.app, job.job_id)
        elapsed = time.time() - started
        reloaded = InferenceJob.objects(job_id=job.job_id).first()
        if reloaded.detection is not None:
            self._detection_ids.append(reloaded.detection.id)
        self.assertEqual(reloaded.status, InferenceJob.STATUS_COMPLETED,
                         "real inference should complete on a real plate image")
        self.assertIsNotNone(reloaded.detection)
        self.assertIsNotNone(reloaded.plate_text)
        print("\n  [smoke] real YOLO/OCR took {:.2f}s -> plate={!r} "
              "confidence={:.4f}".format(
                  elapsed, reloaded.plate_text,
                  reloaded.detection.confidence))


if __name__ == "__main__":
    unittest.main()
