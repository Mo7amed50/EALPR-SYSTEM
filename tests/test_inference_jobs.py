"""Phase 2 / Blocks C2 and C3 — asynchronous inference job tests.

Covers the 19 cases required by C2.10 and the 32 required by C3. Real YOLO/OCR
inference is replaced by a controlled fake in almost every test, so the suite is
fast and does not touch the GPU. Exactly one real-inference smoke test exists and
it is skipped unless EALPR_C2_SMOKE=1.

Data safety: this suite creates DetectionResult / InferenceJob / User documents,
but only with names prefixed ``__c2_test``/``__c3_test`` and a unique per-run
plate marker. Cleanup deletes exactly those documents by id — never a broad
collection delete. Detection and visitor counts are asserted unchanged on the
way out. C3 additionally writes and removes real files in TEMP_DIR, always
tracked by path and always removed by exact path.
"""

import io
import os
import subprocess
import sys
import threading
import time
import types
import unittest
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest import mock

import numpy as np

from tests.base import BASE_DIR, get_app

import models as models_module
from ealpr import inference_jobs
from ealpr.ai_models import INFERENCE_LOCK
from models import DetectionResult, InferenceJob, User

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)

MARKER = "__c2_test_{}".format(uuid.uuid4().hex[:10])

#: Statements Block C3.7 is allowed to add to ``ealpr/ai_models.py``. Anything
#: else appearing in that file fails the comparison, so a future edit cannot
#: quietly change model behaviour while "fixing the race".
C3_7_ALLOWED_ADDITIONS = frozenset({
    "_MODEL_LOAD_LOCK = threading.RLock()",
    "with _MODEL_LOAD_LOCK:",
})


def _base_dir():
    from tests.base import BASE_DIR as base
    return base


def _read_text(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read().replace("\r\n", "\n")


def _git_show(revision):
    return subprocess.run(
        ["git", "show", revision], capture_output=True, text=True,
        cwd=_base_dir(), check=True,
    ).stdout.replace("\r\n", "\n")


def _norm(line):
    """Collapse indentation so a re-indent is not read as a behaviour change."""
    return " ".join(line.split())


def _statements(text):
    """Every non-blank, non-comment line, normalised.

    Comments are excluded on purpose: C3 documents its reasoning heavily, and a
    prose change is not a behaviour change. The C3.7 comparison must therefore
    look at code only, while ``test_ai_sources_are_byte_identical_to_head``
    still guards the files C3 must not touch at all.
    """
    return {
        _norm(line) for line in text.splitlines()
        if _norm(line) and not line.lstrip().startswith("#")
    }


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
        self._quiesce_worker()
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
        self.jobs_before = InferenceJob.objects.count()
        self._model_loader_patches = [
            # The POST endpoint imports the loader from ealpr.ai_models inside
            # the request, so patch it there too. Without this the suite would
            # load real YOLO weights just to pass the availability gate.
            mock.patch("ealpr.ai_models.get_plate_detection_model",
                       lambda: object()),
            mock.patch("ealpr.ai_models.get_ocr_model", lambda: object()),
        ]
        self._patches = [
            mock.patch.object(inference_jobs, "get_plate_detection_model",
                              lambda: object()),
            mock.patch.object(inference_jobs, "get_ocr_model",
                              lambda: object()),
            mock.patch.object(inference_jobs, "save_detection_images",
                              self._fake_save_images),
        ] + self._model_loader_patches
        for p in self._patches:
            p.start()
        self.addCleanup(self._cleanup)

    def _fake_save_images(self, original_bytes, processed_bytes):
        return ("/tmp/{}_original.jpg".format(MARKER),
                "/tmp/{}_processed.jpg".format(MARKER))

    def _drain_queue(self):
        """Discard leftover job ids so queue-depth assertions are meaningful.

        Delegates to the module helper so the ``_queued_ids`` mirror stays in
        step; clearing the queue by hand would leave ids marked as queued and
        recovery would then refuse to re-queue them.
        """
        inference_jobs.drain_queue()

    def _live_workers(self):
        return [t for t in threading.enumerate()
                if t.name == "inference-worker" and t.is_alive()]

    def _quiesce_worker(self):
        """Stop any consumer and prove none survives into the next test.

        A worker left mid-job would keep running into the *following* test, where
        it can pick up that test's mocks and make an unrelated assertion fail in
        a way that reads like a product bug. That is exactly how C3's
        "recovery must not execute stale work" test was first made flaky.

        ``wait_for_idle`` is only a useful barrier while a consumer exists: when
        jobs were submitted with the worker suppressed (``no_worker``) the queue
        is never drained, ``_idle`` stays clear, and waiting on it would burn the
        full timeout every time for no benefit.
        """
        if self._live_workers():
            inference_jobs.wait_for_idle(timeout=60.0)
        inference_jobs.shutdown(timeout=60.0)
        self._drain_queue()
        leaked = self._live_workers()
        self.assertEqual(
            leaked, [],
            "a consumer thread outlived the test; refusing to run the next test "
            "against a polluted worker")

    def _cleanup(self):
        self._quiesce_worker()
        # Delete only what this test created, by id.
        if self._detection_ids:
            DetectionResult.objects(id__in=self._detection_ids).delete()
        if self._job_ids:
            InferenceJob.objects(job_id__in=self._job_ids).delete()
        # Belt and braces against the whole class of leak, not just one test: any
        # job owned by this test's two throwaway users is removed even if the
        # test forgot to record its id. A leaked job is not harmless - C3
        # recovery requeues queued jobs, so one left behind can be picked up by a
        # later test and make an unrelated assertion fail. (This exact leak is
        # what made C3's "recovery must not execute stale work" test flaky.)
        InferenceJob.objects(user__in=[self.owner, self.other]).delete()
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

        # The suite must be a no-op on the database. A mismatch means a test
        # created something it did not clean up, which would leak into later
        # runs (and did, before this assertion existed).
        self.assertEqual(
            InferenceJob.objects.count(), self.jobs_before,
            "this test leaked InferenceJob documents")
        self.assertEqual(
            DetectionResult.objects.count(), self.detections_before,
            "this test leaked DetectionResult documents")

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

    @contextmanager
    def real_model_loaders(self):
        """Restore the genuine ealpr.ai_models loaders for this block.

        The base harness replaces them with ``lambda: object()`` so the suite
        never loads real weights. A test that is specifically about model
        *initialisation* has to reach the real loader, or it would only be
        testing the stub. The patches are stopped for the block and restarted on
        exit so no later test is affected.
        """
        for patcher in self._model_loader_patches:
            patcher.stop()
        try:
            yield
        finally:
            for patcher in self._model_loader_patches:
                patcher.start()


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
        for path in ("ealpr/utils.py", "ealpr/blueprints/detection.py"):
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

    def test_ai_models_kept_every_existing_statement(self):
        """``ai_models.py`` is exempted from the byte check above for one reason.

        Block C3.7 adds a module-level lock so simultaneous first access cannot
        construct the models twice. That necessarily re-indents
        ``_load_yolo_models``'s body, so a byte comparison would fail even though
        not one statement changed. Rather than dropping the check, this one is
        stricter in the way that matters: the *set of statements* in the file must
        be identical to the C2 baseline apart from the two additions C3.7
        documents, so a behavioural edit cannot hide inside the exemption.
        """
        path = "ealpr/ai_models.py"
        committed = _git_show("a829096a:" + path)
        current = _read_text(os.path.join(_base_dir(), path))
        added = _statements(current) - _statements(committed)
        self.assertEqual(
            added, C3_7_ALLOWED_ADDITIONS,
            "ai_models.py gained statements beyond the documented C3.7 lock")
        self.assertEqual(
            _statements(committed) - _statements(current), set(),
            "ai_models.py lost statements; C3.7 must not remove behaviour")

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


# ═══════════════════════════════════════════════════════════════════════════
# Block C3 — job recovery and lifecycle hardening
# ═══════════════════════════════════════════════════════════════════════════

class RecoveryTestCase(JobTestCase):
    """Base for C3 recovery tests.

    Recovery is simulated *in process* by draining the in-memory queue - which is
    exactly what a restart does to it - and leaving the Mongo documents alone.
    That is the honest model of the failure: the queue is memory, the documents
    are durable.

    ``make_staged_job`` writes a file with the real ``inference_job_`` prefix
    rather than the harness's ``c2test_`` prefix, because artifact cleanup is
    ownership-based and would (correctly) refuse to touch a file that does not
    look like an async-job artifact.
    """

    def make_staged_job(self, status, created_delta_s, started_delta_s=None,
                        prefix=None, owner=None, track_file=True):
        """Create a job with a staged file, aged by the given offsets."""
        path = os.path.join(
            _temp_dir(),
            "{}{}_{}".format(prefix if prefix is not None else "inference_job_",
                             uuid.uuid4().hex, "c3.jpg"),
        )
        with open(path, "wb") as handle:
            handle.write(PNG_BYTES)
        if track_file:
            self._staged_paths.append(path)
        job = InferenceJob(
            job_id=uuid.uuid4().hex,
            user=owner if owner is not None else self.owner,
            status=status,
            image_path=path,
            original_filename="c3.jpg",
            created_at=datetime.utcnow() - timedelta(seconds=created_delta_s),
            started_at=(None if started_delta_s is None
                        else datetime.utcnow() - timedelta(seconds=started_delta_s)),
        )
        job.save()
        self._job_ids.append(job.job_id)
        return job

    def reload(self, job):
        return InferenceJob.objects(job_id=job.job_id).first()

    def run_recovery(self):
        return inference_jobs.recover_jobs()


class StartupRecoveryTests(RecoveryTestCase):
    """C3.1 / C3.2 — a restart must not strand valid queued work."""

    def test_c3_1_queued_job_survives_simulated_restart(self):
        """A restart empties the in-memory queue; the document must persist."""
        job = self.make_staged_job(InferenceJob.STATUS_QUEUED, created_delta_s=5)
        self._drain_queue()  # <- the restart
        self.assertEqual(inference_jobs.queue_depth(), 0)
        self.assertEqual(self.reload(job).status, InferenceJob.STATUS_QUEUED)

    def test_c3_2_queued_job_is_rediscovered(self):
        job = self.make_staged_job(InferenceJob.STATUS_QUEUED, created_delta_s=5)
        self._drain_queue()
        counts = self.run_recovery()
        self.assertEqual(counts["requeued"], 1)
        self.assertIn(job.job_id, inference_jobs.queued_ids())

    def test_c3_3_queued_job_is_re_enqueued_into_the_in_memory_queue(self):
        self.make_staged_job(InferenceJob.STATUS_QUEUED, created_delta_s=5)
        self._drain_queue()
        self.run_recovery()
        self.assertEqual(inference_jobs.queue_depth(), 1)

    def test_c3_4_recovered_job_reaches_processing_and_completed(self):
        job = self.make_staged_job(InferenceJob.STATUS_QUEUED, created_delta_s=5)
        self._drain_queue()
        with mock.patch.object(inference_jobs, "process_license_plate",
                               fake_inference):
            self.run_recovery()
            inference_jobs._ensure_worker(self.app)
            self.assertTrue(inference_jobs.wait_for_idle(timeout=30))
        reloaded = self.track(self.reload(job))
        self.assertEqual(reloaded.status, InferenceJob.STATUS_COMPLETED)
        self.assertIsNotNone(reloaded.started_at)
        self.assertIsNotNone(reloaded.detection)

    def test_c3_5_stale_queued_job_is_safely_failed_not_run(self):
        job = self.make_staged_job(
            InferenceJob.STATUS_QUEUED,
            created_delta_s=inference_jobs.QUEUED_STALE_SECONDS + 600)
        self._drain_queue()
        counts = self.run_recovery()
        self.assertEqual(counts["stale_queued"], 1)
        reloaded = self.reload(job)
        self.assertEqual(reloaded.status, InferenceJob.STATUS_FAILED)
        self.assertEqual(reloaded.error_code, inference_jobs.ERROR_STALE_QUEUED)
        self.assertIsNone(reloaded.detection)
        self.assertNotIn(job.job_id, inference_jobs.queued_ids(),
                         "an abandoned job must not be handed to the worker")

    def test_c3_queued_threshold_is_not_an_arbitrary_short_timeout(self):
        """A job that is merely *slow* must survive recovery.

        Guards the C3 requirement not to fail a healthy long-running job on a
        short arbitrary timeout: the default bounds have to be far wider than the
        worst inference C2 actually measured.
        """
        self.assertGreaterEqual(inference_jobs.PROCESSING_STALE_SECONDS, 300,
                                "processing bound must dwarf the measured ~4s job")
        self.assertGreaterEqual(inference_jobs.QUEUED_STALE_SECONDS, 900,
                                "queued bound must allow a real backlog")
        self.assertGreater(inference_jobs.QUEUED_STALE_SECONDS,
                           inference_jobs.PROCESSING_STALE_SECONDS)


class ProcessingRecoveryTests(RecoveryTestCase):
    """C3.3 — stale processing work is failed, never retried."""

    def test_c3_6_stale_processing_job_is_detected(self):
        self.make_staged_job(
            InferenceJob.STATUS_PROCESSING,
            created_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600,
            started_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600)
        counts = self.run_recovery()
        self.assertEqual(counts["stale_processing"], 1)

    def test_c3_7_stale_processing_job_becomes_failed(self):
        job = self.make_staged_job(
            InferenceJob.STATUS_PROCESSING,
            created_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600,
            started_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600)
        self.run_recovery()
        reloaded = self.reload(job)
        self.assertEqual(reloaded.status, InferenceJob.STATUS_FAILED)
        self.assertEqual(reloaded.error_code,
                         inference_jobs.ERROR_STALE_PROCESSING)
        self.assertIsNotNone(reloaded.completed_at)

    def test_c3_8_stale_processing_job_creates_no_duplicate_detection(self):
        """C3.3's central safety property: recovery must not re-run inference.

        A stale processing job may have died *after* persisting a detection and
        *before* completing the job document. Re-running would add a second
        detection for one image, so recovery must terminate it instead.
        """
        job = self.make_staged_job(
            InferenceJob.STATUS_PROCESSING,
            created_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600,
            started_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600)
        before = DetectionResult.objects.count()

        # Any attempt to actually execute the recovered work is a failure.
        with mock.patch.object(inference_jobs, "process_license_plate",
                               side_effect=fake_inference) as inference:
            self.run_recovery()
            inference_jobs._ensure_worker(self.app)
            self.assertTrue(inference_jobs.wait_for_idle(timeout=20))
            inference.assert_not_called()

        self.assertEqual(self.reload(job).status, InferenceJob.STATUS_FAILED)
        self.assertEqual(DetectionResult.objects.count(), before)
        self.assertIsNone(self.reload(job).detection)

    def test_c3_9_completed_jobs_are_untouched(self):
        job = self.make_staged_job(
            InferenceJob.STATUS_COMPLETED,
            created_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600,
            started_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600)
        job.completed_at = datetime.utcnow() - timedelta(seconds=30)
        job.plate_text = MARKER
        job.save()
        counts = self.run_recovery()
        self.assertEqual(counts["stale_processing"], 0)
        reloaded = self.reload(job)
        self.assertEqual(reloaded.status, InferenceJob.STATUS_COMPLETED)
        self.assertEqual(reloaded.plate_text, MARKER)
        self.assertIsNone(reloaded.error_code)

    def test_c3_10_failed_jobs_are_untouched(self):
        job = self.make_staged_job(
            InferenceJob.STATUS_FAILED,
            created_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600,
            started_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600)
        job.completed_at = datetime.utcnow() - timedelta(seconds=30)
        job.error_code = inference_jobs.ERROR_NO_PLATE
        job.save()
        before = self.reload(job).error_code
        self.run_recovery()
        reloaded = self.reload(job)
        self.assertEqual(reloaded.status, InferenceJob.STATUS_FAILED)
        self.assertEqual(reloaded.error_code, before,
                         "recovery must not overwrite an existing failure reason")

    def test_c3_queued_job_with_deleted_owner_is_not_requeued(self):
        """A job that cannot possibly be persisted must not be handed to the worker.

        ``_persist_detection`` writes ``processed_by=job.user``. If the user
        document is gone the job can never complete, so requeuing it would spend
        a full serialised YOLO/OCR inference and then fail on a write. Recovery
        fails it up front instead.
        """
        job = self.make_staged_job(InferenceJob.STATUS_QUEUED, created_delta_s=1)
        # Remove the owner *without* deleting the job, simulating a job orphaned
        # by a user deletion.
        self.owner.delete()
        with mock.patch.object(inference_jobs, "process_license_plate",
                               side_effect=fake_inference) as inference:
            counts = self.run_recovery()
            inference_jobs._ensure_worker(self.app)
            self.assertTrue(inference_jobs.wait_for_idle(timeout=20))
            inference.assert_not_called()

        self.assertEqual(counts["requeued"], 0)
        self.assertEqual(counts["unrunnable_queued"], 1)
        reloaded = self.reload(job)
        self.assertEqual(reloaded.status, InferenceJob.STATUS_FAILED)
        self.assertEqual(reloaded.error_code, inference_jobs.ERROR_STARTUP_RECOVERY)

    def test_c3_5b_every_stale_queued_job_is_failed_not_just_the_first(self):
        """Regression: the stale transition must be a bulk update.

        ``QuerySet.modify()`` is a find_one_and_update and touches exactly one
        document, which is right for the claim and wrong for recovery. With a
        single stale job the two are indistinguishable, so a one-job test passes
        either way - which is why this case exists. Before the fix, 3 stale jobs
        produced 1 failure and 2 jobs left hanging, i.e. the bug this block is
        meant to remove.
        """
        jobs = [self.make_staged_job(
            InferenceJob.STATUS_QUEUED,
            created_delta_s=inference_jobs.QUEUED_STALE_SECONDS + 600)
            for _ in range(3)]
        self._drain_queue()
        counts = self.run_recovery()
        self.assertEqual(counts["stale_queued"], 3,
                         "all three stale jobs must be failed, not one")
        for job in jobs:
            reloaded = self.reload(job)
            self.assertEqual(reloaded.status, InferenceJob.STATUS_FAILED)
            self.assertEqual(reloaded.error_code,
                             inference_jobs.ERROR_STALE_QUEUED)
            self.assertFalse(os.path.exists(reloaded.image_path))

    def test_c3_7b_every_stale_processing_job_is_failed(self):
        """Same bulk-update property for the processing state."""
        jobs = [self.make_staged_job(
            InferenceJob.STATUS_PROCESSING,
            created_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600,
            started_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600)
            for _ in range(2)]
        counts = self.run_recovery()
        self.assertEqual(counts["stale_processing"], 2)
        for job in jobs:
            self.assertEqual(self.reload(job).status,
                             InferenceJob.STATUS_FAILED)
            self.assertEqual(self.reload(job).error_code,
                             inference_jobs.ERROR_STALE_PROCESSING)

    def test_c3_healthy_processing_job_is_not_disturbed(self):
        """A job that started seconds ago is still running: leave it alone."""
        job = self.make_staged_job(
            InferenceJob.STATUS_PROCESSING, created_delta_s=5, started_delta_s=4)
        self.run_recovery()
        reloaded = self.reload(job)
        self.assertEqual(reloaded.status, InferenceJob.STATUS_PROCESSING)
        self.assertIsNone(reloaded.error_code)
        self.assertTrue(os.path.exists(reloaded.image_path),
                        "a live job's staged file must survive recovery")


class StartupIdempotencyTests(RecoveryTestCase):
    """C3.4 — initialisation is safe to call repeatedly."""

    def test_c3_11_initialization_called_twice(self):
        first = inference_jobs.initialize_inference_jobs(self.app, force=True)
        second = inference_jobs.initialize_inference_jobs(self.app, force=True)
        self.assertIn("requeued", first)
        self.assertIn("requeued", second)

    def test_c3_12_exactly_one_consumer_exists(self):
        """The invariant that matters: never a second inference thread.

        A job is left queued so recovery genuinely has work and really does start
        a consumer; asserting "<= 1" against zero workers would prove nothing.
        """
        self.make_staged_job(InferenceJob.STATUS_QUEUED, created_delta_s=1)
        self._drain_queue()
        for _ in range(5):
            inference_jobs.initialize_inference_jobs(self.app, force=True)
        try:
            alive = [t for t in threading.enumerate()
                     if t.name == "inference-worker" and t.is_alive()]
            self.assertEqual(len(alive), 1,
                             "startup must leave exactly one consumer, found {}"
                             .format(len(alive)))
            self.assertTrue(inference_jobs.worker_is_running())
        finally:
            inference_jobs.shutdown()

    def test_c3_12b_repeated_initialization_does_not_multiply_threads(self):
        """Same property, asserted as a growth bound rather than a snapshot."""
        self.make_staged_job(InferenceJob.STATUS_QUEUED, created_delta_s=1)
        self._drain_queue()
        inference_jobs.initialize_inference_jobs(self.app, force=True)
        first = inference_jobs.worker_is_running()
        for _ in range(10):
            inference_jobs.initialize_inference_jobs(self.app, force=True)
        self.assertEqual(first, inference_jobs.worker_is_running(),
                         "the consumer must persist, not be replaced")
        try:
            alive = [t for t in threading.enumerate()
                     if t.name == "inference-worker" and t.is_alive()]
            self.assertEqual(len(alive), 1)
        finally:
            inference_jobs.shutdown()

    def test_c3_13_no_uncontrolled_duplicate_queueing(self):
        """A job already in memory must not be pushed a second time."""
        job = self.make_staged_job(InferenceJob.STATUS_QUEUED, created_delta_s=5)
        self.run_recovery()
        self.assertEqual(inference_jobs.queue_depth(), 1)
        for _ in range(4):
            counts = self.run_recovery()
            self.assertEqual(counts["requeued"], 0)
            self.assertEqual(counts["skipped_already_queued"], 1)
        self.assertEqual(inference_jobs.queue_depth(), 1)
        self.assertEqual(inference_jobs.queued_ids(), {job.job_id})


class TempCleanupTests(RecoveryTestCase):
    """C3.5 — abandoned artifacts are removed by ownership, never by age."""

    def test_c3_14_active_job_artifact_is_preserved(self):
        job = self.make_staged_job(InferenceJob.STATUS_QUEUED, created_delta_s=1)
        self.run_recovery()
        self.assertTrue(os.path.exists(job.image_path))

    def test_c3_15_stale_queued_artifact_is_removed(self):
        job = self.make_staged_job(
            InferenceJob.STATUS_QUEUED,
            created_delta_s=inference_jobs.QUEUED_STALE_SECONDS + 600)
        self.assertTrue(os.path.exists(job.image_path))
        self.run_recovery()
        self.assertFalse(os.path.exists(job.image_path))

    def test_c3_16_stale_processing_artifact_is_removed(self):
        job = self.make_staged_job(
            InferenceJob.STATUS_PROCESSING,
            created_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600,
            started_delta_s=inference_jobs.PROCESSING_STALE_SECONDS + 600)
        self.run_recovery()
        self.assertFalse(os.path.exists(job.image_path))

    def test_c3_17_unrelated_temp_files_are_preserved(self):
        """The safety property. A blanket age-based sweep would fail this."""
        from config import TEMP_DIR
        old = (datetime.utcnow()
               - timedelta(seconds=inference_jobs.QUEUED_STALE_SECONDS * 10))
        # Two shapes of "unrelated": the synchronous endpoints' own upload_
        # prefix, and a file that is not an upload at all.
        unrelated = [
            os.path.join(str(TEMP_DIR),
                         "upload_{}_{}.jpg".format(uuid.uuid4().hex, "sync.jpg")),
            os.path.join(str(TEMP_DIR),
                         "notes_{}.txt".format(uuid.uuid4().hex[:8])),
            os.path.join(str(TEMP_DIR),
                         "inference_job_looking_but_not_ours_{}.jpg".format(
                             uuid.uuid4().hex[:8])),
        ]
        for path in unrelated:
            with open(path, "wb") as handle:
                handle.write(PNG_BYTES)
            old_ts = old.timestamp()
            os.utime(path, (old_ts, old_ts))  # ancient, yet must survive
            self._staged_paths.append(path)
        try:
            self.run_recovery()
            for path in unrelated:
                self.assertTrue(os.path.exists(path),
                                "recovery deleted an unrelated file: {}".format(
                                    os.path.basename(path)))
        finally:
            for path in unrelated:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

    def test_c3_18_missing_artifact_does_not_crash_recovery(self):
        """A job whose file was already removed must not abort the sweep."""
        gone = self.make_staged_job(
            InferenceJob.STATUS_QUEUED,
            created_delta_s=inference_jobs.QUEUED_STALE_SECONDS + 600)
        other = self.make_staged_job(InferenceJob.STATUS_QUEUED, created_delta_s=1)
        os.remove(gone.image_path)
        counts = self.run_recovery()  # must not raise
        self.assertEqual(counts["stale_queued"], 1)
        self.assertEqual(self.reload(gone).status, InferenceJob.STATUS_FAILED)
        self.assertEqual(self.reload(other).status, InferenceJob.STATUS_QUEUED,
                         "one bad job must not stop the others being recovered")

    def test_c3_cleanup_refuses_paths_without_the_job_prefix(self):
        """The guard itself, so a future refactor cannot widen it silently."""
        from config import TEMP_DIR
        victim = os.path.join(str(TEMP_DIR),
                              "upload_{}_sync.jpg".format(uuid.uuid4().hex))
        with open(victim, "wb") as handle:
            handle.write(PNG_BYTES)
        self._staged_paths.append(victim)
        self.assertFalse(inference_jobs.recover_staged_artifact(victim))
        self.assertFalse(inference_jobs.recover_staged_artifact(None))
        self.assertFalse(inference_jobs.recover_staged_artifact(""))
        self.assertTrue(os.path.exists(victim))

    def test_c3_orphan_inference_artifact_is_reclaimed(self):
        """A prefixed file with no owning job is provably abandoned."""
        from config import TEMP_DIR
        path = os.path.join(str(TEMP_DIR),
                            "inference_job_{}_orphan.jpg".format(uuid.uuid4().hex))
        with open(path, "wb") as handle:
            handle.write(PNG_BYTES)
        old = (datetime.utcnow()
               - timedelta(seconds=inference_jobs.QUEUED_STALE_SECONDS + 600))
        os.utime(path, (old.timestamp(), old.timestamp()))
        self._staged_paths.append(path)
        removed = inference_jobs.cleanup_orphan_artifacts()
        self.assertFalse(os.path.exists(path),
                         "an ownerless inference artifact should be reclaimed")
        self.assertGreaterEqual(removed, 1)

    def test_c3_recent_orphan_is_not_reclaimed(self):
        """The margin that protects a request mid-stage (file, no document yet)."""
        from config import TEMP_DIR
        path = os.path.join(str(TEMP_DIR),
                            "inference_job_{}_fresh.jpg".format(uuid.uuid4().hex))
        with open(path, "wb") as handle:
            handle.write(PNG_BYTES)
        self._staged_paths.append(path)
        try:
            inference_jobs.cleanup_orphan_artifacts()
            self.assertTrue(os.path.exists(path),
                            "a brand-new ownerless file may belong to a live request")
        finally:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


class RetentionTests(RecoveryTestCase):
    """C3.6 — bounded storage, terminal jobs only."""

    def test_c3_19_ttl_index_exists_on_completed_at(self):
        names = {}
        for index in InferenceJob._get_collection().list_indexes():
            names[index["name"]] = index
        self.assertIn("completed_at_1", names,
                      "the C3 TTL retention index must be created at startup")
        self.assertEqual(
            int(names["completed_at_1"].get("expireAfterSeconds", -1)),
            models_module.JOB_RETENTION_SECONDS)

    def test_c3_20_only_terminal_jobs_carry_a_ttl_timestamp(self):
        """The mechanism that makes 'terminal only' safe.

        MongoDB's TTL monitor ignores any document whose indexed field is not a
        date, so the restriction is enforced by which field each state writes -
        verified against this cluster before being relied on.
        """
        queued = self.make_staged_job(InferenceJob.STATUS_QUEUED, created_delta_s=1)
        processing = self.make_staged_job(InferenceJob.STATUS_PROCESSING,
                                          created_delta_s=1, started_delta_s=1)
        for job in (queued, processing):
            raw = InferenceJob._get_collection().find_one(
                {"job_id": job.job_id})
            self.assertIsNone(raw.get("completed_at"),
                              "{} must not carry a TTL-triggering value".format(
                                  job.status))

    def test_c3_21_queued_jobs_are_not_expired_by_retention(self):
        job = self.make_staged_job(
            InferenceJob.STATUS_QUEUED,
            created_delta_s=models_module.JOB_RETENTION_SECONDS + 100000)
        self.run_recovery()
        self.assertIsNotNone(self.reload(job),
                             "retention must never remove a queued job")

    def test_c3_22_processing_jobs_are_not_expired_by_retention(self):
        job = self.make_staged_job(
            InferenceJob.STATUS_PROCESSING,
            created_delta_s=models_module.JOB_RETENTION_SECONDS + 100000,
            started_delta_s=5)
        self.run_recovery()
        self.assertIsNotNone(self.reload(job),
                             "retention must never remove a processing job")

    def test_c3_23_detection_result_survives_job_cleanup(self):
        """Job retention is metadata-only; the detection is independent."""
        with mock.patch.object(inference_jobs, "process_license_plate",
                               fake_inference):
            job = self.make_staged_job(InferenceJob.STATUS_QUEUED,
                                       created_delta_s=1)
            inference_jobs.run_job(self.app, job.job_id)
        reloaded = self.track(self.reload(job))
        self.assertEqual(reloaded.status, InferenceJob.STATUS_COMPLETED)
        detection_id = reloaded.detection.id
        self.assertIsNotNone(detection_id)

        # Delete the job exactly as the TTL monitor eventually will.
        InferenceJob.objects(job_id=job.job_id).delete()
        self._job_ids.remove(job.job_id)

        survivor = DetectionResult.objects(id=detection_id).first()
        self.assertIsNotNone(survivor,
                             "deleting an InferenceJob must not delete its detection")
        self.assertEqual(survivor.plate_number, MARKER)
        self.assertIsNotNone(survivor.original_image_path)
        # The assertion is complete; now dispose of the record this test made.
        # It stays in _detection_ids so tearDown still cleans up if any of the
        # assertions above raised first.
        DetectionResult.objects(id=detection_id).delete()

    def test_c3_retention_is_configurable(self):
        self.assertEqual(models_module.JOB_RETENTION_SECONDS,
                         int(os.getenv("INFERENCE_JOB_RETENTION_SECONDS",
                                       60 * 60 * 24 * 30)))


class ModelInitRaceTests(RecoveryTestCase):
    """C3.7 — simultaneous first access must construct the models once."""

    def _isolate_models(self):
        """Reset the module globals, restoring the real ones on cleanup.

        The real YOLO objects are never discarded or replaced: they are saved and
        put back. And no weights are ever read, because ``_load_yolo_models`` is
        driven against a fake ``ultralytics`` module in every test that uses this.
        """
        ai = sys.modules["ealpr.ai_models"]
        saved = (ai.plate_detection_model, ai.ocr_model, ai._models_loaded)
        ai.plate_detection_model = None
        ai.ocr_model = None
        ai._models_loaded = False

        def restore():
            ai.plate_detection_model, ai.ocr_model, ai._models_loaded = saved

        self.addCleanup(restore)

    @contextmanager
    def fake_ultralytics(self):
        """Intercept ``from ultralytics import YOLO`` inside the loader."""
        class FakeYOLO:
            def __init__(self, path, tracker):
                self.path = path
                tracker.append(path)

        built = []
        module = types.ModuleType("ultralytics")
        module.YOLO = lambda path: FakeYOLO(path, built)

        def slow_yolo(path):
            # A real load takes seconds. Sleeping is what makes the race
            # observable: without it, threads serialise by luck and the test
            # would pass even with the bug reintroduced.
            time.sleep(0.05)
            return FakeYOLO(path, built)

        module.YOLO = slow_yolo
        with mock.patch.dict(sys.modules, {"ultralytics": module}):
            yield built

    @contextmanager
    def present_weights(self):
        with mock.patch.object(sys.modules["ealpr.ai_models"], "MODEL_PATH") as p1, \
                mock.patch.object(sys.modules["ealpr.ai_models"], "OCR_MODEL_PATH") as p2:
            p1.exists.return_value = True
            p2.exists.return_value = True
            yield

    def test_c3_24_concurrent_first_access_loads_exactly_once(self):
        """The headline C3.7 property: one construction, however many callers."""
        self._isolate_models()
        barrier = threading.Barrier(8)
        seen = []

        with self.real_model_loaders(), self.fake_ultralytics() as built, \
                self.present_weights():
            def first_call():
                barrier.wait(timeout=20)
                seen.append(sys.modules["ealpr.ai_models"].get_plate_detection_model())

            threads = [threading.Thread(target=first_call, daemon=True)
                       for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=40)

        # One load pass constructs exactly two models: plate + ocr. A
        # reintroduced race would produce 2 * 8.
        self.assertEqual(len(built), 2,
                         "models were constructed {} times, expected exactly "
                         "one pair".format(len(built)))
        self.assertEqual(len(seen), 8)

    def test_c3_25_all_callers_receive_the_same_model_objects(self):
        self._isolate_models()
        barrier = threading.Barrier(6)
        plates, ocrs = [], []

        with self.real_model_loaders(), self.fake_ultralytics(), \
                self.present_weights():
            def call():
                ai = sys.modules["ealpr.ai_models"]
                barrier.wait(timeout=20)
                plates.append(ai.get_plate_detection_model())
                ocrs.append(ai.get_ocr_model())

            threads = [threading.Thread(target=call, daemon=True)
                       for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=40)

        self.assertTrue(plates and all(plates), "every caller got a model")
        self.assertEqual(len({id(m) for m in plates}), 1,
                         "every caller must share one plate model instance")
        self.assertEqual(len({id(m) for m in ocrs}), 1,
                         "every caller must share one ocr model instance")

    def test_c3_26_model_init_lock_is_distinct_from_the_inference_lock(self):
        ai = sys.modules["ealpr.ai_models"]
        self.assertIsNot(ai._MODEL_LOAD_LOCK, INFERENCE_LOCK,
                         "initialisation must not share the inference lock")
        self.assertTrue(hasattr(ai, "_MODEL_LOAD_LOCK"))
        # INFERENCE_LOCK must still be an RLock guarding predict, unchanged.
        self.assertEqual(type(INFERENCE_LOCK).__name__, "RLock")

    def test_c3_26b_inference_lock_still_serialises_prediction(self):
        """The inference lock's behaviour is untouched by C3.7."""
        import ealpr.ai_models as ai
        overlap = []
        active = []
        guard = threading.Lock()

        class SlowModel:
            def predict(self, source, **kwargs):
                with guard:
                    active.append(1)
                    overlap.append(len(active))
                time.sleep(0.05)
                with guard:
                    active.pop()
                return ["prediction"]

        def call():
            ai.predict_with_lock(SlowModel(), "src")

        threads = [threading.Thread(target=call, daemon=True) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(max(overlap), 1,
                         "INFERENCE_LOCK must still allow one inference at a time")
        self.assertEqual(len(overlap), 5)


class C2RegressionUnderC3Tests(RecoveryTestCase):
    """C3.7 of the plan — the C2 guarantees must survive C3."""

    def test_c3_27_async_post_still_returns_202(self):
        with self.no_worker():
            client = self.app.test_client()
            self.login(client, self.owner)
            data = {"image": (io.BytesIO(PNG_BYTES), "plate.png")}
            response = client.post("/api/inference/jobs", data=data,
                                   content_type="multipart/form-data")
        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], InferenceJob.STATUS_QUEUED)
        self._job_ids.append(payload["job_id"])

    def test_c3_28_request_still_does_not_execute_inference(self):
        with self.no_worker():
            with mock.patch.object(inference_jobs, "process_license_plate",
                                   side_effect=fake_inference) as inference:
                client = self.app.test_client()
                self.login(client, self.owner)
                data = {"image": (io.BytesIO(PNG_BYTES), "plate.png")}
                client.post("/api/inference/jobs", data=data,
                            content_type="multipart/form-data")
                inference.assert_not_called()

    def test_c3_29_atomic_claim_still_prevents_duplicate_execution(self):
        """The C2 claim guarantee must survive C3.

        Two runners race for one job; exactly one may win the atomic claim, so
        exactly one DetectionResult may exist for it.
        """
        job = self.make_staged_job(InferenceJob.STATUS_QUEUED, created_delta_s=1)
        before = DetectionResult.objects.count()
        with mock.patch.object(inference_jobs, "process_license_plate",
                               fake_inference):
            inference_jobs._ensure_worker(self.app)
            t = threading.Thread(target=inference_jobs.run_job,
                                 args=(self.app, job.job_id), daemon=True)
            t.start()
            inference_jobs.run_job(self.app, job.job_id)
            t.join(timeout=30)

        reloaded = self.track(self.reload(job))
        self.assertEqual(reloaded.attempts, 1,
                         "exactly one runner may claim the job")
        self.assertEqual(reloaded.status, InferenceJob.STATUS_COMPLETED)
        self.assertEqual(DetectionResult.objects.count(), before + 1,
                         "a double claim would have produced a duplicate detection")

    def test_c3_30_ownership_is_still_enforced(self):
        job = self.make_staged_job(InferenceJob.STATUS_QUEUED, created_delta_s=1)
        client = self.app.test_client()
        self.login(client, self.other)
        response = client.get("/api/inference/jobs/{}".format(job.job_id))
        self.assertEqual(response.status_code, 404,
                         "a non-owner must get 404, never the job's data")

    def test_c3_31_recovery_errors_are_sanitised(self):
        job = self.make_staged_job(
            InferenceJob.STATUS_QUEUED,
            created_delta_s=inference_jobs.QUEUED_STALE_SECONDS + 600)
        self.run_recovery()
        client = self.app.test_client()
        self.login(client, self.owner)
        response = client.get("/api/inference/jobs/{}".format(job.job_id))
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertEqual(response.get_json()["error_code"],
                         inference_jobs.ERROR_STALE_QUEUED)
        for leak in ("Traceback", "site-packages", "RuntimeError", ".jpg",
                     "temp", job.image_path):
            self.assertNotIn(leak, body,
                             "recovery status leaked {!r}".format(leak))

    def test_c3_32_process_plate_response_contract_unchanged(self):
        client = self.app.test_client()
        self.login(client, self.owner)
        self.assertEqual(client.post("/process_plate", data={}).status_code, 400)
        bad = client.post(
            "/process_plate",
            data={"file": (io.BytesIO(b"not an image"), "notes.txt")},
            content_type="multipart/form-data")
        self.assertEqual(bad.status_code, 400)
        self.assertFalse(bad.get_json().get("success", False))

    def test_c3_recovery_never_raises_when_mongo_is_unavailable(self):
        """C3.10: a database outage degrades to 'skipped', not a crash."""
        with mock.patch.object(inference_jobs, "recover_jobs",
                               side_effect=RuntimeError("mongo down")):
            counts = inference_jobs.initialize_inference_jobs(self.app, force=True)
        self.assertTrue(counts.get("error") or counts.get("skipped"))

    def test_c3_recover_jobs_contains_unexpected_errors(self):
        """The sweep's own try/except, exercised directly."""
        with mock.patch.object(inference_jobs.InferenceJob, "objects",
                               side_effect=RuntimeError("mongo down")):
            counts = inference_jobs.recover_jobs()
        self.assertEqual(counts["errors"], 1)
        self.assertEqual(counts["requeued"], 0)

    def test_c3_ai_and_detection_sources_are_still_byte_identical(self):
        """C3 must not have touched the pipeline or the sync route."""
        sources = ("ealpr/utils.py", "ealpr/blueprints/detection.py",
                   "ealpr/image_storage.py", "ealpr/image_thumbnails.py")
        for relative in sources:
            current = _read_text(os.path.join(BASE_DIR, relative))
            baseline = _git_show("a829096a:" + relative)
            self.assertEqual(current, baseline,
                             "{} changed; C3 must not modify it".format(relative))


if __name__ == "__main__":
    unittest.main()