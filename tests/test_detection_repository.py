"""DetectionRepository tests.

The collection mixes two image storage layouts: 60 records hold images inline
in ``original_image`` / ``processed_image`` BinaryFields (~103 KB + ~15 KB
each) and 6 records reference files by path. The central invariant is that
listing and metadata views never read a binary field, and only
``get_image_source`` ever does.
"""

from datetime import datetime

from ealpr.repositories import DetectionRepository
from models import DetectionResult
from tests.base import RepositoryTestCase, counted

BINARY_FIELDS = ("original_image", "processed_image")


class BinaryProjectionTests(RepositoryTestCase):
    """No list or metadata query may request an image payload."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.total = DetectionResult.objects.count()
        if cls.total == 0:
            raise AssertionError("DetectionResult collection is empty")
        cls.binary_id = next(
            (
                doc["_id"]
                for doc in DetectionResult._get_collection().find(
                    {"$or": [
                        {"original_image": {"$exists": True}},
                        {"processed_image": {"$exists": True}},
                    ]},
                    {"_id": 1},
                )
            ),
            None,
        )
        if cls.binary_id is None:
            raise AssertionError("no record holds inline image binaries to test")

    def test_history_page_never_reads_images(self):
        with counted() as spy:
            page = DetectionRepository.history_page(page=1)
        self.assertEqual(len(page.detections), min(10, self.total))
        for projection in spy.projections:
            for field in BINARY_FIELDS:
                self.assertNotIn(field, projection)
                self.assertNotIn(field, projection.get("exclude", {}))

    def test_gallery_page_never_reads_images(self):
        with counted() as spy:
            DetectionRepository.gallery_page(page=1, per_page=20)
        for projection in spy.projections:
            for field in BINARY_FIELDS:
                self.assertNotIn(field, projection)
                self.assertNotIn(field, projection.get("exclude", {}))

    def test_activity_feed_never_reads_images(self):
        with counted() as spy:
            DetectionRepository.list_recent(limit=10)
        for projection in spy.projections:
            for field in BINARY_FIELDS:
                self.assertNotIn(field, projection)

    def test_detail_and_metadata_never_read_images(self):
        target = str(self.binary_id)
        with counted() as spy:
            DetectionRepository.get_detail(target)
            DetectionRepository.get_image_metadata(target)
        for projection in spy.projections:
            for field in BINARY_FIELDS:
                self.assertNotIn(field, projection)

    def test_export_never_reads_images(self):
        with counted() as spy:
            DetectionRepository.list_for_export()
        for projection in spy.projections:
            for field in BINARY_FIELDS:
                self.assertNotIn(field, projection)

    def test_only_image_source_reads_a_binary(self):
        """The one permitted exception, and only the requested type."""
        target = str(self.binary_id)
        with counted() as spy:
            DetectionRepository.get_image_source(target, "original")
        self.assertTrue(
            any("original_image" in p for p in spy.projections),
            "get_image_source must actually fetch the image it serves",
        )
        for projection in spy.projections:
            self.assertNotIn(
                "processed_image", projection,
                "an 'original' request must not drag the processed image across",
            )


class PaginationTests(RepositoryTestCase):
    """Block A behaviour that must not drift."""

    def test_history_pagination_covers_every_record_exactly_once(self):
        seen = []
        page = 1
        while True:
            result = DetectionRepository.history_page(page=page)
            if not result.detections:
                break
            seen.extend(str(d.id) for d in result.detections)
            if page >= result.total_pages:
                break
            page += 1
        self.assertEqual(len(seen), result.total)
        self.assertEqual(len(set(seen)), len(seen), "a record appeared on two pages")

    def test_gallery_clamps_an_out_of_range_page(self):
        result = DetectionRepository.gallery_page(page=10_000, per_page=20)
        self.assertEqual(result.page, result.total_pages)
        self.assertTrue(result.detections)

    def test_gallery_page_size_validation_stays_within_block_a(self):
        result = DetectionRepository.gallery_page(page=1, per_page=50)
        self.assertEqual(result.per_page, 50)
        self.assertLessEqual(len(result.detections), 50)

    def test_gallery_status_filter(self):
        result = DetectionRepository.gallery_page(
            page=1, per_page=100, status="unauthorized"
        )
        for row in result.detections:
            self.assertEqual(row.status, "unauthorized")

    def test_gallery_date_filter_today(self):
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        result = DetectionRepository.gallery_page(
            page=1, per_page=100, date_filter="today", now=datetime.utcnow()
        )
        for row in result.detections:
            self.assertGreaterEqual(row.timestamp, today)

    def test_history_uses_a_single_find(self):
        with counted() as spy:
            DetectionRepository.history_page(page=1)
        self.assertEqual(
            spy.commands.get("find", 0), 1,
            f"expected one find plus a count, saw {dict(spy.commands)}",
        )

    def test_invalid_ids_return_none(self):
        self.assertIsNone(DetectionRepository.get_detail("not-an-objectid"))
        self.assertIsNone(DetectionRepository.get_image_metadata("not-an-objectid"))
        self.assertIsNone(
            DetectionRepository.get_image_source("not-an-objectid", "original")
        )

    def test_export_count_matches_rows(self):
        rows, total = DetectionRepository.list_for_export()
        self.assertEqual(total, len(rows))
        self.assertEqual(total, DetectionResult.objects.count())

    def test_count_all(self):
        self.assertEqual(DetectionRepository.count_all(), DetectionResult.objects.count())
