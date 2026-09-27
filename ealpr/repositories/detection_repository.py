"""
ealpr/repositories/detection_repository.py — all DetectionResult read access.

Owns the queries behind the gallery, the detection-history table, the single
detection endpoints, the activity feed and the detection export.

Projection rules that must not regress
--------------------------------------
The collection mixes two image storage layouts: 60 records hold images inline
in ``original_image`` / ``processed_image`` BinaryFields (~103 KB + ~15 KB each)
and 6 records reference files by path. Every method that returns a *list* or a
*metadata view* therefore names the fields it needs and excludes both binary
fields, so listing never transfers image payloads. Only
:meth:`DetectionRepository.get_image_source` is allowed to read a binary field,
because it serves that image.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from bson.objectid import ObjectId
from mongoengine import Q

from models import DetectionResult

# ── Gallery (Block A) ─────────────────────────────────────────────────────────
# Allowed page sizes and the default are part of the Block A public behaviour
# and are re-exported by ealpr.blueprints.detection for compatibility.
GALLERY_ALLOWED_PAGE_SIZES = (20, 50, 100)
GALLERY_DEFAULT_PAGE_SIZE = 20

#: Fields needed to render one gallery card. Image payloads are excluded on
#: purpose; the card's <img> points at /api/images/<id>/processed/thumb.
GALLERY_LIST_FIELDS = (
    "timestamp", "plate_number", "confidence", "status",
    "visitor_name", "original_image_path", "processed_image_path",
)

# ── Detection history table ──────────────────────────────────────────────────
HISTORY_PAGE_SIZE = 10

#: detection_history.html reads only these four attributes (verified against the
#: template). The previous query used no projection at all, so each of the 10
#: rows dragged both BinaryFields across the wire to render a text table.
HISTORY_LIST_FIELDS = ("timestamp", "plate_number", "status", "visitor_name")

#: /api/detection/<id> response fields (Block B already projected these).
DETAIL_FIELDS = (
    "timestamp", "plate_number", "confidence", "status",
    "visitor_name", "processed_by",
)

#: /api/images/<id> metadata response fields (Block B already projected these).
IMAGE_META_FIELDS = ("plate_number", "status", "timestamp", "visitor_name")

#: /api/activities builds {plate_number, status, timestamp} per entry.
ACTIVITIES_FIELDS = ("timestamp", "plate_number", "status")

#: CSV/Excel export columns.
EXPORT_FIELDS = (
    "timestamp", "plate_number", "status", "confidence",
    "visitor_name", "processed_by",
)


class GalleryPage(NamedTuple):
    """One page of gallery results plus the counters the template needs."""

    detections: List[DetectionResult]
    total: int
    page: int
    per_page: int
    total_pages: int


class HistoryPage(NamedTuple):
    """One page of detection-history rows."""

    detections: List[DetectionResult]
    total: int
    page: int
    per_page: int
    total_pages: int


class DetectionRepository:
    """Read access to ``DetectionResult``.

    The class holds no state; every method is a classmethod so blueprints can
    call it without an instance.
    """

    # ── Shared filter construction ──────────────────────────────────────────

    @staticmethod
    def _build_filter(
        search: str = "",
        status: str = "all",
        date_filter: str = "all",
        now: Optional[datetime] = None,
    ) -> Q:
        """Translate plain request values into the gallery/history ``Q``.

        ``now`` is injected rather than read from the clock so a filter set is
        reproducible and a single request uses one consistent instant.
        """
        query = Q()
        if search:
            query &= Q(plate_number__icontains=search)
        if status in ("authorized", "unauthorized"):
            query &= Q(status=status)
        if date_filter in ("today", "week", "month"):
            now = now or datetime.utcnow()
            if date_filter == "today":
                query &= Q(timestamp__gte=now.replace(
                    hour=0, minute=0, second=0, microsecond=0))
            elif date_filter == "week":
                query &= Q(timestamp__gte=now - timedelta(days=7))
            else:
                query &= Q(timestamp__gte=now - timedelta(days=30))
        return query

    @staticmethod
    def _clamp_page(page: int, total_pages: int) -> int:
        """Clamp an out-of-range page to the last page (Block A behaviour)."""
        if page > total_pages:
            return total_pages
        return page

    # ── Counts ──────────────────────────────────────────────────────────────

    @classmethod
    def count_all(cls) -> int:
        """Total detections in the collection (dashboard KPI)."""
        return DetectionResult.objects.count()

    @classmethod
    def count_gallery(
        cls,
        search: str = "",
        status: str = "all",
        date_filter: str = "all",
        now: Optional[datetime] = None,
    ) -> int:
        """Number of detections matching the gallery filter set."""
        return DetectionResult.objects(
            cls._build_filter(search, status, date_filter, now)
        ).count()

    # ── Gallery ─────────────────────────────────────────────────────────────

    @classmethod
    def gallery_page(
        cls,
        page: int,
        per_page: int,
        search: str = "",
        status: str = "all",
        date_filter: str = "all",
        now: Optional[datetime] = None,
    ) -> GalleryPage:
        """Return one gallery page: a filtered count plus the page's records.

        Ordering is ``-timestamp, -id`` so records sharing a timestamp cannot
        shift between pages, and the binary image fields are excluded.
        """
        query = cls._build_filter(search, status, date_filter, now)
        total = DetectionResult.objects(query).count()
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = cls._clamp_page(page, total_pages)
        detections = list(
            DetectionResult.objects(query)
            .only(*GALLERY_LIST_FIELDS)
            .order_by("-timestamp", "-id")
            .skip((page - 1) * per_page)
            .limit(per_page)
        )
        return GalleryPage(detections, total, page, per_page, total_pages)

    # ── Detection history table ─────────────────────────────────────────────

    @classmethod
    def history_page(
        cls,
        page: int,
        per_page: int = HISTORY_PAGE_SIZE,
    ) -> HistoryPage:
        """Return one page of the history table, without any image payload.

        Block A paginated this route but left the query unprojected, so ten
        rows pulled roughly 1.2 MB of inline images to render ten text cells.
        """
        total = DetectionResult.objects.count()
        total_pages = (total + per_page - 1) // per_page
        detections = list(
            DetectionResult.objects
            .only(*HISTORY_LIST_FIELDS)
            .order_by("-timestamp")
            .skip((page - 1) * per_page)
            .limit(per_page)
        )
        return HistoryPage(detections, total, page, per_page, total_pages)

    # ── Single record ───────────────────────────────────────────────────────

    @classmethod
    def get_detail(cls, detection_id: str) -> Optional[DetectionResult]:
        """The fields ``/api/detection/<id>`` returns; no image data."""
        if not ObjectId.is_valid(detection_id):
            return None
        return DetectionResult.objects(id=ObjectId(detection_id)).only(
            *DETAIL_FIELDS
        ).first()

    @classmethod
    def get_image_metadata(cls, detection_id: str) -> Optional[DetectionResult]:
        """The fields ``/api/images/<id>`` returns; no image data."""
        if not ObjectId.is_valid(detection_id):
            return None
        return DetectionResult.objects(id=ObjectId(detection_id)).only(
            *IMAGE_META_FIELDS
        ).first()

    @classmethod
    def get_image_source(
        cls, detection_id: str, img_type: str
    ) -> Optional[DetectionResult]:
        """The one query that may read a binary field, because it serves it.

        Only the requested image type's path and payload are projected, so an
        ``original`` request never also drags the ``processed`` image across.
        """
        if not ObjectId.is_valid(detection_id):
            return None
        if img_type == "original":
            fields = ("original_image_path", "original_image")
        else:
            fields = ("processed_image_path", "processed_image")
        return DetectionResult.objects(id=ObjectId(detection_id)).only(
            *fields
        ).first()

    # ── Activity feed ───────────────────────────────────────────────────────

    @classmethod
    def list_recent(cls, limit: int = 10) -> List[DetectionResult]:
        """Most recent detections for the activity feed.

        Only three scalar fields are read; the feed previously loaded whole
        documents, image binaries included.
        """
        return list(
            DetectionResult.objects
            .only(*ACTIVITIES_FIELDS)
            .order_by("-timestamp")
            .limit(limit)
        )

    # ── Export ──────────────────────────────────────────────────────────────

    @classmethod
    def list_for_export(
        cls,
        status_filter: str = "all",
        date_filter: str = "all",
        now: Optional[datetime] = None,
    ) -> Tuple[List[DetectionResult], int]:
        """Filtered detections for the CSV/Excel export, plus their count.

        Projected to the six exported columns, so the export no longer reads
        either image binary for every row it writes.
        """
        query = DetectionResult.objects
        if status_filter != "all":
            query = query.filter(status=status_filter)
        if date_filter != "all":
            now = now or datetime.utcnow()
            if date_filter == "today":
                query = query.filter(timestamp__gte=now.replace(
                    hour=0, minute=0, second=0))
            elif date_filter == "week":
                query = query.filter(timestamp__gte=now - timedelta(days=7))
            elif date_filter == "month":
                query = query.filter(timestamp__gte=now - timedelta(days=30))

        results = list(
            query.only(*EXPORT_FIELDS).order_by("-timestamp")
        )
        return results, len(results)
