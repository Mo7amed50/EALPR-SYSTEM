"""
ealpr/repositories — data access layer.

Repositories own every MongoEngine query, projection, filter, count and
aggregation pipeline. Blueprints keep HTTP concerns only: request parsing,
authentication/authorization, rendering and response construction.

This is deliberately a thin, explicit layer — not a generic repository
framework. Each method maps to one concrete read (or write) a route performs,
and names the exact fields it needs.
"""

from ealpr.repositories.detection_repository import (
    GALLERY_ALLOWED_PAGE_SIZES,
    GALLERY_DEFAULT_PAGE_SIZE,
    GALLERY_LIST_FIELDS,
    DetectionRepository,
)
from ealpr.repositories.report_repository import ReportRepository
from ealpr.repositories.visitor_repository import VisitorRepository

__all__ = [
    "DetectionRepository",
    "GALLERY_ALLOWED_PAGE_SIZES",
    "GALLERY_DEFAULT_PAGE_SIZE",
    "GALLERY_LIST_FIELDS",
    "ReportRepository",
    "VisitorRepository",
]
