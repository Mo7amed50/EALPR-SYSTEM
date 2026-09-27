"""
ealpr/repositories/visitor_repository.py — all Visitor read access.

Two rules this module exists to enforce:

1. **List reads project.** ``/visitors`` and ``/manage_visitors`` render tables
   and used to pull every field of all 1023 visitors, including ``id_number``
   (a national ID the pages never render) and both department strings.

2. **Write lookups stay unprojected on purpose.** ``_find_visitor`` resolves a
   visitor for routes that then mutate and ``save()`` the document
   (authorize / unauthorize / update / delete). A projection there would make
   MongoEngine write back a partial document and silently erase the fields that
   were not fetched. :meth:`VisitorRepository.find_for_update` therefore loads
   the complete document and says so; :meth:`find_by_identifier` is the
   read-only, projected variant.
"""

from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from bson.objectid import ObjectId
from mongoengine import Q

from models import Visitor

#: Columns rendered by templates/visitors.html.
READONLY_LIST_FIELDS = (
    "visitor_id", "visitor_code", "name", "license_plate",
    "entry_date", "entry_time", "exit_time", "authorized",
)

#: Columns rendered by templates/manage_visitors.html.
MANAGE_LIST_FIELDS = (
    "visitor_id", "visitor_code", "name", "license_plate",
    "entry_date", "entry_time", "responsible_department",
    "general_department", "status",
)

#: Columns returned by GET /api/visitors/<id>.
DETAIL_FIELDS = (
    "visitor_id", "visitor_code", "name", "license_plate",
    "entry_date", "entry_time", "exit_time", "authorized", "status",
    "responsible_department", "general_department",
)

#: Columns returned by GET /api/search/plate.
SEARCH_MATCH_FIELDS = ("name", "license_plate", "authorized")

#: Columns written into the styled daily / weekly / monthly Excel report.
PERIOD_REPORT_FIELDS = (
    "visitor_id", "visitor_code", "name", "license_plate",
    "entry_datetime_utc", "status", "responsible_department",
)


class DashboardCounts(NamedTuple):
    """The four dashboard KPIs.

    ``authorized`` / ``unauthorized`` count the ``authorized`` boolean, which is
    deliberately independent of the ``status`` string: on the live database
    these disagree (812 authorized=True vs 811 status='authorized'). Merging
    them would silently change a number, so the pipeline keeps them separate.
    """

    authorized: int
    unauthorized: int
    active: int


class VisitorRepository:
    """Read access to ``Visitor``. Stateless; all methods are classmethods."""

    # ── Visitor lists ───────────────────────────────────────────────────────

    @classmethod
    def list_readonly(cls) -> List[Visitor]:
        """Every visitor, projected to the read-only list columns.

        Still unpaginated: ``visitors.html`` renders one row per visitor and has
        no pagination control, so paginating here would change the page. See
        the "Remaining Block C work" section of the C1 report.
        """
        return list(Visitor.objects.only(*READONLY_LIST_FIELDS))

    @classmethod
    def search_manage(cls, search_query: str = "") -> List[Visitor]:
        """Newest-first management list, optionally filtered by free text."""
        query = Visitor.objects
        if search_query:
            query = query.filter(
                Q(name__icontains=search_query)
                | Q(visitor_code__icontains=search_query)
                | Q(license_plate__icontains=search_query)
            )
        return list(
            query.only(*MANAGE_LIST_FIELDS).order_by("-entry_datetime_utc")
        )

    # ── Lookups ─────────────────────────────────────────────────────────────

    @classmethod
    def find_by_identifier(
        cls, visitor_id: str, fields: Tuple[str, ...] = DETAIL_FIELDS
    ) -> Optional[Visitor]:
        """Read-only resolution by sequence id, ObjectId, code or plate.

        The four tiers are resolved in one round trip instead of up to four.
        A single ``$or`` query cannot be ranked the way the original cascade
        was, so the candidates come back together and are re-ranked in the same
        precedence the original used: numeric ``visitor_id`` first, then a valid
        ObjectId, then ``visitor_code``, then ``license_plate``.
        """
        if not visitor_id:
            return None
        visitor_id = str(visitor_id).strip()

        tiers: List[Tuple[str, Any]] = []
        if visitor_id.isdigit():
            tiers.append(("visitor_id", int(visitor_id)))
        if ObjectId.is_valid(visitor_id):
            tiers.append(("id", ObjectId(visitor_id)))
        tiers.append(("visitor_code", visitor_id))
        tiers.append(("license_plate", visitor_id))

        criteria = Q()
        for field, value in tiers:
            criteria |= Q(**{field: value})
        # The tier keys must themselves be loaded, otherwise the re-ranking
        # below cannot see which candidate matched which tier.
        projection = tuple(dict.fromkeys(tuple(fields) + tuple(f for f, _ in tiers)))
        candidates = list(Visitor.objects(criteria).only(*projection))
        if not candidates:
            return None
        for field, value in tiers:
            for doc in candidates:
                # MongoEngine documents expose fields as attributes, not dict
                # keys; "id" resolves to the primary-key property.
                if getattr(doc, field, None) == value:
                    return doc
        return candidates[0]

    @classmethod
    def find_for_update(cls, visitor_id: str) -> Optional[Visitor]:
        """Unprojected resolution for routes that will ``save()`` the document.

        Authorize, unauthorize, update and delete all mutate and re-save the
        visitor. Loading a projection here would make MongoEngine persist a
        partial document and drop the fields that were never fetched, so this
        method deliberately returns the complete document.
        """
        if not visitor_id:
            return None
        visitor_id = str(visitor_id).strip()

        tiers: List[Tuple[str, Any]] = []
        if visitor_id.isdigit():
            tiers.append(("visitor_id", int(visitor_id)))
        if ObjectId.is_valid(visitor_id):
            tiers.append(("id", ObjectId(visitor_id)))
        tiers.append(("visitor_code", visitor_id))
        tiers.append(("license_plate", visitor_id))

        for field, value in tiers:
            found = Visitor.objects(**{field: value}).first()
            if found:
                return found
        return None

    @classmethod
    def find_by_plate(cls, plate: str) -> Optional[Visitor]:
        """Exact plate lookup used by the detection-detail modal."""
        if not plate:
            return None
        return Visitor.objects(license_plate=plate).only(*DETAIL_FIELDS).first()

    @classmethod
    def find_duplicate(
        cls, license_plate: str, visitor_code: Optional[str] = None
    ) -> Optional[Visitor]:
        """Existing visitor with this plate, or this code when one is given."""
        query = Q(license_plate=license_plate)
        if visitor_code:
            query = Q(visitor_code=visitor_code) | query
        return Visitor.objects(query).first()

    @classmethod
    def search_plate_or_name(cls, query_text: str) -> Optional[Visitor]:
        """First visitor whose plate or name contains ``query_text``."""
        if not query_text:
            return None
        return Visitor.objects(
            Q(license_plate__icontains=query_text)
            | Q(name__icontains=query_text)
        ).only(*SEARCH_MATCH_FIELDS).first()

    @classmethod
    def list_for_period_report(
        cls, start_date: datetime, end_date: datetime
    ) -> List[Visitor]:
        """Visitors entering ``[start_date, end_date]``, for the Excel reports.

        Projected to the seven columns those workbooks emit. Ordering is
        unchanged (``entry_datetime_utc`` ascending) so the printed rows match
        the previous export.
        """
        return list(
            Visitor.objects(
                entry_datetime_utc__gte=start_date,
                entry_datetime_utc__lte=end_date,
            ).only(*PERIOD_REPORT_FIELDS).order_by("entry_datetime_utc")
        )

    # ── Aggregates ──────────────────────────────────────────────────────────

    @classmethod
    def dashboard_counts(cls) -> DashboardCounts:
        """Authorized / unauthorized / active visitor counts in one query.

        Replaces three separate ``count()`` calls (three ``aggregate``
        round trips) with a single ``$group``. Semantics are unchanged:
        ``authorized`` counts ``authorized == true`` only, ``unauthorized``
        counts ``authorized == false`` only, and a document with the field
        absent or null counts in neither, exactly as the original queries did.

        ``active`` needs care. ``exit_time`` is *absent* rather than null on
        every still-visiting record (317 of 1023 on the live data), and in an
        aggregation expression a missing field never compares equal to null, so
        the obvious ``$in: [null, ""]`` silently yields 0. Counting "the field
        is not a date" matches what the original ``exit_time__in=[None, ""]``
        resolved to, and also covers a null or empty-string value if one ever
        appears.
        """
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "authorized": {
                        "$sum": {"$cond": [{"$eq": ["$authorized", True]}, 1, 0]}
                    },
                    "unauthorized": {
                        "$sum": {"$cond": [{"$eq": ["$authorized", False]}, 1, 0]}
                    },
                    "active": {
                        "$sum": {
                            "$cond": [
                                {"$ne": [{"$type": "$exit_time"}, "date"]}, 1, 0
                            ]
                        }
                    },
                }
            }
        ]
        rows = list(Visitor._get_collection().aggregate(pipeline))
        if not rows:
            return DashboardCounts(0, 0, 0)
        row = rows[0]
        return DashboardCounts(
            int(row.get("authorized") or 0),
            int(row.get("unauthorized") or 0),
            int(row.get("active") or 0),
        )
