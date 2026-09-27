"""
ealpr/repositories/report_repository.py — visitor report aggregation.

``/api/analytics/visitor-stats`` and ``/api/analytics/export-excel`` both used
to pull every visitor document into Python and aggregate there: 1023 documents,
then an O(n x 7) pass for the daily breakdown, an O(n x 24) pass for the peak
hours, and a full Python loop for departments. The aggregation runs in MongoDB
here instead, in a single round trip via ``$facet``.

Semantic equivalence
--------------------
Each step below records exactly what the Python did, because several details
are load-bearing and a "cleaner" pipeline would silently change a number.

* **Daily buckets** matched ``entry_datetime_utc.strftime("%Y-%m-%d")`` against
  ``(now - i days).strftime("%Y-%m-%d")``. Both are UTC calendar dates, so
  ``$dateToString {format: "%Y-%m-%d"}`` over the raw UTC datetime is
  equivalent. Documents with a null/missing ``entry_datetime_utc`` were skipped
  by the Python truthiness test and are excluded by the ``$match``.
* **Days with no visitors still produce a row.** The Python loop always
  appended all 7 (or 30) days, including zero counts, so the missing days are
  filled in here rather than left to the caller to invent.
* **Peak hours are an OR of two different clocks.** A visitor counted towards
  hour *H* when *either* the local ``entry_time`` string began with ``"H:"``
  *or* the raw UTC hour equalled *H*. Those two can disagree, so a visitor can
  land in two different buckets. Both branches are reproduced, and the two
  branches are unioned rather than concatenated so a visitor matching the same
  hour on both clocks is still counted once — which is what the Python
  ``or`` did.
* **Departments skip blank values** (``if dept:``), so null, missing and empty
  are all excluded.
* The Python department loop incremented a status bucket only when the status
  string was one of the dict's keys. The only statuses the application can
  store are ``pending`` / ``authorized`` / ``unauthorized`` (enforced by
  ``add_visitor`` and ``update_visitor``) and the live collection contains
  exactly those three, so the three-key form is exact. Had a fourth status
  existed the original would have ignored it, and so does this pipeline.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Tuple

from models import Visitor

_STATUSES = ("authorized", "unauthorized", "pending")


class StatusSummary(NamedTuple):
    """Whole-collection visitor totals."""

    total: int
    authorized: int
    unauthorized: int
    pending: int


class DailyRow(NamedTuple):
    """One calendar day of the daily breakdown."""

    date: str          # "YYYY-MM-DD"
    day: str           # "Mon"
    total: int
    authorized: int
    unauthorized: int
    pending: int


class PeakHourRow(NamedTuple):
    """One hour-of-day bucket of the peak-hours breakdown."""

    hour: str          # "HH:00"
    total: int
    authorized: int
    unauthorized: int
    pending: int


class DepartmentRow(NamedTuple):
    """One responsible-department breakdown."""

    department: str
    total: int
    authorized: int
    unauthorized: int
    pending: int


class ExcelTables(NamedTuple):
    """The four sheets' data for ``/api/analytics/export-excel``.

    ``daily`` is newest-first and ``hourly`` / ``departments`` are dicts keyed
    ``"HH:00"`` and department name respectively, matching the ordering the
    workbook previously produced.
    """

    summary: Dict[str, int]
    daily: List[DailyRow]
    hourly: Dict[str, Dict[str, int]]
    departments: Dict[str, Dict[str, int]]

    @property
    def departments_sorted_by_total(self) -> List[Dict[str, Any]]:
        """Departments ordered by total descending, as the sheet printed them."""
        return [
            {"department": dept, **counts}
            for dept, counts in sorted(
                self.departments.items(),
                key=lambda item: item[1]["total"],
                reverse=True,
            )
        ]


def _status_sums() -> Dict[str, Any]:
    """Conditional counters for the three known statuses."""
    return {
        status: {"$sum": {"$cond": [{"$eq": ["$status", status]}, 1, 0]}}
        for status in _STATUSES
    }


def _padded_utc_hour() -> Dict[str, Any]:
    """``$hour`` of ``entry_datetime_utc`` as a zero-padded "HH" string.

    ``$hour`` yields a number and ``$toString`` renders 8 as ``"8"``, not
    ``"08"``. The local ``entry_time`` prefix is genuinely two characters, so
    the two branches only line up in a shared bucket space once the UTC hour is
    padded to match. Without this, hours 00-09 silently lost their UTC
    contribution.
    """
    return {
        "$toString": {
            "$cond": [
                {"$lt": [{"$hour": "$entry_datetime_utc"}, 10]},
                {"$concat": ["0", {"$toString": {"$hour": "$entry_datetime_utc"}}]},
                {"$toString": {"$hour": "$entry_datetime_utc"}},
            ]
        }
    }


def _peak_hour_buckets_stage() -> List[Dict[str, Any]]:
    """The two shared peak-hour stages: derive hour keys, then union them.

    Mirrors the Python ``(local prefix) OR (UTC hour)`` test. ``$setUnion``
    performs the de-duplication the Python boolean OR implied, so a visitor
    whose local and UTC clocks name the same hour is still counted once.
    """
    return [
        {
            "$project": {
                "status": 1,
                # First two characters of the local "HH:MM:SS" string.
                "h_local": {
                    "$substrBytes": [{"$ifNull": ["$entry_time", ""]}, 0, 2]
                },
                # Hour of the raw UTC timestamp, zero-padded to "HH".
                "h_utc": _padded_utc_hour(),
            }
        },
        {
            "$project": {
                "status": 1,
                "buckets": {
                    "$setUnion": [
                        {
                            "$cond": [
                                {
                                    "$regexMatch": {
                                        "input": "$h_local",
                                        "regex": "^[0-9][0-9]$",
                                    }
                                },
                                ["$h_local"],
                                [],
                            ]
                        },
                        ["$h_utc"],
                    ]
                },
            }
        },
        {"$unwind": "$buckets"},
        {
            "$group": {
                "_id": "$buckets",
                "total": {"$sum": 1},
                **_status_sums(),
            }
        },
    ]


class ReportRepository:
    """Aggregation access for the visitor reports. Stateless."""

    @classmethod
    def _collection(cls):
        return Visitor._get_collection()

    # ── Whole-collection summary ────────────────────────────────────────────

    @classmethod
    def status_summary(cls) -> StatusSummary:
        """Total visitors and the three status counts, in one query."""
        pipeline = [{"$group": dict({"_id": "$status", "n": {"$sum": 1}}, **{
            status: {"$sum": {"$cond": [{"$eq": ["$status", status]}, 1, 0]}}
            for status in _STATUSES
        })}]
        rows = list(cls._collection().aggregate(pipeline))
        total = sum(int(r.get("n") or 0) for r in rows)
        if len(rows) == 1 and rows[0].get("_id") is None:
            # A single null-status group means every document shares it.
            pass
        counts = {status: 0 for status in _STATUSES}
        for row in rows:
            status = row.get("_id")
            if status in counts:
                counts[status] = int(row.get("n") or 0)
        return StatusSummary(total, counts["authorized"],
                             counts["unauthorized"], counts["pending"])

    # ── Daily breakdown ─────────────────────────────────────────────────────

    @classmethod
    def daily_rows(cls, end_date: datetime, days: int) -> List[DailyRow]:
        """Visitor counts per UTC calendar day, newest day first.

        ``end_date`` minus ``i`` days for ``i`` in ``0..days-1``, exactly the
        range the Python loop walked. Every day is returned even when it has no
        visitors, matching the original zero-filled rows.
        """
        pipeline = [
            {"$match": {"entry_datetime_utc": {"$ne": None}}},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$entry_datetime_utc",
                        }
                    },
                    "total": {"$sum": 1},
                    **_status_sums(),
                }
            },
        ]
        by_date = {
            row["_id"]: row
            for row in cls._collection().aggregate(pipeline)
        }

        rows: List[DailyRow] = []
        for i in range(days):
            day = end_date - timedelta(days=i)
            date_str = day.strftime("%Y-%m-%d")
            row = by_date.get(date_str)
            rows.append(DailyRow(
                date=date_str,
                day=day.strftime("%a"),
                total=int(row["total"]) if row else 0,
                authorized=int(row.get("authorized", 0)) if row else 0,
                unauthorized=int(row.get("unauthorized", 0)) if row else 0,
                pending=int(row.get("pending", 0)) if row else 0,
            ))
        return rows

    # ── Peak hours ──────────────────────────────────────────────────────────

    @classmethod
    def peak_hour_rows(
        cls, end_date: datetime, days: int = 30
    ) -> List[PeakHourRow]:
        """Per-hour visitor counts over the trailing window.

        Reproduces the original ``(local entry_time prefix) OR (UTC hour)``
        test. All 24 hours are always returned, including empty ones, because
        the Python loop always wrote a key for every hour.
        """
        window_start = end_date - timedelta(days=days)
        pipeline = [
            # month_docs in the original: a non-null entry_datetime_utc inside
            # the trailing window.
            {"$match": {"entry_datetime_utc": {"$gte": window_start}}},
            *_peak_hour_buckets_stage(),
        ]
        by_hour = {row["_id"]: row for row in cls._collection().aggregate(pipeline)}

        rows: List[PeakHourRow] = []
        for hour in range(24):
            key = f"{hour:02d}"
            row = by_hour.get(key)
            rows.append(PeakHourRow(
                hour=f"{hour:02d}:00",
                total=int(row["total"]) if row else 0,
                authorized=int(row.get("authorized", 0)) if row else 0,
                unauthorized=int(row.get("unauthorized", 0)) if row else 0,
                pending=int(row.get("pending", 0)) if row else 0,
            ))
        return rows

    # ── Departments ─────────────────────────────────────────────────────────

    @classmethod
    def department_rows(cls) -> List[DepartmentRow]:
        """Per-responsible-department counts, blank departments excluded."""
        pipeline = [
            {"$match": {"responsible_department": {"$nin": [None, ""]}}},
            {
                "$group": {
                    "_id": "$responsible_department",
                    "total": {"$sum": 1},
                    **_status_sums(),
                }
            },
        ]
        return [
            DepartmentRow(
                department=str(row["_id"]),
                total=int(row["total"]),
                authorized=int(row.get("authorized", 0)),
                unauthorized=int(row.get("unauthorized", 0)),
                pending=int(row.get("pending", 0)),
            )
            for row in cls._collection().aggregate(pipeline)
        ]

    # ── Combined reads (JSON endpoint and Excel export) ────────────────────

    @classmethod
    def _facets(
        cls, end_date: datetime
    ) -> Dict[str, Any]:
        """Run the shared four-way ``$facet`` aggregation once.

        All four breakdowns need the same collection scan, so they share one
        command. Returns the raw facet document, keyed ``summary`` / ``daily``
        / ``peak`` / ``departments``.
        """
        window_start = end_date - timedelta(days=30)
        pipeline = [
            {
                "$facet": {
                    "summary": [
                        {
                            "$group": {
                                "_id": "$status",
                                "n": {"$sum": 1},
                                **_status_sums(),
                            }
                        }
                    ],
                    "daily": [
                        {"$match": {"entry_datetime_utc": {"$ne": None}}},
                        {
                            "$group": {
                                "_id": {
                                    "$dateToString": {
                                        "format": "%Y-%m-%d",
                                        "date": "$entry_datetime_utc",
                                    }
                                },
                                "total": {"$sum": 1},
                                **_status_sums(),
                            }
                        },
                    ],
                    "peak": [
                        {"$match": {"entry_datetime_utc": {"$gte": window_start}}},
                        *_peak_hour_buckets_stage(),
                    ],
                    "departments": [
                        {"$match": {"responsible_department": {"$nin": [None, ""]}}},
                        {
                            "$group": {
                                "_id": "$responsible_department",
                                "total": {"$sum": 1},
                                **_status_sums(),
                            }
                        },
                    ],
                }
            }
        ]
        # A $facet pipeline returns exactly one document keyed by facet name.
        rows = list(cls._collection().aggregate(pipeline))
        if not rows:
            return {"summary": [], "daily": [], "peak": [], "departments": []}
        return rows[0]

    @staticmethod
    def _summary_counts(facets: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
        """``(total, {status: count})`` from the summary facet."""
        total = 0
        counts = {status: 0 for status in _STATUSES}
        for row in facets["summary"]:
            n = int(row.get("n") or 0)
            total += n
            status = row.get("_id")
            if status in counts:
                counts[status] = n
        return total, counts

    @classmethod
    def _daily_series(
        cls, facets: Dict[str, Any], end_date: datetime, days: int
    ) -> List[DailyRow]:
        """``days`` daily rows, zero-filled, newest first (the original order)."""
        by_date = {row["_id"]: row for row in facets["daily"]}
        series: List[DailyRow] = []
        for i in range(days):
            day = end_date - timedelta(days=i)
            date_str = day.strftime("%Y-%m-%d")
            row = by_date.get(date_str)
            series.append(DailyRow(
                date=date_str, day=day.strftime("%a"),
                total=int(row["total"]) if row else 0,
                authorized=int(row.get("authorized", 0)) if row else 0,
                unauthorized=int(row.get("unauthorized", 0)) if row else 0,
                pending=int(row.get("pending", 0)) if row else 0,
            ))
        return series

    @staticmethod
    def _peak_hours(facets: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """All 24 hour buckets, always present, zero-filled."""
        by_hour = {row["_id"]: row for row in facets["peak"]}
        hours: Dict[str, Dict[str, int]] = {}
        for hour in range(24):
            row = by_hour.get(f"{hour:02d}")
            hours[f"{hour:02d}:00"] = {
                "total": int(row["total"]) if row else 0,
                "authorized": int(row.get("authorized", 0)) if row else 0,
                "unauthorized": int(row.get("unauthorized", 0)) if row else 0,
                "pending": int(row.get("pending", 0)) if row else 0,
            }
        return hours

    @staticmethod
    def _departments(facets: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        by_dept = {str(row["_id"]): row for row in facets["departments"]}
        return {
            dept: {
                "total": int(row["total"]),
                "authorized": int(row.get("authorized", 0)),
                "unauthorized": int(row.get("unauthorized", 0)),
                "pending": int(row.get("pending", 0)),
            }
            for dept, row in by_dept.items()
        }

    @classmethod
    def visitor_stats(cls, end_date: datetime, daily_days: int = 7) -> Dict[str, Any]:
        """Everything ``/api/analytics/visitor-stats`` returns, in one command.

        Uses ``$facet`` so the four independent breakdowns share a single
        collection scan and a single round trip, replacing one full
        ``find`` (plus its ``getMore``) and all the Python looping.
        """
        facets = cls._facets(end_date)
        total_visitors, counts = cls._summary_counts(facets)
        daily = cls._daily_series(facets, end_date, daily_days)
        return {
            "total_visitors": total_visitors,
            "authorized_visitors": counts["authorized"],
            "unauthorized_visitors": counts["unauthorized"],
            "pending_visitors": counts["pending"],
            # daily_traffic and daily_status are the same underlying rows; the
            # route re-derives the two JSON shapes from them.
            "daily_traffic": daily,
            "daily_status": daily,
            "peak_hours": cls._peak_hours(facets),
            "department_stats": cls._departments(facets),
        }

    @classmethod
    def excel_tables(cls, end_date: datetime) -> ExcelTables:
        """The four tables ``/api/analytics/export-excel`` writes.

        Same single ``$facet`` command as :meth:`visitor_stats`, but with the
        30-day daily window the workbook uses instead of the JSON endpoint's
        7. Ordering matches the original workbook: daily newest-first,
        hourly 00:00-23:00, departments sorted by total descending.
        """
        facets = cls._facets(end_date)
        total_visitors, counts = cls._summary_counts(facets)
        return ExcelTables(
            summary={
                "total_visitors": total_visitors,
                "authorized": counts["authorized"],
                "unauthorized": counts["unauthorized"],
                "pending": counts["pending"],
            },
            daily=cls._daily_series(facets, end_date, 30),
            hourly=cls._peak_hours(facets),
            departments=cls._departments(facets),
        )
