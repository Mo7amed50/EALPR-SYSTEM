"""ReportRepository tests.

The MongoDB pipelines here replaced Python loops that walked the whole visitor
collection. The risk with such a rewrite is silent divergence, so these tests
keep the original algorithms in the file as oracles and compare the two at
several different ``end_date`` instants, including ones chosen to make the
30-day window cover the older records (which is what exposed the zero-padding
bug in the UTC hour key).

Everything here is read-only.
"""

from datetime import datetime, timedelta

from ealpr.repositories import ReportRepository
from models import Visitor
from tests.base import RepositoryTestCase, counted

_STATUSES = ("authorized", "unauthorized", "pending")


def _raw_docs():
    """The unprojected document set the original code operated on."""
    return list(Visitor._get_collection().find({}, {
        "status": 1,
        "entry_datetime_utc": 1,
        "entry_time": 1,
        "responsible_department": 1,
    }))


# ── Oracles transcribed from the original reports.py ─────────────────────────

def oracle_summary(docs):
    total = len(docs)
    return {
        "total": total,
        "authorized": sum(1 for d in docs if d.get("status") == "authorized"),
        "unauthorized": sum(1 for d in docs if d.get("status") == "unauthorized"),
        "pending": sum(1 for d in docs if d.get("status") == "pending"),
    }


def oracle_daily(docs, end_date, days):
    """Newest-first, zero-filled, exactly as the workbook built it."""
    rows = []
    for i in range(days):
        day = end_date - timedelta(days=i)
        d_str = day.strftime("%Y-%m-%d")
        bucket = [
            d for d in docs
            if d.get("entry_datetime_utc")
            and d["entry_datetime_utc"].strftime("%Y-%m-%d") == d_str
        ]
        rows.append({
            "date": d_str,
            "day": day.strftime("%a"),
            "total": len(bucket),
            "authorized": sum(1 for d in bucket if d.get("status") == "authorized"),
            "unauthorized": sum(1 for d in bucket if d.get("status") == "unauthorized"),
            "pending": sum(1 for d in bucket if d.get("status") == "pending"),
        })
    return rows


def oracle_peak_hours(docs, end_date, days=30):
    month_start = end_date - timedelta(days=days)
    month_docs = [
        d for d in docs
        if d.get("entry_datetime_utc") and d["entry_datetime_utc"] >= month_start
    ]
    out = {}
    for hour in range(24):
        prefix = f"{hour:02d}:"
        bucket = [
            d for d in month_docs
            if (d.get("entry_time") and str(d["entry_time"]).startswith(prefix))
            or (d.get("entry_datetime_utc") and d["entry_datetime_utc"].hour == hour)
        ]
        out[f"{hour:02d}:00"] = {
            "total": len(bucket),
            "authorized": sum(1 for d in bucket if d.get("status") == "authorized"),
            "unauthorized": sum(1 for d in bucket if d.get("status") == "unauthorized"),
            "pending": sum(1 for d in bucket if d.get("status") == "pending"),
        }
    return out


def oracle_departments(docs):
    stats = {}
    for d in docs:
        dept = d.get("responsible_department")
        if dept:
            stats.setdefault(dept, {"total": 0, "authorized": 0, "unauthorized": 0, "pending": 0})
            stats[dept]["total"] += 1
            if d.get("status") in stats[dept]:
                stats[dept][d.get("status")] += 1
    return stats


class AggregationEquivalenceTests(RepositoryTestCase):
    """The pipelines must agree with the Python originals, at many instants."""

    #: "now", a 30-day-window instant that includes the older April records,
    #: a mid-range date, a month boundary and a far-future date.
    ENDS = (
        None,
        datetime(2026, 4, 6, 12, 0, 0),
        datetime(2026, 9, 20, 23, 59, 59),
        datetime(2027, 1, 1, 0, 0, 0),
    )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.docs = _raw_docs()
        if not cls.docs:
            raise AssertionError("no visitors to compare against")

    def _ends(self):
        for end in self.ENDS:
            yield end or datetime.utcnow()

    def test_summary_matches(self):
        expected = oracle_summary(self.docs)
        for end in self._ends():
            with self.subTest(end_date=end.isoformat()):
                stats = ReportRepository.visitor_stats(end)
                self.assertEqual(stats["total_visitors"], expected["total"])
                self.assertEqual(stats["authorized_visitors"], expected["authorized"])
                self.assertEqual(stats["unauthorized_visitors"], expected["unauthorized"])
                self.assertEqual(stats["pending_visitors"], expected["pending"])

    def test_seven_day_daily_matches(self):
        for end in self._ends():
            with self.subTest(end_date=end.isoformat()):
                stats = ReportRepository.visitor_stats(end, daily_days=7)
                expected = oracle_daily(self.docs, end, 7)
                actual = [
                    {
                        "date": r.date, "day": r.day, "total": r.total,
                        "authorized": r.authorized,
                        "unauthorized": r.unauthorized,
                        "pending": r.pending,
                    }
                    for r in stats["daily_traffic"]
                ]
                self.assertEqual(actual, expected)

    def test_peak_hours_match_including_early_morning_hours(self):
        """Regression guard for the UTC hour zero-padding bug.

        ``$toString`` renders hour 8 as ``"8"``, which silently emptied the
        ``08`` bucket until the UTC hour was padded. Only a window that
        contains the 08:xx records exposes it.
        """
        for end in self._ends():
            with self.subTest(end_date=end.isoformat()):
                stats = ReportRepository.visitor_stats(end)
                expected = oracle_peak_hours(self.docs, end)
                self.assertEqual(stats["peak_hours"], expected)
                for hour in range(24):
                    key = f"{hour:02d}:00"
                    self.assertEqual(stats["peak_hours"][key]["total"],
                                     expected[key]["total"], f"bucket {key}")

    def test_departments_match(self):
        for end in self._ends():
            with self.subTest(end_date=end.isoformat()):
                stats = ReportRepository.visitor_stats(end)
                self.assertEqual(stats["department_stats"], oracle_departments(self.docs))

    def test_excel_tables_match(self):
        for end in self._ends():
            with self.subTest(end_date=end.isoformat()):
                tables = ReportRepository.excel_tables(end)
                expected_summary = oracle_summary(self.docs)
                self.assertEqual(tables.summary["total_visitors"], expected_summary["total"])
                self.assertEqual(tables.summary["authorized"], expected_summary["authorized"])

                expected_daily = oracle_daily(self.docs, end, 30)
                actual_daily = [
                    {
                        "date": r.date, "day": r.day, "total": r.total,
                        "authorized": r.authorized,
                        "unauthorized": r.unauthorized,
                        "pending": r.pending,
                    }
                    for r in tables.daily
                ]
                self.assertEqual(actual_daily, expected_daily)
                self.assertEqual(tables.hourly, oracle_peak_hours(self.docs, end))
                self.assertEqual(tables.departments, oracle_departments(self.docs))

    def test_departments_are_sorted_by_total_descending(self):
        tables = ReportRepository.excel_tables(datetime.utcnow())
        totals = [row["total"] for row in tables.departments_sorted_by_total]
        self.assertEqual(totals, sorted(totals, reverse=True))


class QueryShapeTests(RepositoryTestCase):
    """The point of the rewrite: far fewer commands and no bulk transfer."""

    def test_visitor_stats_costs_one_command(self):
        with counted() as spy:
            ReportRepository.visitor_stats(datetime.utcnow())
        self.assertEqual(
            spy.reads, 1,
            f"the $facet must be a single command, saw {dict(spy.commands)}",
        )

    def test_excel_tables_costs_one_command(self):
        with counted() as spy:
            ReportRepository.excel_tables(datetime.utcnow())
        self.assertEqual(
            spy.reads, 1,
            f"the $facet must be a single command, saw {dict(spy.commands)}",
        )

    def test_no_visitor_documents_are_transferred(self):
        """A full pull would show up as getMore; the facet returns a few rows."""
        with counted() as spy:
            ReportRepository.visitor_stats(datetime.utcnow())
        self.assertEqual(
            spy.commands.get("getMore", 0), 0,
            f"the aggregation must not stream documents, saw {dict(spy.commands)}",
        )

    def test_empty_collection_yields_zeroed_shapes(self):
        """The zero-fill contract holds even with no data at all."""
        rows = ReportRepository.daily_rows(datetime(1990, 1, 1), days=7)
        self.assertEqual(len(rows), 7)
        for row in rows:
            self.assertEqual(row.total, 0)
            self.assertEqual(row.authorized, 0)
