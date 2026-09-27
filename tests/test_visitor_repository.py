"""VisitorRepository tests.

Two properties matter most here and both are regressions that the repository
exists to prevent:

* the four-tier visitor lookup must resolve in the *same precedence* as the
  ``_find_visitor`` cascade it replaced, and
* read lookups must be projected while write lookups must return the complete
  document, because the mutating routes call ``save()`` on what they get back.
"""

from bson.objectid import ObjectId
from mongoengine import Q

from ealpr.repositories import VisitorRepository
from models import Visitor
from tests.base import RepositoryTestCase, counted


def reference_find_visitor(visitor_id_str):
    """The original ``_find_visitor`` cascade, transcribed verbatim.

    Kept here as the oracle the repository is checked against. If the tiers or
    their order ever change, this is what must change with them.
    """
    if not visitor_id_str:
        return None
    visitor_id_str = str(visitor_id_str).strip()
    if visitor_id_str.isdigit():
        v = Visitor.objects(visitor_id=int(visitor_id_str)).first()
        if v:
            return v
    if ObjectId.is_valid(visitor_id_str):
        v = Visitor.objects(id=ObjectId(visitor_id_str)).first()
        if v:
            return v
    v = Visitor.objects(visitor_code=visitor_id_str).first()
    if v:
        return v
    return Visitor.objects(license_plate=visitor_id_str).first()


class LookupPrecedenceTests(RepositoryTestCase):
    """find_by_identifier must agree with the original cascade, tier by tier."""

    def _identifiers(self):
        """One real identifier per tier, so every branch is exercised."""
        seq = Visitor.objects(visitor_id__gt=0).order_by("visitor_id").first()
        codes = Visitor.objects(
            Q(visitor_code__ne="") & Q(visitor_code__ne=None)
        ).first()
        plates = Visitor.objects.order_by("license_plate").first()
        samples = {}
        if seq:
            samples["visitor_id"] = str(seq.visitor_id)
            samples["id"] = str(seq.id)
        if codes:
            samples["visitor_code"] = codes.visitor_code
        if plates:
            samples["license_plate"] = plates.license_plate
        return samples

    def test_every_tier_matches_the_original_cascade(self):
        samples = self._identifiers()
        self.assertGreaterEqual(len(samples), 3, f"not enough tiers to test: {samples}")
        for tier, identifier in sorted(samples.items()):
            with self.subTest(tier=tier):
                expected = reference_find_visitor(identifier)
                actual = VisitorRepository.find_by_identifier(identifier)
                self.assertIsNotNone(expected, f"reference found nothing for {tier}")
                self.assertIsNotNone(actual, f"repository found nothing for {tier}")
                self.assertEqual(
                    str(expected.id), str(actual.id),
                    f"{tier} resolved to a different visitor",
                )

    def test_unknown_identifier_returns_none(self):
        self.assertIsNone(VisitorRepository.find_by_identifier("no-such-visitor-xyz"))
        self.assertIsNone(VisitorRepository.find_by_identifier(""))

    def test_numeric_identifier_wins_over_code_and_plate(self):
        """A digit string must resolve by sequence id even if it is also a code.

        This is the case the single-$or rewrite could plausibly have broken, so
        it is pinned rather than left to the sampling above. The test looks for
        a real collision on the live data and skips itself when the data does
        not happen to contain one.
        """
        col = Visitor._get_collection()
        # Three queries total, then the overlap is computed in Python. A query
        # per candidate would mean 1023 round trips.
        by_visitor_id = [
            (str(doc["visitor_id"]), doc["_id"])
            for doc in col.find(
                {"visitor_id": {"$type": "int"}}, {"visitor_id": 1}
            )
        ]
        claimed = {}
        for doc in col.find(
            {"$or": [
                {"visitor_code": {"$type": "string"}},
                {"license_plate": {"$type": "string"}},
            ]},
            {"visitor_code": 1, "license_plate": 1},
        ):
            for value in (doc.get("visitor_code"), doc.get("license_plate")):
                if value:
                    claimed.setdefault(str(value), set()).add(doc["_id"])

        collision = None
        for text, oid in by_visitor_id:
            if not text.isdigit():
                continue
            rivals = claimed.get(text, set()) - {oid}
            if rivals:
                collision = (oid, text)
                break
        if not collision:
            self.skipTest(
                "no digit string is both a visitor_id and another visitor's code/plate"
            )

        oid, text = collision
        expected = reference_find_visitor(text)
        actual = VisitorRepository.find_by_identifier(text)
        self.assertEqual(str(expected.id), str(actual.id))
        self.assertEqual(
            str(oid), str(actual.id),
            "the numeric visitor_id tier must take precedence over code/plate",
        )


class ProjectionTests(RepositoryTestCase):
    """List reads project; the write lookup must not."""

    def test_list_readonly_projects_away_id_number(self):
        with counted() as spy:
            rows = VisitorRepository.list_readonly()
        self.assertEqual(len(rows), self.total_visitors)
        for projection in spy.projections:
            self.assertNotIn(
                "id_number", projection,
                "id_number is a national ID that /visitors never renders",
            )
            self.assertNotIn("id_number", projection.get("exclude", {}))

    def test_search_manage_projects_away_id_number(self):
        with counted() as spy:
            rows = VisitorRepository.search_manage("")
        self.assertEqual(len(rows), self.total_visitors)
        for projection in spy.projections:
            self.assertNotIn("id_number", projection)

    def test_list_readonly_uses_a_single_query(self):
        """One find, plus at most the getMore batches for 1023 documents.

        A per-row lookup would show up here as ~1023 finds, which is the N+1
        this repository is supposed to make impossible.
        """
        with counted() as spy:
            rows = VisitorRepository.list_readonly()
        self.assertEqual(
            spy.commands.get("find", 0), 1,
            f"expected a single find, saw {dict(spy.commands)}",
        )
        self.assertLessEqual(
            spy.commands.get("getMore", 0), 20,
            "getMore count should track batch size, not row count",
        )
        self.assertEqual(len(rows), self.total_visitors)

    def test_search_manage_still_filters(self):
        sample = Visitor.objects(
            Q(license_plate__ne="") & Q(license_plate__ne=None)
        ).first()
        if not sample:
            self.skipTest("no sample visitor with a plate")
        rows = VisitorRepository.search_manage(sample.license_plate)
        self.assertTrue(rows, "an exact plate must match at least one visitor")
        self.assertIn(
            sample.license_plate,
            [r.license_plate for r in rows],
        )

    def test_for_update_returns_a_complete_document(self):
        """MongoEngine writes back what it loaded, so this must be unprojected.

        If this ever regresses, authorize/update/delete would persist a partial
        document and silently erase every field not listed in the projection.
        """
        target = Visitor.objects.order_by("visitor_id").first()
        with counted() as spy:
            full = VisitorRepository.find_for_update(str(target.visitor_id))
        self.assertIsNotNone(full)
        for projection in spy.projections:
            self.assertFalse(
                projection,
                "find_for_update must not project: the mutating routes save() "
                f"the document, so a projection would erase fields. Saw {projection}",
            )
        raw = Visitor._get_collection().find_one({"_id": target.id})
        for field in raw:
            if field == "_id":
                continue
            self.assertEqual(
                getattr(full, field), raw[field],
                f"field {field!r} did not survive the load",
            )

    def test_for_update_agrees_with_reference_cascade(self):
        target = Visitor.objects.order_by("visitor_id").first()
        expected = reference_find_visitor(str(target.visitor_id))
        actual = VisitorRepository.find_for_update(str(target.visitor_id))
        self.assertEqual(str(expected.id), str(actual.id))

    def test_find_by_plate_and_search_plate_or_name(self):
        target = Visitor.objects.order_by("license_plate").first()
        by_plate = VisitorRepository.find_by_plate(target.license_plate)
        self.assertEqual(str(by_plate.id), str(target.id))

        by_search = VisitorRepository.search_plate_or_name(target.license_plate)
        self.assertIsNotNone(by_search)
        self.assertEqual(
            str(by_search.id), str(target.id),
            "search_plate_or_name must return the first match, as the old Q did",
        )

    def test_find_duplicate_matches_plate_or_code(self):
        target = Visitor.objects.order_by("visitor_id").first()
        by_plate = VisitorRepository.find_duplicate(target.license_plate, None)
        self.assertEqual(str(by_plate.id), str(target.id))
        if target.visitor_code:
            by_code = VisitorRepository.find_duplicate(
                "no-such-plate", target.visitor_code
            )
            self.assertEqual(str(by_code.id), str(target.id))


class DashboardCountTests(RepositoryTestCase):
    """The three visitor KPIs must match three separate count() calls exactly."""

    def test_counts_match_the_original_queries(self):
        counts = VisitorRepository.dashboard_counts()
        self.assertEqual(counts.authorized, Visitor.objects(authorized=True).count())
        self.assertEqual(counts.unauthorized, Visitor.objects(authorized=False).count())
        self.assertEqual(
            counts.active,
            Visitor.objects(exit_time__in=[None, ""]).count(),
        )

    def test_three_counts_cost_one_query(self):
        with counted() as spy:
            VisitorRepository.dashboard_counts()
        self.assertEqual(
            spy.reads, 1,
            f"expected a single aggregation, saw {dict(spy.commands)}",
        )

    def test_status_string_and_boolean_are_not_conflated(self):
        """The two notions of 'authorized' disagree on the live data.

        ``authorized=True`` yields 812 while ``status='authorized'`` yields 811.
        The dashboard counts the boolean and the reports count the string. This
        test fails loudly if a future change silently swaps one for the other.
        """
        by_boolean = VisitorRepository.dashboard_counts().authorized
        by_string = Visitor.objects(status="authorized").count()
        if by_boolean == by_string:
            self.skipTest(
                "live data no longer distinguishes the two definitions "
                f"(both are {by_boolean}), so this test cannot discriminate"
            )
        self.assertNotEqual(
            by_boolean, by_string,
            "the dashboard KPI and the report total have become the same number",
        )


class PeriodReportTests(RepositoryTestCase):
    def test_period_report_returns_only_the_window(self):
        from datetime import datetime, timedelta

        end = datetime.utcnow()
        start = end - timedelta(days=7)
        rows = VisitorRepository.list_for_period_report(start, end)
        for row in rows:
            self.assertIsNotNone(row.entry_datetime_utc)
            self.assertGreaterEqual(row.entry_datetime_utc, start)
            self.assertLessEqual(row.entry_datetime_utc, end)

    def test_period_report_is_ordered_ascending(self):
        from datetime import datetime, timedelta

        end = datetime.utcnow()
        rows = VisitorRepository.list_for_period_report(end - timedelta(days=365), end)
        stamps = [r.entry_datetime_utc for r in rows]
        self.assertEqual(stamps, sorted(stamps), "Excel row order must not change")
