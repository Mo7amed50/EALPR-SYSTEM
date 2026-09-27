"""End-to-end read-only route tests.

These drive the real Flask app through the routes Block C1 touched and assert
the two things that matter after a data-access refactor:

* the HTTP responses still look exactly as they did, and
* each route issues a bounded number of MongoDB commands, so a per-row query
  cannot creep back in unnoticed.

They also pin the explicit non-goals of the block: the AI detection path must
still read the visitor collection directly, and no route may start writing.

Run with:  .venv\\Scripts\\python.exe -m unittest tests.test_routes_readonly -v
"""

import re
import unittest
from pathlib import Path

from tests.base import BASE_DIR, RepositoryTestCase, counted, get_app

DETECTION_SOURCE = Path(BASE_DIR) / "ealpr" / "blueprints" / "detection.py"

#: Route path -> (method, max read commands, must-not-appear-in-response).
#:
#: The command budgets are the "no N+1" contract. They are deliberately tight:
#: each budget counts the real data queries the route needs and nothing more, so
#: a regression shows up as a failure rather than as a slow page nobody notices.
ROUTES = (
    ("/", "GET", 3, 200),
    ("/api/dashboard/stats", "GET", 3, 200),
    ("/uploaded_images", "GET", 3, 200),
    ("/uploaded_images?page=2&per_page=50", "GET", 3, 200),
    ("/uploaded_images?status=unauthorized", "GET", 3, 200),
    ("/detection_history", "GET", 3, 200),
    ("/visitors", "GET", 3, 200),
    ("/manage_visitors", "GET", 3, 200),
    ("/api/activities", "GET", 2, 200),
    ("/api/analytics/visitor-stats", "GET", 2, 200),
    # No matching plate: a 404 is the correct answer, and it must still be
    # reached with a single projected lookup.
    ("/api/search/plate?q=zzzznomatchzzzz", "GET", 2, 404),
)


class RouteResponseTests(RepositoryTestCase):
    """Every touched read route answers 200 within its query budget."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        app = get_app()
        app.config["TESTING"] = True
        cls.client = app.test_client()

        from models import User

        admin = User.objects(username="admin").first()
        if admin is None:
            raise unittest.SkipTest("no admin user to authenticate as")
        with cls.client.session_transaction() as sess:
            sess["_user_id"] = str(admin.id)
            sess["_fresh"] = True

    def test_routes_respect_their_query_budget(self):
        for path, method, budget, expected_status in ROUTES:
            with self.subTest(path=path):
                with counted() as spy:
                    response = self.client.open(path, method=method)
                self.assertEqual(
                    response.status_code, expected_status,
                    f"{path} returned {response.status_code}, "
                    f"expected {expected_status}",
                )
                self.assertLessEqual(
                    spy.reads, budget,
                    f"{path} issued {dict(spy.commands)}, budget was {budget}",
                )

    def test_dashboard_kpis_are_consistent(self):
        counts = self.client.get("/api/dashboard/stats").get_json()
        self.assertTrue(counts["success"])
        stats = counts["stats"]
        self.assertEqual(
            set(stats),
            {"total_detections", "authorized_visitors",
             "unauthorized_visitors", "active_visitors"},
        )
        from models import DetectionResult, Visitor

        self.assertEqual(stats["total_detections"], DetectionResult.objects.count())
        self.assertEqual(
            stats["authorized_visitors"], Visitor.objects(authorized=True).count()
        )
        self.assertEqual(
            stats["unauthorized_visitors"], Visitor.objects(authorized=False).count()
        )
        self.assertEqual(
            stats["active_visitors"], Visitor.objects(exit_time__in=[None, ""]).count(),
            "the active KPI must keep its original meaning",
        )

    def test_visitor_stats_shape(self):
        body = self.client.get("/api/analytics/visitor-stats").get_json()
        self.assertTrue(body["success"])
        for key in ("summary", "daily_traffic", "daily_status",
                    "peak_hours", "department_stats"):
            self.assertIn(key, body)
        self.assertEqual(len(body["daily_traffic"]), 7)
        self.assertEqual(len(body["daily_status"]), 7)
        self.assertEqual(len(body["peak_hours"]), 24)
        self.assertEqual(
            set(body["summary"]),
            {"total_visitors", "authorized_visitors", "unauthorized_visitors",
             "pending_visitors", "authorization_rate"},
        )
        # The endpoint always answered oldest-first.
        dates = [row["date"] for row in body["daily_traffic"]]
        self.assertEqual(dates, sorted(dates), "daily_traffic must be oldest first")
        self.assertEqual(
            [row["date"] for row in body["daily_status"]], dates,
            "daily_status and daily_traffic must cover the same days in order",
        )

    def test_activities_shape(self):
        body = self.client.get("/api/activities").get_json()
        self.assertTrue(body["success"])
        self.assertLessEqual(len(body["activities"]), 10)
        for entry in body["activities"]:
            self.assertEqual(set(entry), {"action", "details", "timestamp"})

    def test_visitor_pages_do_not_leak_id_number(self):
        for path in ("/visitors", "/manage_visitors"):
            with self.subTest(path=path):
                body = self.client.get(path).get_data(as_text=True)
                self.assertNotIn("id_number", body)

    def test_gallery_and_history_pagination_controls_present(self):
        gallery = self.client.get("/uploaded_images").get_data(as_text=True)
        self.assertIn("per_page", gallery)
        history = self.client.get("/detection_history").get_data(as_text=True)
        self.assertIn("detection_history", history.lower())


class AiPathIsUntouchedTests(unittest.TestCase):
    """C1 explicitly excluded the AI path; assert it structurally, not by hope.

    ``/process_plate`` resolves the visitor itself and then runs inference, and
    it writes a DetectionResult. It was deliberately left reading
    ``Visitor.objects(...)`` directly rather than going through the repository,
    because the repository is a read-only projection layer and this path needs
    a complete, savable document. This pins that.
    """

    @classmethod
    def setUpClass(cls):
        cls.source = DETECTION_SOURCE.read_text(encoding="utf-8")

    def _body(self, signature):
        start = self.source.index(signature)
        end = self.source.find("\ndef ", start + 1)
        if end == -1:
            end = len(self.source)
        return self.source[start:end]

    def test_process_plate_still_queries_visitors_directly(self):
        body = self._body("def process_plate(")
        self.assertIn(
            "Visitor.objects(", body,
            "/process_plate must keep its own direct visitor lookup",
        )
        self.assertNotIn(
            "VisitorRepository", body,
            "/process_plate was out of scope for C1 and must not use the repository",
        )
        self.assertIn(
            "detection_record.save()", body,
            "/process_plate must still write its detection record",
        )

    def test_process_video_does_not_use_the_repository(self):
        body = self._body("def process_video(")
        self.assertNotIn(
            "VisitorRepository", body,
            "/process_video was out of scope for C1",
        )
        self.assertNotIn("DetectionRepository", body)

    def test_camera_and_socket_handlers_untouched(self):
        self.assertNotIn("Repository", self._body("def video_feed("))
        self.assertNotIn("Repository", self._body("def handle_connect("))
        self.assertNotIn("Repository", self._body("def handle_disconnect("))

    def test_no_repository_import_leaked_into_write_paths(self):
        """A mutating route must never resolve a projected document.

        The authorize / unauthorize / update / delete actions are all handled by
        one ``/api/visitors/action`` route, which saves the document it loaded.
        """
        visitors_source = (
            Path(BASE_DIR) / "ealpr" / "blueprints" / "visitors.py"
        ).read_text(encoding="utf-8")
        for func in ("def visitor_action(", "def update_visitor(",
                     "def delete_visitor("):
            if func not in visitors_source:
                continue
            start = visitors_source.index(func)
            end = visitors_source.find("\ndef ", start + 1)
            if end == -1:
                end = len(visitors_source)
            with self.subTest(func=func):
                self.assertIn(
                    "for_update=True", visitors_source[start:end],
                    f"{func} must load the complete document before saving it",
                )

    def test_read_only_lookups_use_the_projected_variant(self):
        visitors_source = (
            Path(BASE_DIR) / "ealpr" / "blueprints" / "visitors.py"
        ).read_text(encoding="utf-8")
        start = visitors_source.index("def get_visitor(")
        end = visitors_source.find("\ndef ", start + 1)
        body = visitors_source[start:end]
        self.assertIn("_find_visitor(visitor_id)", body)
        self.assertNotIn(
            "for_update=True", body,
            "a read-only GET must not pay for an unprojected load",
        )
