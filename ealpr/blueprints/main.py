"""
blueprints/main.py — Dashboard and request lifecycle hooks.

Routes:
    GET /  (dashboard)

Hooks:
    before_request — update last_login timestamp
    after_request  — log every request as a UserActivity record
"""
from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import login_required, current_user
from datetime import datetime

from ealpr import extensions
from ealpr.repositories import DetectionRepository, VisitorRepository
from models import UserActivity

main_bp = Blueprint("main", __name__)


def _dashboard_stats():
    """The four dashboard KPIs, read through the repositories.

    Phase 2 / Block C1: the three visitor counters were three separate
    ``count()`` calls (three round trips). They are now a single ``$group``
    aggregation, so the dashboard costs two queries instead of four. The
    figures and their semantics are unchanged.
    """
    total_detections = DetectionRepository.count_all()
    counts = VisitorRepository.dashboard_counts()
    return {
        "total_detections": total_detections,
        "authorized_visitors": counts.authorized,
        "unauthorized_visitors": counts.unauthorized,
        "active_visitors": counts.active,
    }


@main_bp.before_app_request
def before_request():
    pass


@main_bp.after_app_request
def after_request(response):
    if not current_user.is_authenticated or not extensions.DB_ENABLED:
        return response
    if request.endpoint in (None, "static"):
        return response
    if request.method == "GET" and request.path.startswith("/api/"):
        return response
    if request.method not in ("POST", "PUT", "DELETE", "PATCH"):
        return response

    # Save activity in a background thread so it never delays the HTTP response
    import threading
    from flask import copy_current_request_context
    endpoint = request.endpoint or request.path
    method_path = f"{request.method} {request.path}"
    ip = request.remote_addr
    user = current_user._get_current_object() if hasattr(current_user, '_get_current_object') else None

    def _save():
        try:
            activity = UserActivity(
                user=user,
                action=endpoint,
                details=method_path,
                ip_address=ip,
                timestamp=datetime.utcnow(),
            )
            activity.save()
        except Exception:
            pass

    threading.Thread(target=_save, daemon=True).start()
    return response


@main_bp.route("/")
@login_required
def index():
    if not extensions.DB_ENABLED:
        return render_template(
            "index.html",
            total_detections=0,
            authorized_visitors=0,
            unauthorized_visitors=0,
            active_visitors=0,
        )
    stats = _dashboard_stats()
    return render_template(
        "index.html",
        total_detections=stats["total_detections"],
        authorized_visitors=stats["authorized_visitors"],
        unauthorized_visitors=stats["unauthorized_visitors"],
        active_visitors=stats["active_visitors"],
    )


@main_bp.route("/api/dashboard/stats")
@login_required
def get_dashboard_stats():
    if not extensions.DB_ENABLED:
        return jsonify({
            "success": False,
            "stats": {
                "total_detections": 0,
                "authorized_visitors": 0,
                "unauthorized_visitors": 0,
                "active_visitors": 0
            }
        })
    try:
        return jsonify({"success": True, "stats": _dashboard_stats()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

