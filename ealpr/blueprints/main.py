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

from ealpr.extensions import DB_ENABLED
from models import Visitor, DetectionResult, UserActivity

main_bp = Blueprint("main", __name__)


@main_bp.before_app_request
def before_request():
    pass


@main_bp.after_app_request
def after_request(response):
    if not current_user.is_authenticated or not DB_ENABLED:
        return response
    if request.endpoint in (None, "static"):
        return response
    if request.method == "GET" and request.path.startswith("/api/"):
        return response
    if request.method not in ("POST", "PUT", "DELETE", "PATCH"):
        return response

    try:
        activity = UserActivity(
            user=current_user,
            action=request.endpoint or request.path,
            details=f"{request.method} {request.path}",
            ip_address=request.remote_addr,
            timestamp=datetime.utcnow(),
        )
        activity.save()
    except Exception:
        pass
    return response


@main_bp.route("/")
@login_required
def index():
    if not DB_ENABLED:
        return render_template(
            "index.html",
            total_detections=0,
            authorized_visitors=0,
            unauthorized_visitors=0,
            active_visitors=0,
        )
    total_detections = DetectionResult.objects.count()
    authorized_visitors = Visitor.objects(authorized=True).count()
    unauthorized_visitors = Visitor.objects(authorized=False).count()
    active_visitors = Visitor.objects(exit_time__exists=False).count()
    return render_template(
        "index.html",
        total_detections=total_detections,
        authorized_visitors=authorized_visitors,
        unauthorized_visitors=unauthorized_visitors,
        active_visitors=active_visitors,
    )


@main_bp.route("/api/dashboard/stats")
@login_required
def get_dashboard_stats():
    if not DB_ENABLED:
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
        total_detections = DetectionResult.objects.count()
        authorized_visitors = Visitor.objects(authorized=True).count()
        unauthorized_visitors = Visitor.objects(authorized=False).count()
        active_visitors = Visitor.objects(exit_time__exists=False).count()
        return jsonify({
            "success": True,
            "stats": {
                "total_detections": total_detections,
                "authorized_visitors": authorized_visitors,
                "unauthorized_visitors": unauthorized_visitors,
                "active_visitors": active_visitors
            }
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

