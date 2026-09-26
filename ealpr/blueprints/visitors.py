"""
blueprints/visitors.py — Visitor management routes.

Routes:
    GET      /visitors            (read-only list)
    GET/POST /manage_visitors     (full management UI)
    POST     /api/visitors
    GET      /api/visitors/<id>
    PUT      /api/visitors/<id>
    DELETE   /api/visitors/<id>
    POST     /api/visitors/clear
"""
from datetime import datetime

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user
from mongoengine.queryset.visitor import Q

from ealpr import extensions
from ealpr.decorators import db_required
from models import Visitor
from ealpr.utils import utc_to_cairo

visitors_bp = Blueprint("visitors", __name__)


@visitors_bp.before_request
def check_db_available():
    if not extensions.DB_ENABLED:
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({
                "success": False,
                "message": "Database unavailable. Please try again later."
            }), 503
        flash("Database is offline. Visitor management is currently unavailable.", "warning")
        return redirect(url_for("main.index"))


def _find_visitor(visitor_id_str):
    """Resolve a Visitor by sequence visitor_id (int), MongoDB ObjectId, or visitor_code."""
    if not visitor_id_str:
        return None
    visitor_id_str = str(visitor_id_str).strip()
    if visitor_id_str.isdigit():
        v = Visitor.objects(visitor_id=int(visitor_id_str)).first()
        if v:
            return v
    from bson.objectid import ObjectId
    if ObjectId.is_valid(visitor_id_str):
        v = Visitor.objects(id=ObjectId(visitor_id_str)).first()
        if v:
            return v
    v = Visitor.objects(visitor_code=visitor_id_str).first()
    if v:
        return v
    return Visitor.objects(license_plate=visitor_id_str).first()


# ── Read-only visitor list ───────────────────────────────────────────────────

@visitors_bp.route("/visitors")
@login_required
@db_required
def visitors():
    all_visitors = Visitor.objects.all()
    return render_template("visitors.html", visitors=all_visitors)


# ── Full management UI ───────────────────────────────────────────────────────

@visitors_bp.route("/manage_visitors", methods=["GET", "POST"])
@login_required
@db_required
def manage_visitors():
    if request.method == "POST":
        action = request.form.get("action")
        visitor_id = request.form.get("visitor_id")

        if action == "clear":
            if not current_user.is_admin:
                flash("Access denied. Admin privileges required.", "error")
                return redirect(url_for("visitors.manage_visitors"))
            try:
                Visitor.objects.delete()
                flash("All visitors cleared successfully", "success")
            except Exception as e:
                flash(f"Failed to clear visitors: {e}", "error")
            return redirect(url_for("visitors.manage_visitors"))

        if not visitor_id:
            flash("Visitor ID is required", "error")
            return redirect(url_for("visitors.manage_visitors"))

        visitor = _find_visitor(visitor_id)
        if not visitor:
            flash("Visitor not found", "error")
            return redirect(url_for("visitors.manage_visitors"))

        if action == "authorize":
            visitor.authorized = True
            visitor.status = "authorized"
            flash("Visitor authorized successfully", "success")
        elif action == "unauthorize":
            visitor.authorized = False
            visitor.status = "unauthorized"
            flash("Visitor unauthorized successfully", "success")
        elif action == "delete":
            visitor.delete()
            flash("Visitor deleted successfully", "success")
            return redirect(url_for("visitors.manage_visitors"))
        elif action == "update":
            name = request.form.get("name")
            visitor_code = request.form.get("visitor_code")
            license_plate = request.form.get("license_plate")
            if name:
                visitor.name = name
            if visitor_code:
                visitor.visitor_code = visitor_code
            if license_plate:
                visitor.license_plate = license_plate
            flash("Visitor information updated successfully", "success")

        visitor.save()
        return redirect(url_for("visitors.manage_visitors"))

    search_query = request.args.get("search_query", "").strip()
    id_param = request.args.get("id", "").strip()

    if id_param:
        target = _find_visitor(id_param)
        if target:
            return render_template("manage_visitors.html", visitors=[target], search_query=id_param)

    query = Visitor.objects
    if search_query:
        query = query.filter(
            Q(name__icontains=search_query)
            | Q(visitor_code__icontains=search_query)
            | Q(license_plate__icontains=search_query)
        )
    all_visitors = query.order_by("-entry_datetime_utc").all()
    return render_template("manage_visitors.html", visitors=all_visitors, search_query=search_query)


# ── API: Add visitor ─────────────────────────────────────────────────────────

@visitors_bp.route("/api/visitors", methods=["POST"])
@login_required
@db_required
def add_visitor():
    data = request.get_json(silent=True) if request.is_json else request.form
    data = data or {}
    name = str(data.get("name", "")).strip()
    visitor_code = str(data.get("visitor_code", "")).strip() or None
    license_plate = str(data.get("license_plate", "")).strip()
    status = str(data.get("status", "pending")).strip().lower()
    responsible_department = data.get("responsible_department")
    general_department = data.get("general_department")

    if not all([name, license_plate]):
        return jsonify({"success": False, "message": "Name and license plate are required"}), 400
    if status not in {"pending", "authorized", "unauthorized"}:
        return jsonify({"success": False, "message": "Invalid visitor status"}), 400

    duplicate_query = Q(license_plate=license_plate)
    if visitor_code:
        duplicate_query = Q(visitor_code=visitor_code) | duplicate_query
    existing_visitor = Visitor.objects(duplicate_query).first()
    if existing_visitor:
        return jsonify({"success": False, "message": "Visitor with this code or license plate already exists"}), 400

    try:
        visitor = Visitor(
            name=name,
            visitor_code=visitor_code,
            license_plate=license_plate,
            status=status,
            responsible_department=responsible_department,
            general_department=general_department,
            authorized=status == "authorized",
            entry_datetime_utc=datetime.utcnow(),
        )
        visitor.save()
        return jsonify({
            "success": True,
            "message": "Visitor added successfully",
            "visitor": {
                "id": str(visitor.id),
                "name": visitor.name,
                "visitor_id": str(visitor.visitor_id) if visitor.visitor_id else None,
                "visitor_code": visitor.visitor_code,
                "license_plate": visitor.license_plate,
                "status": visitor.status,
                "responsible_department": visitor.responsible_department,
                "general_department": visitor.general_department,
            },
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to add visitor: {e}"}), 500


# ── API: Get visitor ─────────────────────────────────────────────────────────

@visitors_bp.route("/api/visitors/<string:visitor_id>", methods=["GET"])
@login_required
@db_required
def get_visitor(visitor_id):
    visitor = _find_visitor(visitor_id)
    if not visitor:
        return jsonify({"success": False, "message": "Visitor not found"}), 404
    return jsonify({
        "success": True,
        "id": str(visitor.id),
        "name": visitor.name,
        "visitor_id": str(visitor.visitor_id) if visitor.visitor_id else "",
        "visitor_code": visitor.visitor_code or "",
        "license_plate": visitor.license_plate,
        "entry_time": visitor.entry_time or "",
        "entry_date": visitor.entry_date or "",
        "exit_time": utc_to_cairo(visitor.exit_time) if visitor.exit_time else None,
        "authorized": visitor.authorized,
        "status": visitor.status,
        "responsible_department": visitor.responsible_department or None,
        "general_department": visitor.general_department or None,
    })


# ── API: Update visitor ──────────────────────────────────────────────────────

@visitors_bp.route("/api/visitors/<string:visitor_id>", methods=["PUT"])
@login_required
@db_required
def update_visitor(visitor_id):
    visitor = _find_visitor(visitor_id)
    if not visitor:
        return jsonify({"success": False, "message": "Visitor not found"}), 404

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "message": "A JSON object is required"}), 400
    if "name" in data:
        visitor.name = data["name"]
    if "visitor_code" in data and data["visitor_code"]:
        visitor.visitor_code = data["visitor_code"]
    if "license_plate" in data:
        visitor.license_plate = data["license_plate"]
    if "status" in data:
        if data["status"] not in {"pending", "authorized", "unauthorized"}:
            return jsonify({"success": False, "message": "Invalid visitor status"}), 400
        visitor.status = data["status"]
        visitor.authorized = data["status"] == "authorized"
    elif "authorized" in data:
        visitor.authorized = bool(data["authorized"])
        visitor.status = "authorized" if visitor.authorized else "unauthorized"
    if "responsible_department" in data:
        visitor.responsible_department = data["responsible_department"]
    if "general_department" in data:
        visitor.general_department = data["general_department"]
    if "exit_time" in data:
        try:
            visitor.exit_time = (
                datetime.strptime(data["exit_time"], "%Y-%m-%d %H:%M:%S")
                if data["exit_time"] else None
            )
        except (TypeError, ValueError):
            return jsonify({"success": False, "message": "Invalid exit time format"}), 400

    entry_date_str = data.get("entry_date")
    entry_time_str = data.get("entry_time")
    if entry_date_str and entry_time_str:
        try:
            visitor.entry_datetime_utc = datetime.strptime(
                f"{entry_date_str} {entry_time_str}", "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            return jsonify({"success": False, "message": "Invalid date or time format for entry time"}), 400

    visitor.save()
    return jsonify({"success": True, "message": "Visitor updated successfully"})


# ── API: Delete visitor ──────────────────────────────────────────────────────

@visitors_bp.route("/api/visitors/<string:visitor_id>", methods=["DELETE"])
@login_required
@db_required
def delete_visitor(visitor_id):
    visitor = _find_visitor(visitor_id)
    if not visitor:
        return jsonify({"success": False, "message": "Visitor not found"}), 404
    visitor.delete()
    return jsonify({"success": True, "message": "Visitor deleted successfully"})


# ── API: Clear all visitors (admin) ──────────────────────────────────────────

@visitors_bp.route("/api/visitors/clear", methods=["POST"])
@login_required
@db_required
def clear_visitors():
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied. Admin privileges required."}), 403
    try:
        Visitor.objects.delete()
        return jsonify({"success": True, "message": "All visitors cleared successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to clear visitors: {e}"}), 500
