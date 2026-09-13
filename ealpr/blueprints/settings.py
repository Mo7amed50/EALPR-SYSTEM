"""
blueprints/settings.py — System settings routes (admin only).

Routes:
    GET        /settings
    GET        /api/settings
    PUT        /api/settings          (bulk update: confidence_threshold, etc.)
    GET        /api/settings/<id>
    PUT        /api/settings/<id>
    DELETE     /api/settings/<id>
"""
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user
from bson.objectid import ObjectId, InvalidId

from ealpr import extensions
from ealpr.decorators import db_required

from models import SystemSettings
from ealpr.utils import utc_to_cairo

settings_bp = Blueprint("settings", __name__)


@settings_bp.before_request
def check_db_available():
    if not extensions.DB_ENABLED:
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({
                "success": False,
                "message": "Database unavailable. Please try again later."
            }), 503
        flash("Database is offline. Settings are currently unavailable.", "warning")
        return redirect(url_for("main.index"))


@settings_bp.route("/settings")
@login_required
@db_required
def settings():
    if not current_user.is_admin:
        flash("Access denied. Admin privileges required.")
        return redirect(url_for("main.index"))
    all_settings = SystemSettings.objects.all()
    return render_template("settings.html", settings=all_settings)


@settings_bp.route("/api/settings", methods=["GET"])
@login_required
@db_required
def get_settings():
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403
    all_settings = SystemSettings.objects.all()
    return jsonify({
        "success": True,
        "settings": [
            {
                "id": str(s.id),
                "key": s.key,
                "value": s.value,
                "description": s.description,
                "updated_at": utc_to_cairo(s.updated_at),
            }
            for s in all_settings
        ],
    })


@settings_bp.route("/api/settings", methods=["PUT"])
@login_required
@db_required
def update_detection_settings():
    """Bulk-update specific well-known settings by key name."""
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "message": "A JSON object is required"}), 400
    try:
        if "confidence_threshold" in data:
            SystemSettings.objects(key="confidence_threshold").update_one(
                set__value=str(data["confidence_threshold"])
            )
        if "processing_mode" in data:
            SystemSettings.objects(key="processing_mode").update_one(
                set__value=data["processing_mode"]
            )
        if "auto_process" in data:
            SystemSettings.objects(key="auto_process").update_one(
                set__value=str(data["auto_process"]).lower()
            )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@settings_bp.route("/api/settings", methods=["POST"])
@login_required
@db_required
def create_setting():
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403
    data = request.form if request.form else request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "message": "Setting data is required"}), 400
    key = str(data.get("key", "")).strip()
    value = data.get("value")
    if not key or value is None:
        return jsonify({"success": False, "message": "Key and value are required"}), 400
    if SystemSettings.objects(key=key).first():
        return jsonify({"success": False, "message": "Setting key already exists"}), 409
    try:
        setting = SystemSettings(
            key=key,
            value=str(value),
            description=str(data.get("description", "")).strip(),
            updated_by=current_user,
        )
        setting.save()
        return jsonify({"success": True, "setting": {"id": str(setting.id)}}), 201
    except Exception as e:
        return jsonify({"success": False, "message": f"Error creating setting: {e}"}), 500


@settings_bp.route("/api/settings/<string:setting_id>", methods=["GET"])
@login_required
@db_required
def get_setting(setting_id):
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403
    try:
        setting = SystemSettings.objects(id=ObjectId(setting_id)).first()
        if not setting:
            return jsonify({"success": False, "message": "Setting not found"}), 404
        return jsonify({
            "success": True,
            "setting": {
                "id": str(setting.id),
                "key": setting.key,
                "value": setting.value,
                "description": setting.description,
                "updated_at": utc_to_cairo(setting.updated_at),
            },
        })
    except InvalidId:
        return jsonify({"success": False, "message": "Invalid setting ID format"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"Error fetching setting: {e}"}), 500


@settings_bp.route("/api/settings/<string:setting_id>", methods=["PUT"])
@login_required
@db_required
def update_setting(setting_id):
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403
    try:
        setting = SystemSettings.objects(id=ObjectId(setting_id)).first()
        if not setting:
            return jsonify({"success": False, "message": "Setting not found"}), 404
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"success": False, "message": "A JSON object is required"}), 400
        if "value" in data:
            setting.value = str(data["value"])
            setting.updated_by = current_user
        setting.save()
        return jsonify({"success": True})
    except InvalidId:
        return jsonify({"success": False, "message": "Invalid setting ID format"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"Error updating setting: {e}"}), 500


@settings_bp.route("/api/settings/<string:setting_id>", methods=["DELETE"])
@login_required
@db_required
def delete_setting(setting_id):
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403
    try:
        setting = SystemSettings.objects(id=ObjectId(setting_id)).first()
        if not setting:
            return jsonify({"success": False, "message": "Setting not found"}), 404
        setting.delete()
        return jsonify({"success": True})
    except InvalidId:
        return jsonify({"success": False, "message": "Invalid setting ID format"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"Error deleting setting: {e}"}), 500
