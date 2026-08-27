"""
blueprints/users.py — User management routes (admin only).

Routes:
    GET    /users
    POST   /api/users
    GET    /api/users/<user_id>
    PUT    /api/users/<user_id>
    DELETE /api/users/<user_id>
    GET    /api/users/<user_id>/activities
"""
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user
from bson.objectid import ObjectId, InvalidId
from datetime import datetime

from ealpr.extensions import DB_ENABLED
from ealpr.decorators import db_required

from models import User, UserActivity
from ealpr.utils import utc_to_cairo

users_bp = Blueprint("users", __name__)


@users_bp.before_request
def check_db_available():
    if not DB_ENABLED:
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({
                "success": False,
                "message": "Database unavailable. Please try again later."
            }), 503
        flash("Database is offline. User management is currently unavailable.", "warning")
        return redirect(url_for("main.index"))


@users_bp.route("/users")
@login_required
@db_required
def users():
    if not current_user.is_admin:
        flash("Access denied. Admin privileges required.")
        return redirect(url_for("main.index"))
    all_users = User.objects.all()
    return render_template("users.html", users=all_users)


@users_bp.route("/api/users", methods=["POST"])
@login_required
@db_required
def create_user():
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403

    username = request.form.get("username")
    password = request.form.get("password")
    is_admin = request.form.get("is_admin") == "true"

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required"}), 400
    if len(username) < 3:
        return jsonify({"success": False, "message": "Username must be at least 3 characters long"}), 400
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters long"}), 400
    if User.objects(username=username).first():
        return jsonify({"success": False, "message": "Username already exists"}), 400

    user = User(username=username, is_admin=is_admin)
    user.set_password(password)
    user.save()
    return jsonify({"success": True})


@users_bp.route("/api/users/<string:user_id>", methods=["GET"])
@login_required
@db_required
def get_user(user_id):
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403
    try:
        user = User.objects(id=ObjectId(user_id)).first()
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404
        return jsonify({
            "success": True,
            "user": {
                "id": str(user.id),
                "username": user.username,
                "is_admin": user.is_admin,
            },
        })
    except InvalidId:
        return jsonify({"success": False, "message": "Invalid user ID format"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"Error fetching user: {e}"}), 500


@users_bp.route("/api/users/<string:user_id>", methods=["PUT"])
@login_required
@db_required
def update_user(user_id):
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403
    try:
        user = User.objects(id=ObjectId(user_id)).first()
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404

        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400

        if "username" in data:
            if not data["username"]:
                return jsonify({"success": False, "message": "Username cannot be empty"}), 400
            if len(data["username"]) < 3:
                return jsonify({"success": False, "message": "Username must be at least 3 characters long"}), 400
            if User.objects(username=data["username"], id__ne=ObjectId(user_id)).first():
                return jsonify({"success": False, "message": "Username already exists"}), 400
            user.username = data["username"]

        if "password" in data and data["password"]:
            if len(data["password"]) < 6:
                return jsonify({"success": False, "message": "Password must be at least 6 characters long"}), 400
            user.set_password(data["password"])

        if "is_admin" in data:
            user.is_admin = data["is_admin"]

        user.save()
        return jsonify({"success": True, "message": "User updated successfully"})
    except InvalidId:
        return jsonify({"success": False, "message": "Invalid user ID format"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"Error updating user: {e}"}), 500


@users_bp.route("/api/users/<string:user_id>", methods=["DELETE"])
@login_required
@db_required
def delete_user(user_id):
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403
    try:
        user = User.objects(id=ObjectId(user_id)).first()
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404
        if str(user.id) == str(current_user.id):
            return jsonify({"success": False, "message": "Cannot delete your own account"}), 400
        user.delete()
        return jsonify({"success": True})
    except InvalidId:
        return jsonify({"success": False, "message": "Invalid user ID format"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"Error deleting user: {e}"}), 500


@users_bp.route("/api/users/<string:user_id>/activities")
@login_required
@db_required
def get_user_activities(user_id):
    if not current_user.is_admin:
        return jsonify({"success": False, "message": "Access denied"}), 403
    try:
        user = User.objects(id=ObjectId(user_id)).first()
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404
        activities = UserActivity.objects(user=user).order_by("-timestamp").limit(50)
        return jsonify({
            "success": True,
            "activities": [
                {
                    "action": a.action,
                    "details": a.details,
                    "ip_address": a.ip_address,
                    "timestamp": utc_to_cairo(a.timestamp),
                }
                for a in activities
            ],
        })
    except InvalidId:
        return jsonify({"success": False, "message": "Invalid user ID format"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"Error fetching activities: {e}"}), 500
