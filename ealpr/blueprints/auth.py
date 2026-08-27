"""
blueprints/auth.py — Authentication routes.

Routes:
    GET/POST /login
    GET      /logout
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, login_required, logout_user, current_user
from datetime import datetime
from bson import ObjectId

from config import DEFAULT_ADMIN_PASSWORD, OFFLINE_ADMIN_PASSWORD
from models import User

auth_bp = Blueprint("auth", __name__)


def _build_offline_user():
    user = User(
        id=ObjectId("000000000000000000000001"),
        username="admin",
        is_admin=True,
        is_active=True,
    )
    user.set_password(OFFLINE_ADMIN_PASSWORD)
    return user


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    from ealpr.extensions import DB_ENABLED
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not DB_ENABLED:
            if username == "admin" and password == OFFLINE_ADMIN_PASSWORD:
                login_user(_build_offline_user())
                flash("Signed in using offline fallback mode.", "info")
                return redirect(url_for("main.index"))
            flash("Offline mode: use admin / the configured offline password.", "warning")
            return render_template("login.html")

        user = User.objects(username=username).first()

        if user:
            # Lockout check: 5 failures within 5 minutes
            if user.failed_login_attempts >= 5 and (
                datetime.utcnow() - user.last_failed_login
            ).total_seconds() < 300:
                flash("Account temporarily locked. Please try again later.")
                return render_template("login.html")

            if user.check_password(password):
                user.failed_login_attempts = 0
                user.last_login = datetime.utcnow()
                user.save()
                login_user(user, remember=bool(request.form.get("remember")))
                return redirect(url_for("main.index"))
            else:
                user.failed_login_attempts += 1
                user.last_failed_login = datetime.utcnow()
                user.save()
                if user.failed_login_attempts >= 5:
                    flash("Too many failed attempts. Account locked for 5 minutes.")
                else:
                    flash("Invalid username or password")
        else:
            flash("Invalid username or password")

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
