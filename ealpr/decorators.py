"""
decorators.py — Decorators to handle DB-offline/read-only mode.
"""
from functools import wraps
from flask import jsonify, flash, redirect, url_for, request
from ealpr.extensions import DB_ENABLED

def db_required(f):
    """
    Decorator to protect routes that require database connectivity.
    If the database is offline, redirects with a warning or returns a 503 JSON error.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not DB_ENABLED:
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.path.startswith('/api/'):
                return jsonify({
                    'success': False,
                    'message': 'Database is offline. This action is currently unavailable.'
                }), 503
            flash('Database is offline. This feature is currently unavailable.', 'danger')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function
