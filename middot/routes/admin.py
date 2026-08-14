"""
管理员路由模块
==============
处理管理员登录、会话和内存管理接口
"""

import os
import hmac
import time
import threading
from flask import Blueprint, jsonify, request, send_from_directory
from ..config import ADMIN_USERNAME, ADMIN_PASSWORD, ADMIN_COOKIE, ADMIN_COOKIE_MAX_AGE, _admin_login_lock, _admin_login_attempts
from ..models.db import get_db, _now
from ..utils.cookie import _admin_cookie_encode, _admin_cookie_valid

admin_bp = Blueprint('admin', __name__)


def _admin_required(fn):
    """管理员认证装饰器"""
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _admin_cookie_valid(request.cookies.get(ADMIN_COOKIE)):
            return jsonify({"error": "未授权"}), 401
        return fn(*args, **kwargs)
    return wrapper


@admin_bp.route("/admin")
def admin_page():
    """管理员页面"""
    response = send_from_directory("../static", "admin.html")
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
    )
    return response


@admin_bp.route("/api/admin/session")
def api_admin_session():
    """检查管理员会话状态"""
    return jsonify({"authenticated": _admin_cookie_valid(request.cookies.get(ADMIN_COOKIE))})


@admin_bp.route("/api/admin/login", methods=["POST"])
def api_admin_login():
    """管理员登录"""
    remote = str(request.remote_addr or "unknown")
    now = time.time()
    with _admin_login_lock:
        recent = [stamp for stamp in _admin_login_attempts.get(remote, []) if now - stamp < 600]
        _admin_login_attempts[remote] = recent
        if len(recent) >= 5:
            return jsonify({"error": "登录尝试过多，请 10 分钟后再试"}), 429

    data = request.get_json(silent=True) or {}
    username_ok = hmac.compare_digest(str(data.get("username") or ""), ADMIN_USERNAME)
    password_ok = hmac.compare_digest(str(data.get("password") or ""), ADMIN_PASSWORD)

    if not (username_ok and password_ok):
        with _admin_login_lock:
            _admin_login_attempts.setdefault(remote, []).append(now)
        return jsonify({"error": "账号或密码错误"}), 401

    with _admin_login_lock:
        _admin_login_attempts.pop(remote, None)

    expires_at = _now() + ADMIN_COOKIE_MAX_AGE
    response = jsonify({"ok": True, "expires_at": expires_at})
    response.set_cookie(
        ADMIN_COOKIE, _admin_cookie_encode(expires_at), max_age=ADMIN_COOKIE_MAX_AGE,
        httponly=True, samesite="Strict", secure=request.is_secure, path="/",
    )
    return response


@admin_bp.route("/api/admin/logout", methods=["POST"])
@_admin_required
def api_admin_logout():
    """管理员登出"""
    response = jsonify({"ok": True})
    response.delete_cookie(ADMIN_COOKIE, path="/")
    return response
