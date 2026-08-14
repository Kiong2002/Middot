"""
路由注册模块
============
将所有路由蓝图注册到 Flask 应用
"""

from flask import Flask, send_from_directory, jsonify
from ..config import AMAP_JS_KEY, PORT, SESSION_TTL, DEEPSEEK_API_KEY, AMAP_KEY


def register_routes(app: Flask):
    """注册所有路由"""

    # 静态页面和配置路由
    @app.route("/")
    def index():
        return send_from_directory("../static", "index.html")

    @app.route("/api/config")
    def get_config():
        return jsonify({
            "amap_key": AMAP_JS_KEY,
            "amap_js_code": "",
            "has_amap_key": bool(AMAP_JS_KEY),
            "version": "v2",
            "port": PORT,
        })

    # 注册管理员路由
    try:
        from .admin import admin_bp
        app.register_blueprint(admin_bp)
    except ImportError:
        pass

    # 其他路由模块（逐步迁移）
    # from .me import me_bp
    # from .favorites import favorites_bp
    # from .rooms import rooms_bp
    # from .search import search_bp
    # from .assistant import assistant_bp
    # from .memories import memories_bp
