"""
智能中间点推荐系统 v2 — 多 Agent 架构
======================================
Agent1（规划）:  LLM 理解需求 → 结构化搜索参数
Agent2（搜索）:  LLM + 受控工具 → 搜索候选地点（上下文精简）
路线计算：纯 Python 直接调高德 API → A/B 分别计算
Agent3（总结）:  LLM 生成推荐文字

入口：python -m middot 或 python app_v2_modular.py
"""

from flask import Flask
from flask_cors import CORS
from .config import DEEPSEEK_API_KEY, AMAP_KEY, PORT, SESSION_TTL
from .extensions import init_extensions, llm_client
from .models.db import init_middot_db
from .routes import register_routes


def create_app() -> Flask:
    """应用工厂"""
    app = Flask(__name__, static_folder="../static")
    CORS(app)

    init_extensions(app)
    init_middot_db()
    register_routes(app)

    return app


def print_startup_info():
    """打印启动信息"""
    print("=" * 60)
    print("  智能中间点推荐系统 v2（多 Agent 架构 - 模块化版）")
    print("=" * 60)
    print(f"  DeepSeek API Key: {'已配置' if DEEPSEEK_API_KEY else '未配置 ⚠️'}")
    print(f"  高德地图 API Key:  {'已配置' if AMAP_KEY else '未配置 ⚠️'}")
    print(f"  Session 缓存：内存（TTL {SESSION_TTL // 3600} 小时）")
    print("=" * 60)
    print(f"  访问地址：http://localhost:{PORT}")
    print("=" * 60)
