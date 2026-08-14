"""
通用工具函数
============
提供 JSON 提取、POI 压缩等通用工具
"""

import json
import re


def extract_json(text: str) -> dict | list | None:
    """从 LLM 回答中提取第一个合法 JSON 对象或数组"""
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def compact_poi(poi: dict) -> dict:
    """压缩 POI 数据，移除冗余字段"""
    return {
        "id": poi.get("id"),
        "name": poi.get("name"),
        "address": poi.get("address"),
        "lng": poi.get("lng") or poi.get("location", {}).get("lng"),
        "lat": poi.get("lat") or poi.get("location", {}).get("lat"),
        "category": poi.get("category"),
        "reason": poi.get("reason", ""),
    }


def format_route(route: dict) -> dict:
    """格式化路线数据"""
    return {
        "distance": route.get("distance"),
        "duration": route.get("duration"),
        "transit_mode": route.get("transit_mode"),
        "steps": route.get("steps", []),
    }
