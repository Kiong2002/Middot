"""
高德地图 API 工具函数集
======================
封装了地理编码、路线规划、POI 搜索等底层高德 REST API 调用。
所有函数均为纯函数，不依赖 Flask 上下文。
"""

import math
import datetime
import os
import time
import requests
from requests import exceptions as request_exceptions
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
AMAP_KEY: str         = os.getenv("AMAP_KEY", "")
AMAP_JS_KEY: str      = os.getenv("AMAP_JS_KEY", AMAP_KEY)


# ─────────────────────────────────────────────
# 地理编码
# ─────────────────────────────────────────────

def amap_geocode(address: str, city: str | None = None) -> dict:
    """地址 → 经纬度坐标。city 参数可缩小歧义（'北京北航' 有时被匹配到深圳，传 city='北京' 可锁死）。"""
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {"key": AMAP_KEY, "address": address, "output": "json"}
    if city:
        params["city"] = city
    # 公网服务器到高德偶尔会在连接或读取阶段瞬时超时。地理编码是幂等请求，
    # 因此仅对网络超时做短退避重试；业务错误仍立即返回，避免掩盖真实问题。
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=(6, 12))
            break
        except (request_exceptions.ConnectTimeout, request_exceptions.ReadTimeout):
            if attempt == 2:
                return {"success": False, "error": "定位服务暂时不可用，请稍后重试"}
            time.sleep(0.35 * (attempt + 1))
    assert resp is not None
    data = resp.json()
    if data.get("status") == "1" and data.get("geocodes"):
        geocode = data["geocodes"][0]
        lng, lat = geocode["location"].split(",")
        return {
            "success": True,
            "lng": float(lng),
            "lat": float(lat),
            "formatted_address": geocode.get("formatted_address", address),
            "city": geocode.get("city", ""),
        }
    return {"success": False, "error": data.get("info", "地理编码失败")}


# ─────────────────────────────────────────────
# 出发时间解析
# ─────────────────────────────────────────────

def _parse_departure_time(departure_time: str | None) -> tuple[str, str]:
    """
    解析出发时间字符串，返回 (date_str, time_str)。
    格式：'HH:MM' 或 'YYYY-MM-DD HH:MM'，None 时默认下一工作日中午 12:00。
    """
    now = datetime.datetime.now()

    if not departure_time:
        candidate = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if now.hour >= 12:
            candidate += datetime.timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += datetime.timedelta(days=1)
        return candidate.strftime("%Y-%m-%d"), "12:00"

    departure_time = departure_time.strip()
    if len(departure_time) == 5 and ":" in departure_time:
        h, m = map(int, departure_time.split(":"))
        candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if candidate <= now:
            candidate += datetime.timedelta(days=1)
        return candidate.strftime("%Y-%m-%d"), departure_time
    else:
        try:
            dt = datetime.datetime.strptime(departure_time, "%Y-%m-%d %H:%M")
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
        except ValueError:
            return now.strftime("%Y-%m-%d"), "12:00"


# ─────────────────────────────────────────────
# 路线规划
# ─────────────────────────────────────────────

def amap_driving_route(
    origin_lng: float, origin_lat: float,
    dest_lng: float, dest_lat: float,
    strategy: int = 0,
) -> dict:
    """驾车路线规划"""
    url = "https://restapi.amap.com/v3/direction/driving"
    params = {
        "key": AMAP_KEY,
        "origin": f"{origin_lng},{origin_lat}",
        "destination": f"{dest_lng},{dest_lat}",
        "strategy": strategy,
        "output": "json",
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if data.get("status") == "1" and data.get("route", {}).get("paths"):
        path = data["route"]["paths"][0]
        duration_s = int(path.get("duration", 0))
        distance_m = int(path.get("distance", 0))
        return {
            "success": True,
            "mode": "driving",
            "duration_minutes": round(duration_s / 60),
            "distance_km": round(distance_m / 1000, 1),
            "duration_text": f"{round(duration_s / 60)}分钟",
            "distance_text": f"{round(distance_m / 1000, 1)}公里",
        }
    return {"success": False, "error": "驾车路线规划失败", "mode": "driving"}


def amap_transit_route(
    origin_lng: float, origin_lat: float,
    dest_lng: float, dest_lat: float,
    city: str = "北京",
    departure_time: str | None = None,
) -> dict:
    """公交 / 地铁路线规划，支持指定出发时间"""
    date_str, time_str = _parse_departure_time(departure_time)
    url = "https://restapi.amap.com/v3/direction/transit/integrated"
    params = {
        "key": AMAP_KEY,
        "origin": f"{origin_lng},{origin_lat}",
        "destination": f"{dest_lng},{dest_lat}",
        "city": city,
        "strategy": 0,
        "nightflag": 0,
        "date": date_str,
        "time": time_str,
        "output": "json",
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if data.get("status") == "1" and data.get("route", {}).get("transits"):
        transit = data["route"]["transits"][0]
        duration_s = int(transit.get("duration", 0))

        segments = transit.get("segments", [])
        total_distance_m = 0
        for seg in segments:
            walking = seg.get("walking") or {}
            total_distance_m += int(walking.get("distance", 0))
            bus = seg.get("bus") or {}
            for line in bus.get("buslines", []):
                total_distance_m += int(line.get("distance", 0))

        lines = []
        for seg in segments:
            walking = seg.get("walking") or {}
            walk_s = int(walking.get("duration", 0))
            bus = seg.get("bus") or {}
            if walk_s >= 60:
                lines.append(f"步行{round(walk_s / 60)}分钟")
            for line in bus.get("buslines", []):
                name = line.get("name", "")
                if name:
                    lines.append(name)

        summary = " → ".join(lines) or "公共交通"
        distance_m = total_distance_m or int(data["route"].get("distance", 0))
        return {
            "success": True,
            "mode": "transit",
            "duration_minutes": round(duration_s / 60),
            "distance_km": round(distance_m / 1000, 1),
            "duration_text": f"{round(duration_s / 60)}分钟",
            "distance_text": f"{round(distance_m / 1000, 1)}公里",
            "line_summary": summary,
        }
    return {"success": False, "error": "公交路线规划失败", "mode": "transit"}


def amap_walking_route(
    origin_lng: float, origin_lat: float,
    dest_lng: float, dest_lat: float,
) -> dict:
    """步行路线规划"""
    url = "https://restapi.amap.com/v3/direction/walking"
    params = {
        "key": AMAP_KEY,
        "origin": f"{origin_lng},{origin_lat}",
        "destination": f"{dest_lng},{dest_lat}",
        "output": "json",
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if data.get("status") == "1" and data.get("route", {}).get("paths"):
        path = data["route"]["paths"][0]
        duration_s = int(path.get("duration", 0))
        distance_m = int(path.get("distance", 0))
        return {
            "success": True,
            "mode": "walking",
            "duration_minutes": round(duration_s / 60),
            "distance_km": round(distance_m / 1000, 1),
            "duration_text": f"{round(duration_s / 60)}分钟",
            "distance_text": f"{round(distance_m / 1000, 1)}公里",
        }
    return {"success": False, "error": "步行路线规划失败", "mode": "walking"}


def amap_cycling_route(
    origin_lng: float, origin_lat: float,
    dest_lng: float, dest_lat: float,
) -> dict:
    """骑行（共享单车 / 自行车）路线规划"""
    url = "https://restapi.amap.com/v4/direction/bicycling"
    params = {
        "key": AMAP_KEY,
        "origin": f"{origin_lng},{origin_lat}",
        "destination": f"{dest_lng},{dest_lat}",
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if data.get("errcode") == 0:
        paths = data.get("data", {}).get("paths", [])
        if paths:
            path = paths[0]
            duration_s = int(path.get("duration", 0))
            distance_m = int(path.get("distance", 0))
            return {
                "success": True,
                "mode": "cycling",
                "duration_minutes": round(duration_s / 60),
                "distance_km": round(distance_m / 1000, 1),
                "duration_text": f"{round(duration_s / 60)}分钟",
                "distance_text": f"{round(distance_m / 1000, 1)}公里",
            }
    return {"success": False, "error": "骑行路线规划失败", "mode": "cycling"}


def haversine_distance(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    """计算两点间 Haversine 直线距离（千米）"""
    R = 6371
    dlng = math.radians(lng2 - lng1)
    dlat = math.radians(lat2 - lat1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def amap_get_best_route(
    origin_lng: float, origin_lat: float,
    dest_lng: float, dest_lat: float,
    city: str = "北京",
    prefer: str = "auto",
    departure_time: str | None = None,
) -> dict:
    """
    自动对比多种交通方式，返回最快方案。
    prefer: auto | transit | driving | walking | cycling
    """
    dist_km = haversine_distance(origin_lng, origin_lat, dest_lng, dest_lat)

    if prefer == "driving":
        return amap_driving_route(origin_lng, origin_lat, dest_lng, dest_lat)

    tasks: dict[str, callable] = {}
    if dist_km < 2.5:
        tasks["walking"] = lambda: amap_walking_route(origin_lng, origin_lat, dest_lng, dest_lat)
    if prefer in ("auto", "cycling") and dist_km < 8:
        tasks["cycling"] = lambda: amap_cycling_route(origin_lng, origin_lat, dest_lng, dest_lat)
    if prefer in ("auto", "transit"):
        tasks["transit"] = lambda: amap_transit_route(
            origin_lng, origin_lat, dest_lng, dest_lat, city, departure_time)
    if prefer == "auto":
        tasks["driving"] = lambda: amap_driving_route(
            origin_lng, origin_lat, dest_lng, dest_lat)

    results: dict[str, dict] = {}
    if not tasks:
        return {"success": False, "error": "无可用交通方式", "mode": "unknown"}

    # 多方式并发（各方式互不依赖）
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futs = {pool.submit(fn): mode for mode, fn in tasks.items()}
        for fut in futs:
            mode = futs[fut]
            try:
                r = fut.result()
            except Exception:
                continue
            if r.get("success"):
                results[mode] = r

    if not results:
        return {"success": False, "error": "所有交通方式均查询失败", "mode": "unknown"}

    best = min(results.values(), key=lambda x: x.get("duration_minutes", 9999))
    best["all_modes"] = {
        mode: {
            "duration_minutes": r.get("duration_minutes"),
            "duration_text":    r.get("duration_text"),
            "distance_text":    r.get("distance_text"),
            "line_summary":     r.get("line_summary", ""),
        }
        for mode, r in results.items()
    }
    return best


# ─────────────────────────────────────────────
# POI 搜索
# ─────────────────────────────────────────────

def amap_search_nearby(
    center_lng: float, center_lat: float,
    keyword: str,
    radius: int = 3000,
    sort_by: str = "distance",
) -> dict:
    """周边 POI 搜索"""
    url = "https://restapi.amap.com/v3/place/around"
    params = {
        "key": AMAP_KEY,
        "location": f"{center_lng},{center_lat}",
        "keywords": keyword,
        "radius": radius,
        "sortrule": sort_by,
        "output": "json",
        "offset": 25,
        "page": 1,
        "extensions": "all",
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if data.get("status") == "1":
        pois = []
        for poi in data.get("pois", []):
            location = poi.get("location", "").split(",")
            if len(location) != 2:
                continue
            try:
                rating = float(poi.get("biz_ext", {}).get("rating", 0) or 0)
                cost = poi.get("biz_ext", {}).get("cost", "")
                pois.append({
                    "id": poi.get("id", ""),
                    "name": poi.get("name", ""),
                    "address": poi.get("address", ""),
                    "lng": float(location[0]),
                    "lat": float(location[1]),
                    "rating": rating,
                    "cost_per_person": cost,
                    "type": poi.get("type", ""),
                    "tel": poi.get("tel", ""),
                    "distance": int(poi.get("distance", 0)),
                    "photos": [p.get("url", "") for p in poi.get("photos", [])[:2]],
                })
            except (ValueError, TypeError):
                pass
        return {"success": True, "count": len(pois), "pois": pois}
    return {"success": False, "error": data.get("info", "搜索失败"), "pois": []}


# ─────────────────────────────────────────────
# 中点计算 / 会面点
# ─────────────────────────────────────────────

def find_balanced_midpoint(
    lng1: float, lat1: float, lng2: float, lat2: float
) -> dict:
    """
    计算两点地理中点（经纬度平均值），并根据两地直线距离给出建议搜索半径。
    建议半径 = 两地距离 × 35%，最小 500 m，最大 8 km。
    """
    mid_lng = (lng1 + lng2) / 2
    mid_lat = (lat1 + lat2) / 2
    dist_km = haversine_distance(lng1, lat1, lng2, lat2)
    radius_m = int(max(500, min(8000, dist_km * 1000 * 0.35)))
    return {
        "midpoint": {"lng": mid_lng, "lat": mid_lat},
        "total_distance_km": round(dist_km, 1),
        "suggested_search_radius_m": radius_m,
        "note": (
            f"两地直线距离 {round(dist_km, 1)}km，"
            f"建议以中点为圆心搜索半径 {radius_m}m 内的地点。"
        ),
    }


def fair_meeting_point(participants: list[dict]) -> dict:
    """
    N 人的建议会面中心点 + 搜索半径。
    实现为几何质心（各点经纬度平均），半径 = 最远参与者到质心距离 × 60%（500m~8km）。
    真正的 min-max 公平优化在下游 calculate_routes_multi 里通过评分实现。

    participants: [{"lng":..., "lat":..., ...}, ...]
    """
    if not participants:
        return {"midpoint": {"lng": 0, "lat": 0}, "suggested_search_radius_m": 3000}
    n = len(participants)
    mid_lng = sum(p["lng"] for p in participants) / n
    mid_lat = sum(p["lat"] for p in participants) / n
    max_km = max(
        haversine_distance(p["lng"], p["lat"], mid_lng, mid_lat)
        for p in participants
    )
    radius_m = int(max(500, min(8000, max_km * 1000 * 0.6)))
    return {
        "midpoint": {"lng": mid_lng, "lat": mid_lat},
        "n_participants": n,
        "max_distance_to_center_km": round(max_km, 1),
        "suggested_search_radius_m": radius_m,
    }


# ─────────────────────────────────────────────
# 行政区边界 + 区域内搜索
# ─────────────────────────────────────────────

def amap_district_polygon(city: str, district: str | None = None) -> dict:
    """
    获取行政区边界（返回矩形 bounding box + 中心点）。
    district 为 None 时返回整城市；否则返回该区。

    返回：{"success":True, "name":..., "center":{lng,lat},
           "bbox":{min_lng,min_lat,max_lng,max_lat}}
    """
    keyword = district or city
    url = "https://restapi.amap.com/v3/config/district"
    params = {
        "key":          AMAP_KEY,
        "keywords":     keyword,
        "subdistrict":  0,
        "extensions":   "all",
        "output":       "json",
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if data.get("status") != "1" or not data.get("districts"):
        return {"success": False, "error": data.get("info", f"未找到「{keyword}」")}

    dist_obj = data["districts"][0]
    center_str  = dist_obj.get("center", "")
    polyline    = dist_obj.get("polyline", "")

    try:
        c_lng, c_lat = [float(x) for x in center_str.split(",")]
    except (ValueError, AttributeError):
        return {"success": False, "error": "行政区返回缺少 center 字段"}

    lngs, lats = [], []
    for sub in polyline.split("|"):
        for pt in sub.split(";"):
            if "," in pt:
                try:
                    x, y = pt.split(",")
                    lngs.append(float(x)); lats.append(float(y))
                except ValueError:
                    continue

    if not lngs:
        # 没边界数据时用中心点小范围作为兜底
        bbox = {"min_lng": c_lng - 0.05, "min_lat": c_lat - 0.05,
                "max_lng": c_lng + 0.05, "max_lat": c_lat + 0.05}
    else:
        bbox = {"min_lng": min(lngs), "min_lat": min(lats),
                "max_lng": max(lngs), "max_lat": max(lats)}

    return {
        "success": True,
        "name":    dist_obj.get("name", keyword),
        "level":   dist_obj.get("level", ""),
        "adcode":  dist_obj.get("adcode", ""),
        "center":  {"lng": c_lng, "lat": c_lat},
        "bbox":    bbox,
    }


def amap_search_in_area(
    bbox: dict, keyword: str,
    sort_by: str = "distance",
    max_pages: int = 2,
) -> dict:
    """
    在给定 bounding box 内搜索 POI（高德 /v3/place/polygon，2 点即矩形）。
    bbox: {"min_lng","min_lat","max_lng","max_lat"}
    """
    url = "https://restapi.amap.com/v3/place/polygon"
    # 高德矩形格式：左上|右下 → (min_lng,max_lat)|(max_lng,min_lat)
    polygon_str = f"{bbox['min_lng']},{bbox['max_lat']}|{bbox['max_lng']},{bbox['min_lat']}"

    all_pois: list[dict] = []
    for page in range(1, max_pages + 1):
        params = {
            "key":        AMAP_KEY,
            "polygon":    polygon_str,
            "keywords":   keyword,
            "output":     "json",
            "offset":     25,
            "page":       page,
            "extensions": "all",
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "1":
            if page == 1:
                return {"success": False, "error": data.get("info", "区域搜索失败"), "pois": []}
            break

        page_pois = data.get("pois", [])
        if not page_pois:
            break

        for poi in page_pois:
            location = poi.get("location", "").split(",")
            if len(location) != 2:
                continue
            try:
                biz_ext = poi.get("biz_ext") or {}
                rating  = float(biz_ext.get("rating", 0) or 0) if isinstance(biz_ext, dict) else 0.0
                cost    = biz_ext.get("cost", "")            if isinstance(biz_ext, dict) else ""
                addr    = poi.get("address", "")             if isinstance(poi.get("address"), str) else ""
                all_pois.append({
                    "id":              poi.get("id", ""),
                    "name":            poi.get("name", ""),
                    "address":         addr,
                    "lng":             float(location[0]),
                    "lat":             float(location[1]),
                    "rating":          rating,
                    "cost_per_person": cost,
                    "type":            poi.get("type", ""),
                    "tel":             poi.get("tel", "") if isinstance(poi.get("tel"), str) else "",
                    "photos":          [p.get("url", "") for p in poi.get("photos", [])[:2]],
                })
            except (ValueError, TypeError):
                continue
        if len(page_pois) < 25:
            break

    return {"success": True, "count": len(all_pois), "pois": all_pois}
