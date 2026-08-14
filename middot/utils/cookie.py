"""
Cookie 工具模块
==============
处理设备身份签名和管理员认证
"""

import hmac
import hashlib
import re
import time
from ..config import DEVICE_SIGNING_SECRET, ADMIN_COOKIE_MAX_AGE


def _device_cookie_encode(device_id: str) -> str:
    """为设备 ID 生成签名 cookie"""
    sig = hmac.new(
        DEVICE_SIGNING_SECRET.encode(), device_id.encode(), hashlib.sha256
    ).hexdigest()[:32]
    return f"{device_id}.{sig}"


def _device_cookie_decode(value: str | None) -> str | None:
    """验证并解码设备 cookie"""
    raw = str(value or "")
    if "." not in raw:
        return None
    device_id, supplied = raw.rsplit(".", 1)
    if not re.fullmatch(r"[0-9a-f]{32}", device_id):
        return None
    expected = _device_cookie_encode(device_id).rsplit(".", 1)[1]
    return device_id if hmac.compare_digest(supplied, expected) else None


def _legacy_signed_device_cookie_decode(value: str | None) -> str | None:
    """只用于切换期验证旧版 API-key 派生签名"""
    from ..config import DEEPSEEK_API_KEY, AMAP_KEY
    raw = str(value or "")
    if "." not in raw:
        return None
    device_id, supplied = raw.rsplit(".", 1)
    if not re.fullmatch(r"[0-9a-f]{32}", device_id):
        return None
    legacy_secret = hashlib.sha256(
        (DEEPSEEK_API_KEY or AMAP_KEY or "middot-local-device-secret").encode()
    ).hexdigest()
    expected = hmac.new(
        legacy_secret.encode(), device_id.encode(), hashlib.sha256
    ).hexdigest()[:32]
    return device_id if hmac.compare_digest(supplied, expected) else None


def _admin_cookie_encode(expires_at: int) -> str:
    """生成管理员 cookie"""
    payload = f"admin:{expires_at}"
    sig = hmac.new(
        DEVICE_SIGNING_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()[:32]
    return f"{payload}.{sig}"


def _admin_cookie_valid(value: str | None) -> bool:
    """验证管理员 cookie 是否有效"""
    raw = str(value or "")
    if "." not in raw:
        return False
    try:
        payload, supplied = raw.rsplit(".", 1)
        parts = payload.split(":")
        if len(parts) != 2 or parts[0] != "admin":
            return False
        expires_at = int(parts[1])
        if expires_at < int(time.time()):
            return False
        expected = hmac.new(
            DEVICE_SIGNING_SECRET.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()[:32]
        return hmac.compare_digest(supplied, expected)
    except (ValueError, IndexError):
        return False
