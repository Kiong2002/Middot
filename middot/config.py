"""
配置和常量模块
=============
集中管理所有配置项、常量和环境变量
"""

import os
import re
import secrets
import threading
import uuid
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# API Keys（从 amap_client 导入）
from amap_client import DEEPSEEK_API_KEY, AMAP_KEY, AMAP_JS_KEY

# ==============================================================================
# 服务配置
# ==============================================================================

# 服务端口，默认 8080，可通过环境变量 PORT 覆盖
PORT = int(os.getenv("PORT", "8080"))

# 路线计算并发上限
ROUTE_MAX_WORKERS = int(os.getenv("ROUTE_MAX_WORKERS", "5"))
ROUTE_LEG_RETRY = int(os.getenv("ROUTE_LEG_RETRY", "1"))

# ==============================================================================
# Session 配置
# ==============================================================================

SESSION_TTL = 3600  # 1 小时

# ==============================================================================
# 数据库配置
# ==============================================================================

MIDDOT_DB_PATH = os.getenv(
    "MIDDOT_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "middot.db")
)

DEVICE_COOKIE = "middot_did"
DEVICE_COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 年
LEGACY_DEVICE_CLAIM_TTL_S = 10

ADMIN_COOKIE = "middot_admin"
ADMIN_COOKIE_MAX_AGE = 12 * 60 * 60
ADMIN_USERNAME = os.getenv("MIDDOT_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("MIDDOT_ADMIN_PASSWORD", "1234")

# 管理员登录限流
_admin_login_attempts: dict[str, list[float]] = {}
_admin_login_lock = threading.Lock()

# ==============================================================================
# 房间配置
# ==============================================================================

ROOM_CODE_ALPHABET = "0123456789"
ROOM_CODE_LEN = 6
ROOM_TTL_S = 60 * 60 * 24  # 未锁定：24h 无活跃回收
ROOM_LOCK_TTL_S = 60 * 60 * 24 * 7  # 锁定后暂存 7 天
ROOM_CODE_REUSE_COOLDOWN_S = 60 * 60 * 24  # 关闭后 24h 内不复用同 code

# 记忆锚点黑名单：太顺口 / 太常见 / 太像验证码
ROOM_CODE_BLACKLIST = frozenset({
    "000000", "111111", "222222", "333333", "444444",
    "555555", "666666", "777777", "888888", "999999",
    "123456", "234567", "345678", "456789", "567890",
    "654321", "121212", "112233", "998877", "520520",
})

# ==============================================================================
# 对话与记忆配置
# ==============================================================================

CONVERSATION_IDLE_S = int(os.getenv("MIDDOT_MEMORY_IDLE_S", "1800"))
CONVERSATION_CONTEXT_EVENTS = 8
MEMORY_JOB_LEASE_S = 10 * 60
MEMORY_DELETE_DEADLINE_S = 24 * 60 * 60

# ==============================================================================
# 设备签名密钥
# ==============================================================================

def _load_device_signing_secret() -> str:
    """
    使用独立、持久的设备签名密钥
    API key 轮换不应让所有设备掉线
    """
    configured = str(os.getenv("MIDDOT_DEVICE_SECRET") or "").strip()
    if configured:
        if len(configured) < 32:
            raise RuntimeError("MIDDOT_DEVICE_SECRET must contain at least 32 characters")
        return configured

    db_dir = os.path.dirname(os.path.abspath(MIDDOT_DB_PATH))
    os.makedirs(db_dir, mode=0o700, exist_ok=True)
    secret_path = os.path.join(db_dir, ".middot_device_secret")
    candidate = secrets.token_hex(32)
    temp_path = f"{secret_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, candidate.encode("ascii"))
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(temp_path, secret_path)
        except FileExistsError:
            pass
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

    try:
        os.chmod(secret_path, 0o600)
        with open(secret_path, "r", encoding="ascii") as handle:
            persisted = handle.read().strip()
    except OSError as exc:
        raise RuntimeError("unable to load persistent device signing secret") from exc
    if not re.fullmatch(r"[0-9a-f]{64}", persisted):
        raise RuntimeError("persistent device signing secret is invalid")
    return persisted


DEVICE_SIGNING_SECRET = _load_device_signing_secret()
