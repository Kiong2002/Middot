"""管理台认证、权限和记忆运维接口的隔离 smoke test。"""

import importlib.util
import os
import tempfile


root = tempfile.mkdtemp(prefix="middot-admin-memory-")
os.environ["MIDDOT_DB_PATH"] = os.path.join(root, "middot.db")
os.environ["MIDDOT_DEVICE_SECRET"] = "admin-memory-smoke-secret-20260813"
os.environ["MIDDOT_ADMIN_USERNAME"] = "admin"
os.environ["MIDDOT_ADMIN_PASSWORD"] = "1234"
app_path = os.environ.get(
    "MIDDOT_APP_TEST_PATH", os.path.join(os.path.dirname(__file__), "app_v2.py")
)
spec = importlib.util.spec_from_file_location("middot_admin_test_app", app_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


client = module.app.test_client()
assert client.get("/admin").status_code == 200
assert client.get("/admin/memory").status_code == 200
assert client.get("/api/admin/memory/overview").status_code == 401
assert client.post(
    "/api/admin/login", json={"username": "admin", "password": "wrong"}
).status_code == 401
logged_in = client.post(
    "/api/admin/login", json={"username": "admin", "password": "1234"}
)
assert logged_in.status_code == 200, logged_in.get_json()
assert client.get("/api/admin/session").get_json()["authenticated"] is True

# 读接口可用，写接口还必须带显式管理请求头，避免跨站表单触发。
assert client.get("/api/admin/memory/overview").status_code == 200
assert client.post("/api/admin/memory/jobs/cleanup").status_code == 403
admin_headers = {"X-Middot-Admin": "1"}
assert client.post(
    "/api/admin/memory/jobs/cleanup", headers=admin_headers
).status_code == 200

# 构造一条未整理对话；管理台应看到水位并能把任务调整为立即执行。
did = "a" * 32
conn = module._db_connect()
conn.execute(
    "INSERT INTO devices(device_id,created_at,last_seen_at) VALUES(?,?,?)",
    (did, module._now(), module._now()),
)
conn.commit()
conn.close()
conversation_id = module._conversation_create(did, "管理台测试对话")
module._conversation_append_event(conversation_id, did, "user", "阿杰在浙大")
conversations = client.get("/api/admin/memory/conversations").get_json()["items"]
row = next(item for item in conversations if item["id"] == conversation_id)
assert row["last_seq"] == 1 and row["last_compiled_seq"] == 0

jobs = client.get("/api/admin/memory/jobs").get_json()["items"]
job = next(item for item in jobs if item["conversation_id"] == conversation_id)
assert client.post(
    f"/api/admin/memory/jobs/{job['id']}/run", headers=admin_headers
).status_code == 200
conn = module._db_connect()
queued = conn.execute("SELECT job_type,status,run_after,target_seq FROM memory_jobs WHERE id=?", (job["id"],)).fetchone()
assert queued["job_type"] == "manual_compile"
assert queued["status"] == "pending" and queued["run_after"] <= module._now()
assert queued["target_seq"] == 1
conn.close()

# Worker 心跳会出现在总览；候选记忆可从管理台忽略。
module.memory_worker_heartbeat("smoke-worker", 12345, module._now())
overview = client.get("/api/admin/memory/overview").get_json()
assert overview["workers"][0]["online"] is True
conn = module._db_connect()
now = module._now()
cursor = conn.execute(
    "INSERT INTO memory_candidates(device_id,kind,entity_key,field_name,candidate_value,confidence,"
    "status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
    (did, "person", "阿杰", "常用出发地", "浙江大学", 0.8, "candidate", now, now),
)
candidate_id = cursor.lastrowid
conn.commit()
conn.close()
assert client.post(
    f"/api/admin/memory/candidates/{candidate_id}/dismiss", headers=admin_headers
).status_code == 200
assert not client.get("/api/admin/memory/candidates").get_json()["items"]

assert client.post("/api/admin/logout", headers=admin_headers).status_code == 200
assert client.get("/api/admin/session").get_json()["authenticated"] is False
assert client.get("/api/admin/memory/jobs").status_code == 401
print("ADMIN_MEMORY_SMOKE_OK")
