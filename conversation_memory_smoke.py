"""隔离数据库中的历史对话 / seq / SQLite任务表契约测试。"""

import importlib.util
import os
import sqlite3
import tempfile


root = tempfile.mkdtemp(prefix="middot-conversation-memory-")
db_path = os.path.join(root, "middot.db")
os.environ["MIDDOT_DB_PATH"] = db_path
os.environ["MIDDOT_DEVICE_SECRET"] = "conversation-memory-test-secret-20260813"
os.environ["MIDDOT_MEMORY_IDLE_S"] = "0"
app_path = os.environ.get("MIDDOT_APP_TEST_PATH", os.path.join(os.path.dirname(__file__), "app_v2.py"))
spec = importlib.util.spec_from_file_location("middot_conversation_test_app", app_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


did = "1" * 32
conn = module._db_connect()
conn.execute(
    "INSERT INTO devices(device_id,created_at,last_seen_at) VALUES(?,?,?)",
    (did, module._now(), module._now()),
)
conn.commit(); conn.close()

conversation_id = module._conversation_create(did, "我和阿杰约饭")
assert module._conversation_append_event(conversation_id, did, "user", "阿杰一般从浙大出发") == 1
assert module._conversation_append_event(conversation_id, did, "assistant", "知道了，本次先按浙大理解") == 2

captured = []


def fake_extract(events, context_events, current_profile):
    captured.append((events, context_events, current_profile))
    return [{
        "kind": "person", "entity_key": "阿杰", "field_name": "常用出发地",
        "candidate_value": "浙江大学", "confidence": 0.8,
        "evidence_summary": "阿杰一般从浙大出发", "status": "candidate",
    }]


module._memory_compile_extract = fake_extract
first = module.memory_worker_once("test-worker")
assert first and first["status"] == "done", first
conn = module._db_connect()
conv = conn.execute("SELECT last_seq,last_compiled_seq FROM conversations WHERE id=?", (conversation_id,)).fetchone()
assert tuple(conv) == (2, 2), tuple(conv)
candidate = conn.execute("SELECT * FROM memory_candidates WHERE device_id=?", (did,)).fetchone()
assert candidate and candidate["status"] == "candidate"
conn.close()

assert module._conversation_append_event(conversation_id, did, "user", "他现在可能搬到滨江了") == 3
assert module._conversation_append_event(conversation_id, did, "assistant", "你是说阿杰吗？") == 4
second = module.memory_worker_once("test-worker")
assert second and second["from_seq"] == 3 and second["target_seq"] == 4, second
assert [event["seq"] for event in captured[-1][0]] == [3, 4]
assert [event["seq"] for event in captured[-1][1]] == [1, 2]

# API 能列出、恢复并继续同一个持久对话。
client = module.app.test_client()
client.set_cookie(module.DEVICE_COOKIE, module._device_cookie_encode(did))
listed = client.get("/api/v2/conversations").get_json()
assert listed["items"][0]["id"] == conversation_id
detail = client.get(f"/api/v2/conversations/{conversation_id}").get_json()
assert [event["seq"] for event in detail["events"]] == [1, 2, 3, 4]
resumed = client.post(f"/api/v2/conversations/{conversation_id}/continue").get_json()
assert resumed["session_id"] and module.session_get(resumed["session_id"])["conversation_id"] == conversation_id

# 未整理的新消息：删除后前端立即隐藏，Worker先编译最后增量再物理清除原文。
assert module._conversation_append_event(conversation_id, did, "user", "这次先按滨江算") == 5
deleted = client.delete(f"/api/v2/conversations/{conversation_id}").get_json()
assert deleted["status"] == "deleting", deleted
assert not client.get("/api/v2/conversations").get_json()["items"]
final = module.memory_worker_once("test-worker")
assert final and final["status"] == "done", final
conn = module._db_connect()
assert conn.execute("SELECT 1 FROM conversations WHERE id=?", (conversation_id,)).fetchone() is None
assert conn.execute("SELECT 1 FROM conversation_events WHERE conversation_id=?", (conversation_id,)).fetchone() is None
assert conn.execute("SELECT 1 FROM memory_jobs WHERE conversation_id=?", (conversation_id,)).fetchone() is None
candidate = conn.execute("SELECT source_conversation_id,evidence_summary FROM memory_candidates WHERE device_id=?", (did,)).fetchone()
assert candidate["source_conversation_id"] is None and candidate["evidence_summary"] is None
assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
conn.close()

# 删除与较旧范围的运行中任务竞态：旧任务只能推进旧水位，不能越过未整理尾部直接清除。
race_id = module._conversation_create(did, "删除竞态")
module._conversation_append_event(race_id, did, "user", "第一段")
module._conversation_append_event(race_id, did, "assistant", "第一段回复")
claimed = module._memory_claim_job("race-worker")
assert claimed and claimed["conversation_id"] == race_id and claimed["target_seq"] == 2
module._conversation_append_event(race_id, did, "user", "第二段")
module._conversation_append_event(race_id, did, "assistant", "第二段回复")
race_delete = client.delete(f"/api/v2/conversations/{race_id}").get_json()
assert race_delete["status"] == "deleting"
old_result = module._memory_process_compile_job(claimed, "race-worker")
conn = module._db_connect()
still_there = conn.execute(
    "SELECT last_seq,last_compiled_seq,status FROM conversations WHERE id=?", (race_id,)
).fetchone()
assert still_there and tuple(still_there) == (4, 0, "deleting"), tuple(still_there or ())
conn.close()
race_final = module.memory_worker_once("race-worker")
assert race_final and race_final["target_seq"] == 4
conn = module._db_connect()
assert conn.execute("SELECT 1 FROM conversations WHERE id=?", (race_id,)).fetchone() is None
conn.close()

print("CONVERSATION_MEMORY_SMOKE_OK")
