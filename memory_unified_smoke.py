"""Wiki 唯一事实源与偏好/人物/反馈投影同步契约测试。"""

import importlib.util
import os
import tempfile


root = tempfile.mkdtemp(prefix="middot-memory-unified-")
os.environ["MIDDOT_DB_PATH"] = os.path.join(root, "middot.db")
os.environ["MIDDOT_DEVICE_SECRET"] = "memory-unified-smoke-secret-20260814"
app_path = os.environ.get("MIDDOT_APP_TEST_PATH", os.path.join(os.path.dirname(__file__), "app_v2.py"))
spec = importlib.util.spec_from_file_location("middot_memory_unified", app_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

did = "8" * 32
now = module._now()
conn = module._db_connect()
conn.execute("INSERT INTO devices(device_id,created_at,last_seen_at) VALUES(?,?,?)", (did, now, now))
preference_id = conn.execute(
    "INSERT INTO agent_memories(device_id,category,memory_key,memory_value,source,status,created_at,updated_at) "
    "VALUES(?,?,?,?, 'explicit','confirmed',?,?)",
    (did, "transport", "default_mode", "公交", now, now),
).lastrowid
person_id = conn.execute(
    "INSERT INTO memory_people(device_id,name,relation,usual_place,city,expires_at,created_at,updated_at) "
    "VALUES(?,?,?,?,?,?,?,?)",
    (did, "阿杰", "朋友", "浙江大学紫金港校区", "杭州", now + 90 * 86400, now, now),
).lastrowid
feedback_id = conn.execute(
    "INSERT INTO memory_feedback(device_id,poi_id,poi_name,signal,reason,created_at,updated_at) "
    "VALUES(?,NULL,?,'liked',NULL,?,?)", (did, "湖畔咖啡", now, now),
).lastrowid
conn.commit(); conn.close()

client = module.app.test_client()
client.set_cookie(module.DEVICE_COOKIE, module._device_cookie_encode(did))

# 第一次读取只做幂等旧数据迁移；图谱不再从旧表拼第二套边。
profile = client.get("/api/v2/memories").get_json()
assert profile["stats"]["active"] == 4, profile["stats"]
conn = module._db_connect()
facts = [dict(row) for row in conn.execute(
    "SELECT * FROM memory_wiki_facts WHERE device_id=? ORDER BY id", (did,)
).fetchall()]
assert len(facts) == 4, facts
assert all(row["domain_kind"] for row in facts), facts
assert len([edge for edge in profile["graph"]["edges"] if edge.get("origin") == "wiki"]) == 4
assert not [edge for edge in profile["graph"]["edges"] if edge.get("origin") in ("person", "preference", "feedback")]
conn.close()

# 业务编辑在同一事务内更新事实与投影，并留下旧事实版本。
changed = client.patch("/api/v2/memories/item", json={
    "kind": "preference", "id": preference_id, "patch": {"value": "步行"},
    "expected_updated_at": now,
})
assert changed.status_code == 200, changed.get_json()
conn = module._db_connect()
fact = conn.execute(
    "SELECT * FROM memory_wiki_facts WHERE device_id=? AND predicate='preference:transport:default_mode'", (did,)
).fetchone()
assert fact and fact["value"] == "步行" and fact["domain_key"] == str(preference_id)
assert conn.execute(
    "SELECT value FROM memory_wiki_fact_versions WHERE device_id=? AND predicate='preference:transport:default_mode'", (did,)
).fetchone()[0] == "公交"
conn.close()

# 通用候选偏好确认后分配稳定槽位，并自动生成推荐代码可读的投影。
conn = module._db_connect()
candidate_id = conn.execute(
    "INSERT INTO memory_candidates(device_id,kind,entity_key,field_name,candidate_value,confidence,persistence_score,"
    "evidence_summary,status,created_at,updated_at,value_type,resolution_confidence,resolution_status) "
    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    (did, "user", "我", "preference", "不吃辣", .94, .9, "用户长期不吃辣", "candidate", now, now,
     "food", .98, "resolved"),
).lastrowid
conn.commit(); conn.close()
confirmed = client.post("/api/v2/memories/candidate-group", json={
    "action": "confirm", "candidate_ids": [candidate_id],
})
assert confirmed.status_code == 200, confirmed.get_json()
conn = module._db_connect()
assert conn.execute(
    "SELECT 1 FROM memory_wiki_facts WHERE device_id=? AND predicate='preference:food:general' AND value='不吃辣'",
    (did,),
).fetchone()
assert conn.execute(
    "SELECT 1 FROM agent_memories WHERE device_id=? AND category='food' AND memory_key='general' AND memory_value='不吃辣'",
    (did,),
).fetchone()

# 删除一条人物关系只清对应事实与投影字段，不误删人物的地点关系。
relation = conn.execute(
    "SELECT id FROM memory_wiki_facts WHERE device_id=? AND subject_key='阿杰' AND predicate='relation'", (did,)
).fetchone()
conn.close()
removed = client.delete("/api/v2/memories/relation", json={
    "origin": "wiki", "id": relation["id"], "predicate": "relation",
})
assert removed.status_code == 200, removed.get_json()
conn = module._db_connect()
person = conn.execute("SELECT relation,usual_place FROM memory_people WHERE id=?", (person_id,)).fetchone()
assert person["relation"] is None and person["usual_place"] == "浙江大学紫金港校区"
assert not conn.execute("SELECT 1 FROM memory_wiki_facts WHERE id=?", (relation["id"],)).fetchone()
assert conn.execute(
    "SELECT 1 FROM memory_wiki_facts WHERE device_id=? AND subject_key='阿杰' AND predicate='usual_place'", (did,)
).fetchone()
conn.close()

print("MEMORY_UNIFIED_SMOKE_OK", preference_id, person_id, feedback_id)
