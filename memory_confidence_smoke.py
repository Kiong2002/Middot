"""双评分、自动晋升、敏感拦截、冲突降级与人工换代契约测试。"""

import importlib.util
import os
import tempfile


root = tempfile.mkdtemp(prefix="middot-memory-confidence-")
os.environ["MIDDOT_DB_PATH"] = os.path.join(root, "middot.db")
os.environ["MIDDOT_DEVICE_SECRET"] = "memory-confidence-smoke-secret-20260813"
app_path = os.environ.get("MIDDOT_APP_TEST_PATH", os.path.join(os.path.dirname(__file__), "app_v2.py"))
spec = importlib.util.spec_from_file_location("middot_memory_confidence_app", app_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

did = "9" * 32
now = module._now()
conn = module._db_connect()
conn.execute("INSERT INTO devices(device_id,created_at,last_seen_at) VALUES(?,?,?)", (did, now, now))


def add(item, conversation, start):
    module._memory_candidate_add_evidence(conn, did, item, conversation, start, start + 1, now + start)
    return module._memory_reconcile_candidates(conn, did)


def candidate(field, value, confidence=.86, persistence=.9):
    return {
        "kind": "person", "entity_key": "阿杰", "field_name": field,
        "candidate_value": value, "confidence": confidence,
        "persistence_score": persistence, "evidence_summary": f"阿杰的{field}是{value}",
        "status": "candidate",
    }


# 一条证据不自动晋升；第二个独立对话使低风险稳定事实超过阈值。
first = add(candidate("hometown", "浙江"), "conversation-a", 1)
assert first["promoted"] == 0
second = add(candidate("hometown", "浙江"), "conversation-b", 3)
assert second["promoted"] == 1, second
fact = conn.execute(
    "SELECT * FROM memory_wiki_facts WHERE device_id=? AND subject_key='阿杰' AND predicate='hometown'", (did,)
).fetchone()
assert fact and fact["status"] == "confirmed" and fact["promotion_reason"] == "auto_high_confidence"
assert .88 <= float(fact["confidence"]) < 1

# 第三方常用位置属于敏感事实，即使证据强也只进入待确认。
add(candidate("usual_place", "浙江大学紫金港校区", .95, .95), "conversation-c", 5)
blocked = add(candidate("usual_place", "浙江大学紫金港校区", .95, .95), "conversation-d", 7)
assert blocked["promoted"] == 0
place = conn.execute(
    "SELECT * FROM memory_candidates WHERE device_id=? AND field_name='usual_place'", (did,)
).fetchone()
assert place["status"] == "candidate" and "敏感" in (place["decision_reason"] or "")
assert not conn.execute("SELECT 1 FROM memory_people WHERE device_id=?", (did,)).fetchone()

# 强冲突不会静默覆盖：旧事实降为 challenged，不再属于当前生效事实。
add(candidate("hometown", "江苏", .9, .9), "conversation-e", 9)
conflict_result = add(candidate("hometown", "江苏", .9, .9), "conversation-f", 11)
assert conflict_result["challenged"] == 1, conflict_result
fact = conn.execute("SELECT * FROM memory_wiki_facts WHERE id=?", (fact["id"],)).fetchone()
assert fact["value"] == "浙江" and fact["status"] == "challenged"
challenger_rows = [dict(row) for row in conn.execute(
    "SELECT * FROM memory_candidates WHERE device_id=? AND candidate_value='江苏' AND status='conflict'", (did,)
).fetchall()]
assert challenger_rows

# 人工确认代表100%采用授权，完成版本换代并保留旧值历史。
group = module._candidate_groups_from_rows(challenger_rows)[0]
module._confirm_candidate_group(conn, did, group, challenger_rows, "江苏")
conn.commit()
fact = conn.execute("SELECT * FROM memory_wiki_facts WHERE id=?", (fact["id"],)).fetchone()
assert fact["value"] == "江苏" and fact["status"] == "confirmed"
assert float(fact["confidence"]) == 1 and float(fact["authority"]) == 1
version = conn.execute("SELECT * FROM memory_wiki_fact_versions WHERE device_id=?", (did,)).fetchone()
assert version and version["value"] == "浙江" and version["change_reason"] == "manual_confirmation"
conn.close()
print("MEMORY_CONFIDENCE_SMOKE_OK")
