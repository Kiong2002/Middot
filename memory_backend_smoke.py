"""Smoke-test the meeting profile migration and API against a disposable DB copy."""

import importlib.util
import json
import os
import time
import uuid
from http.cookies import SimpleCookie


app_path = os.environ.get("MIDDOT_APP_TEST_PATH", "/tmp/middot-app-v2-profile-test.py")
spec = importlib.util.spec_from_file_location("middot_profile_test", app_path)
middot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(middot)


def response_cookie(response, name):
    """Return one response cookie value plus its complete Set-Cookie header."""
    headers = response.headers.getlist("Set-Cookie")
    jar = SimpleCookie()
    for header in headers:
        jar.load(header)
    assert name in jar, (name, headers)
    return jar[name].value, "\n".join(headers)


def assert_keys_absent(value, forbidden, path="result"):
    """Recursively guard model-facing tool payloads against UI-only provenance."""
    if isinstance(value, dict):
        overlap = forbidden.intersection(value)
        assert not overlap, f"forbidden keys {sorted(overlap)} at {path}"
        for key, child in value.items():
            assert_keys_absent(child, forbidden, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_keys_absent(child, forbidden, f"{path}[{index}]")

conn = middot._db_connect()
print("version", conn.execute("PRAGMA user_version").fetchone()[0])
print(
    "sources/events",
    conn.execute("SELECT COUNT(*) FROM memory_sources").fetchone()[0],
    conn.execute("SELECT COUNT(*) FROM memory_fact_events").fetchone()[0],
)
counts = {
    table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    for table in ("agent_memories", "memory_people", "memory_episodes", "memory_feedback")
}
print("projection_counts", counts)
row = conn.execute(
    "SELECT device_id,id,memory_value,updated_at FROM agent_memories ORDER BY id LIMIT 1"
).fetchone()
assert row, "need existing preference for migration test"
device_id = row["device_id"]
record_id = row["id"]
old_value = row["memory_value"]
old_updated = row["updated_at"]
conn.close()

client = middot.app.test_client()
client.set_cookie(middot.DEVICE_COOKIE, middot._device_cookie_encode(device_id))
response = client.get("/api/v2/memories")
assert response.status_code == 200, response.data
profile = response.get_json()
print("get_profile", profile["stats"], profile["preferences"][0]["provenance"]["type"])

new_value = "公交" if old_value != "公交" else "步行"
response = client.patch(
    "/api/v2/memories/item",
    json={
        "kind": "preference",
        "id": record_id,
        "patch": {"value": new_value},
        "expected_updated_at": old_updated,
    },
)
assert response.status_code == 200, response.data
response = client.get(
    f"/api/v2/memories/item/sources?kind=preference&id={record_id}"
)
sources = response.get_json()["sources"]
print("source_chain", [(item["type"], item["action"]) for item in sources])
assert sources[0]["type"] == "profile_edit"
assert any(item["type"] == "legacy_import" for item in sources)

stale_update = client.patch(
    "/api/v2/memories/item",
    json={
        "kind": "preference",
        "id": record_id,
        "patch": {"value": old_value},
        "expected_updated_at": old_updated,
    },
)
assert stale_update.status_code == 409, stale_update.data
print("optimistic_lock", stale_update.get_json()["error"])

# The model may not persist a one-off choice without explicit user authorization.
sid = middot.session_create(
    {
        "memory_did": device_id,
        "current_user_message": "今天坐公交",
        "current_memory_source_ref": "test:no-explicit",
    }
)
blocked, _ = middot._tool_remember_preference(
    sid, {"category": "transport", "key": "default_mode", "value": "公交"}
)
assert not blocked["ok"], blocked
sid = middot.session_create(
    {
        "memory_did": device_id,
        "current_user_message": "请记住我以后默认坐公交",
        "current_memory_source_ref": "test:explicit",
    }
)
written, _ = middot._tool_remember_preference(
    sid, {"category": "transport", "key": "default_mode", "value": "公交"}
)
assert written["ok"], written
print("write_gate", blocked["error"], written["summary"])

negative_cases = [
    ("不要记住我坐公交", {"category":"transport","key":"default_mode","value":"公交"}),
    ("不用保存公交偏好", {"category":"transport","key":"default_mode","value":"公交"}),
]
for message, payload in negative_cases:
    sid = middot.session_create({
        "memory_did": device_id,
        "current_user_message": message,
        "current_memory_source_ref": f"test:negative:{message}",
    })
    result, _ = middot._tool_remember_preference(sid, payload)
    assert not result["ok"], (message, result)

sid = middot.session_create({
    "memory_did": device_id,
    "current_user_message": "请记住我以后默认坐公交",
    "current_memory_source_ref": "test:ungrounded",
})
ungrounded, _ = middot._tool_remember_person(
    sid, {"name":"Bob","relation":"朋友","usual_place":"望京","city":"北京"}
)
assert not ungrounded["ok"], ungrounded

feedback_negative_cases = [
    ("我不喜欢测试火锅店", {"poi_name":"测试火锅店","signal":"liked"}),
    ("我没去过测试火锅店", {"poi_name":"测试火锅店","signal":"visited"}),
    ("我不是不喜欢测试火锅店", {"poi_name":"测试火锅店","signal":"disliked"}),
]
for message, payload in feedback_negative_cases:
    sid = middot.session_create({
        "memory_did": device_id,
        "current_user_message": message,
        "current_memory_source_ref": f"test:negative-feedback:{message}",
    })
    result, _ = middot._tool_remember_feedback(sid, payload)
    assert not result["ok"], (message, result)
print("negative_and_grounding_gates", "OK")

# A person-memory authorization survives clarification turns, but nothing is
# persisted until the user consumes the server-issued visible confirmation card.
draft_name = f"阿杰草稿{uuid.uuid4().hex[:6]}"
draft_sid = middot.session_create({
    "memory_did": device_id,
    "current_user_message": f"请记住：{draft_name}在浙大",
    "current_memory_source_ref": f"test:person-draft:{uuid.uuid4().hex}",
})
middot._memory_track_authorization(draft_sid, f"请记住：{draft_name}在浙大")
needs_detail, detail_patch = middot._tool_remember_person(
    draft_sid,
    # Even if the model guesses a specific campus from old context, the source
    # only said “浙大”, so the server must keep this as a clarification draft.
    {"name": draft_name, "usual_place": "浙江大学紫金港校区", "city": "杭州", "days": 90},
)
assert needs_detail["ok"] and needs_detail.get("waiting_for_detail") and detail_patch is None
conn = middot._db_connect()
assert not conn.execute(
    "SELECT 1 FROM memory_people WHERE device_id=? AND name=?", (device_id, draft_name)
).fetchone()
conn.close()
middot.session_update(draft_sid, {"current_user_message": "紫金港"})
middot._memory_track_authorization(draft_sid, "紫金港")
staged, confirm_patch = middot._tool_remember_person(
    draft_sid,
    {"name": draft_name, "usual_place": "浙江大学紫金港校区", "city": "杭州", "days": 90},
)
assert staged["ok"] and staged.get("waiting_for_user"), staged
assert confirm_patch["purpose"] == "memory_confirmation"
assert "浙江大学紫金港校区" in confirm_patch["question"]
conn = middot._db_connect()
assert not conn.execute(
    "SELECT 1 FROM memory_people WHERE device_id=? AND name=?", (device_id, draft_name)
).fetchone()
conn.close()
question, labels = middot._consume_offer_choice_answers(draft_sid, [{
    "token": confirm_patch["token"], "label": "确认保存",
}])
assert labels == ["确认保存"] and "准备保存" in question
confirmed_summary = middot._apply_memory_confirmation_choice(draft_sid, device_id, labels)
assert "已记住" in confirmed_summary
conn = middot._db_connect()
saved_draft = conn.execute(
    "SELECT usual_place,city FROM memory_people WHERE device_id=? AND name=?",
    (device_id, draft_name),
).fetchone()
assert saved_draft and saved_draft["usual_place"] == "浙江大学紫金港校区"
conn.close()
print("person_memory_draft_clarify_confirm", "OK")

# Restarting schema initialization must not mislabel post-migration facts as legacy imports.
sid = middot.session_create(
    {
        "memory_did": device_id,
        "current_user_message": "请记住我长期不吃香菜",
        "current_memory_source_ref": "test:new-after-v2",
    }
)
written_food, _ = middot._tool_remember_preference(
    sid, {"category": "food", "key": "avoid_cilantro", "value": "不吃香菜"}
)
assert written_food["ok"], written_food
conn = middot._db_connect()
new_record = conn.execute(
    "SELECT id FROM agent_memories WHERE device_id=? AND category='food' AND memory_key='avoid_cilantro'",
    (device_id,),
).fetchone()
before_counts = (
    conn.execute("SELECT COUNT(*) FROM memory_sources").fetchone()[0],
    conn.execute("SELECT COUNT(*) FROM memory_fact_events").fetchone()[0],
)
conn.close()
middot.init_middot_db()
conn = middot._db_connect()
after_counts = (
    conn.execute("SELECT COUNT(*) FROM memory_sources").fetchone()[0],
    conn.execute("SELECT COUNT(*) FROM memory_fact_events").fetchone()[0],
)
assert after_counts == before_counts
assert not conn.execute(
    "SELECT 1 FROM memory_fact_events WHERE idempotency_key=?",
    (f"legacy:preference:{new_record['id']}",),
).fetchone()
conn.close()
print("restart_migration_idempotent", before_counts)

# Updating only a relationship must not turn the old place into a permanent fact.
conn = middot._db_connect()
now = int(time.time())
expires_at = now + 40 * 86400
cursor = conn.execute(
    "INSERT INTO memory_people(device_id,name,relation,usual_place,city,expires_at,created_at,updated_at) "
    "VALUES(?,?,?,?,?,?,?,?)",
    (device_id, "TTL更新测试", "同事", "仍需过期的地点", "杭州", expires_at, now, now),
)
person_id = cursor.lastrowid
conn.commit()
conn.close()
response = client.patch(
    "/api/v2/memories/item",
    json={
        "kind": "person",
        "id": person_id,
        "patch": {"relation": "朋友"},
        "expected_updated_at": now,
    },
)
assert response.status_code == 200, response.data
conn = middot._db_connect()
person_row = conn.execute("SELECT * FROM memory_people WHERE id=?", (person_id,)).fetchone()
assert person_row["usual_place"] == "仍需过期的地点"
assert person_row["expires_at"] == expires_at
conn.close()
print("relation_only_keeps_place_ttl", person_row["expires_at"])

# Sentiment is mutually exclusive while visited remains orthogonal.
sid = middot.session_create(
    {
        "memory_did": device_id,
        "current_user_message": "我喜欢测试火锅店",
        "current_memory_source_ref": "test:liked",
    }
)
liked, _ = middot._tool_remember_feedback(
    sid, {"poi_name": "测试火锅店", "signal": "liked", "reason": "安静"}
)
assert liked["ok"], liked
sid = middot.session_create(
    {
        "memory_did": device_id,
        "current_user_message": "后来发现我不喜欢测试火锅店",
        "current_memory_source_ref": "test:disliked",
    }
)
disliked, _ = middot._tool_remember_feedback(
    sid, {"poi_name": "测试火锅店", "signal": "disliked", "reason": "太吵"}
)
assert disliked["ok"], disliked
conn = middot._db_connect()
sentiments = conn.execute(
    "SELECT signal FROM memory_feedback WHERE device_id=? AND poi_name=? AND signal IN ('liked','disliked')",
    (device_id, "测试火锅店"),
).fetchall()
assert [row["signal"] for row in sentiments] == ["disliked"]
conn.close()
print("sentiment_conflict_resolved", sentiments[0]["signal"])

# An expired location stops affecting planning without hiding the relationship.
conn = middot._db_connect()
conn.execute(
    "INSERT OR REPLACE INTO memory_people(device_id,name,relation,usual_place,city,expires_at,created_at,updated_at) "
    "VALUES(?,?,?,?,?,?,?,?)",
    (
        device_id,
        "过期测试人物",
        "朋友",
        "绝不能进入模型的旧地点",
        "杭州",
        now - 10,
        now - 100,
        now - 100,
    ),
)
conn.commit()
conn.close()
person = [item for item in middot._people_rows(device_id) if item["name"] == "过期测试人物"][0]
context = middot._memory_context(device_id, "过期测试人物")
assert person["relation"] == "朋友" and person["place_status"] == "expired"
assert "绝不能进入模型的旧地点" not in context
print("expired_place", person["place_status"], "relationship_visible", person["relation"])

# Forgetting purges the compiled value and all provenance for that fact.
response = client.delete(
    "/api/v2/memories/item", json={"kind": "preference", "id": record_id}
)
assert response.status_code == 200, response.data
conn = middot._db_connect()
assert not conn.execute("SELECT 1 FROM agent_memories WHERE id=?", (record_id,)).fetchone()
assert not conn.execute(
    "SELECT 1 FROM memory_fact_events WHERE kind='preference' AND record_id=?",
    (record_id,),
).fetchone()
conn.close()
print("forget_purged", response.get_json())

# Room snapshots expose only room-scoped member ids; signed cookies reject tampering.
conn = middot._db_connect()
room_code = "907531"
conn.execute(
    "INSERT OR REPLACE INTO rooms(code,host_device_id,keyword,anchor_json,revision,status,created_at,last_active_at,updated_by,last_ai_actions_json) "
    "VALUES(?,?,?,?,?,'active',?,?,?,?)",
    (room_code, device_id, "咖啡", None, 1, now, now, device_id,
     '[{"id":"a1","ts":1,"actor_did":"%s","actor_name":"我","tool":"set_keyword","undone":false}]' % device_id),
)
conn.execute(
    "INSERT OR REPLACE INTO room_members(room_code,device_id,nickname,role,location_json,prefer,joined_at) "
    "VALUES(?,?,?,?,?,?,?)",
    (room_code, device_id, "我", "host", None, "auto", now),
)
conn.commit()
snapshot = middot._room_snapshot(conn, room_code, device_id)
serialized = str(snapshot)
assert device_id not in serialized
assert snapshot["members"][0]["member_id"]
assert snapshot["last_ai_actions"][0]["actor_member_id"]
conn.close()
assert middot._device_cookie_decode(middot._device_cookie_encode(device_id)) == device_id
assert middot._device_cookie_decode(f"{device_id}.tampered") is None
print("room_identity_redacted_and_cookie_signed", "OK")

# A legacy unsigned 32-hex cookie is exchanged once for a fresh signed identity.
# During the bounded claim window, its durable data and room ownership follow the
# new identity; once the claim expires, replaying the old bearer must not recover it.
legacy_did = uuid.uuid4().hex
legacy_source_ref = f"test:legacy-cookie:{uuid.uuid4().hex}"
legacy_pref_key = f"legacy_cookie_{uuid.uuid4().hex[:10]}"
legacy_now = int(time.time())
conn = middot._db_connect()
conn.execute(
    "INSERT INTO devices(device_id,created_at,last_seen_at) VALUES(?,?,?)",
    (legacy_did, legacy_now, legacy_now),
)
legacy_pref = conn.execute(
    "INSERT INTO agent_memories(device_id,category,memory_key,memory_value,source,status,created_at,updated_at) "
    "VALUES(?,?,?,?, 'explicit','confirmed',?,?)",
    (legacy_did, "food", legacy_pref_key, "旧身份连续迁移测试", legacy_now, legacy_now),
)
legacy_pref_id = int(legacy_pref.lastrowid)
legacy_source = conn.execute(
    "INSERT INTO memory_sources(device_id,source_type,source_ref,source_excerpt,metadata_json,created_at) "
    "VALUES(?, 'explicit_user', ?, ?, NULL, ?)",
    (legacy_did, legacy_source_ref, "旧身份来源连续迁移测试", legacy_now),
)
legacy_source_id = int(legacy_source.lastrowid)
conn.execute(
    "INSERT INTO memory_fact_events(device_id,kind,entity_key,record_id,action,value_json,"
    "changed_fields_json,source_id,happened_at,expires_at,idempotency_key) "
    "VALUES(?,?,?,?, 'assert', ?, ?, ?, ?, NULL, ?)",
    (
        legacy_did, "preference", f"id:{legacy_pref_id}", legacy_pref_id,
        json.dumps({"memory_value": "旧身份连续迁移测试"}, ensure_ascii=False),
        json.dumps(["memory_value"], ensure_ascii=False), legacy_source_id, legacy_now,
        f"legacy-cookie-event:{uuid.uuid4().hex}",
    ),
)
room_seed = int(legacy_did[:12], 16) % 1_000_000
legacy_room_code = None
for offset in range(1_000):
    candidate = f"{(room_seed + offset) % 1_000_000:06d}"
    if not conn.execute("SELECT 1 FROM rooms WHERE code=?", (candidate,)).fetchone():
        legacy_room_code = candidate
        break
assert legacy_room_code is not None
conn.execute(
    "INSERT INTO rooms(code,host_device_id,keyword,anchor_json,revision,status,created_at,last_active_at,updated_by,last_ai_actions_json) "
    "VALUES(?,?,?,?,?,'active',?,?,?,?)",
    (
        legacy_room_code, legacy_did, "迁移测试", None, 1, legacy_now, legacy_now,
        legacy_did,
        '[{"id":"legacy-move","ts":1,"actor_did":"%s","actor_name":"旧身份","tool":"set_keyword","undone":false}]'
        % legacy_did,
    ),
)
conn.execute(
    "INSERT INTO room_members(room_code,device_id,nickname,role,location_json,prefer,joined_at) "
    "VALUES(?,?,?,?,?,?,?)",
    (legacy_room_code, legacy_did, "旧身份", "host", None, "auto", legacy_now),
)
conn.commit()
conn.close()

legacy_client = middot.app.test_client()
legacy_client.set_cookie(middot.DEVICE_COOKIE, legacy_did)
legacy_response = legacy_client.get("/api/v2/memories")
assert legacy_response.status_code == 200, legacy_response.data
signed_cookie, signed_header = response_cookie(legacy_response, middot.DEVICE_COOKIE)
rotated_did = middot._device_cookie_decode(signed_cookie)
assert rotated_did and rotated_did not in (legacy_did,), signed_cookie
assert "." in signed_cookie and "HttpOnly" in signed_header, signed_header
legacy_profile = legacy_response.get_json()
assert any(item["memory_key"] == legacy_pref_key for item in legacy_profile["preferences"])

conn = middot._db_connect()
assert conn.execute(
    "SELECT device_id FROM agent_memories WHERE id=?", (legacy_pref_id,)
).fetchone()["device_id"] == rotated_did
assert conn.execute(
    "SELECT device_id FROM memory_sources WHERE id=?", (legacy_source_id,)
).fetchone()["device_id"] == rotated_did
assert conn.execute(
    "SELECT device_id FROM memory_fact_events WHERE kind='preference' AND record_id=?",
    (legacy_pref_id,),
).fetchone()["device_id"] == rotated_did
room_owner = conn.execute(
    "SELECT host_device_id FROM rooms WHERE code=?", (legacy_room_code,)
).fetchone()
assert room_owner["host_device_id"] == rotated_did
assert conn.execute(
    "SELECT role FROM room_members WHERE room_code=? AND device_id=?",
    (legacy_room_code, rotated_did),
).fetchone()["role"] == "host"
assert not conn.execute(
    "SELECT 1 FROM room_members WHERE room_code=? AND device_id=?",
    (legacy_room_code, legacy_did),
).fetchone()
claim = conn.execute(
    "SELECT new_device_id,expires_at FROM legacy_device_claims WHERE old_device_id=?",
    (legacy_did,),
).fetchone()
assert claim and claim["new_device_id"] == rotated_did and claim["expires_at"] > legacy_now
conn.close()

# Retrying the unsigned cookie inside the mapping window converges on the same DID.
claim_retry_client = middot.app.test_client()
claim_retry_client.set_cookie(middot.DEVICE_COOKIE, legacy_did)
claim_retry_response = claim_retry_client.get("/api/v2/memories")
assert claim_retry_response.status_code == 200, claim_retry_response.data
claim_retry_cookie, _ = response_cookie(claim_retry_response, middot.DEVICE_COOKIE)
assert middot._device_cookie_decode(claim_retry_cookie) == rotated_did
assert any(
    item["memory_key"] == legacy_pref_key
    for item in claim_retry_response.get_json()["preferences"]
)

# Expire the server-side bridge and prove that the old bearer no longer recovers
# the rotated identity or any of its migrated profile data.
conn = middot._db_connect()
conn.execute(
    "UPDATE legacy_device_claims SET expires_at=? WHERE old_device_id=?",
    (int(time.time()) - 1, legacy_did),
)
conn.commit()
conn.close()
expired_client = middot.app.test_client()
expired_client.set_cookie(middot.DEVICE_COOKIE, legacy_did)
expired_response = expired_client.get("/api/v2/memories")
assert expired_response.status_code == 200, expired_response.data
expired_cookie, _ = response_cookie(expired_response, middot.DEVICE_COOKIE)
expired_did = middot._device_cookie_decode(expired_cookie)
assert expired_did and expired_did not in (legacy_did, rotated_did)
assert not any(
    item["memory_key"] == legacy_pref_key
    for item in expired_response.get_json()["preferences"]
)
print("legacy_cookie_rotated_migrated_and_expired", legacy_did, "->", rotated_did)

# Event idempotency is value-sensitive: two successive values from the same
# source/ref/action are distinct revisions and must both survive.
event_record_id = int(new_record["id"])
event_source_ref = f"test:value-sensitive-events:{uuid.uuid4().hex}"
conn = middot._db_connect()
event_source_id = middot._memory_get_or_create_source(
    conn, device_id, "profile_edit", event_source_ref, "同源连续修改", {"test": True}
)
event_values = ("第一版值", "第二版值")
for event_value in event_values:
    middot._memory_append_event(
        conn,
        device_id=device_id,
        kind="preference",
        record_id=event_record_id,
        action="update",
        value={"id": event_record_id, "memory_value": event_value},
        changed_fields=["memory_value"],
        source_id=event_source_id,
        source_ref=event_source_ref,
    )
conn.commit()
event_rows = conn.execute(
    "SELECT value_json FROM memory_fact_events "
    "WHERE device_id=? AND kind='preference' AND record_id=? AND action='update' AND source_id=? "
    "ORDER BY id",
    (device_id, event_record_id, event_source_id),
).fetchall()
conn.close()
assert len(event_rows) == 2, [row["value_json"] for row in event_rows]
assert [json.loads(row["value_json"])["memory_value"] for row in event_rows] == list(event_values)
print("same_source_same_action_distinct_values_preserved", event_values)

# The model-facing list tool exposes compiled facts, never UI-only provenance or
# raw source excerpts.
list_sid = middot.session_create({"memory_did": device_id})
listed_memories, listed_patch = middot._tool_list_memories(list_sid, {})
assert listed_memories["ok"] and listed_patch is None
assert_keys_absent(listed_memories, {"provenance", "source_excerpt"})
print("list_memories_strips_provenance", "OK")

# Destructive whole-profile deletion is UI/API-only; a chat tool call may not
# silently turn an ambiguous request into kind=all.
forget_all_did = uuid.uuid4().hex
forget_all_key = f"must_survive_{uuid.uuid4().hex[:10]}"
conn = middot._db_connect()
conn.execute(
    "INSERT INTO devices(device_id,created_at,last_seen_at) VALUES(?,?,?)",
    (forget_all_did, legacy_now, legacy_now),
)
conn.execute(
    "INSERT INTO agent_memories(device_id,category,memory_key,memory_value,source,status,created_at,updated_at) "
    "VALUES(?,?,?,?, 'explicit','confirmed',?,?)",
    (forget_all_did, "food", forget_all_key, "必须保留", legacy_now, legacy_now),
)
conn.commit()
conn.close()
forget_all_sid = middot.session_create({
    "memory_did": forget_all_did,
    "current_user_message": "清空我的全部会面档案",
    "current_memory_source_ref": f"test:chat-forget-all:{uuid.uuid4().hex}",
})
forget_all_result, _ = middot._tool_forget_memory(forget_all_sid, {"kind": "all"})
assert not forget_all_result["ok"], forget_all_result
conn = middot._db_connect()
assert conn.execute(
    "SELECT 1 FROM agent_memories WHERE device_id=? AND memory_key=?",
    (forget_all_did, forget_all_key),
).fetchone()
conn.close()
print("chat_forget_all_rejected", forget_all_result["error"])

# A pure click on the visible “不用” answer is not affirmative authorization,
# even though the verified semantic message necessarily repeats the question's
# words “记住” and “公交”.
choice_did = uuid.uuid4().hex
choice_text = "要记住以后默认坐公交吗？：不用"
choice_payload = {"category": "transport", "key": "default_mode", "value": "公交"}
conn = middot._db_connect()
conn.execute(
    "INSERT INTO devices(device_id,created_at,last_seen_at) VALUES(?,?,?)",
    (choice_did, legacy_now, legacy_now),
)
conn.commit()
conn.close()
choice_sid = middot.session_create({
    "memory_did": choice_did,
    "current_user_message": choice_text,
    "current_memory_source_ref": f"test:pure-choice-no:{uuid.uuid4().hex}",
})
assert not middot._memory_explicit_intent(
    choice_sid, "preference", choice_payload, choice_text
)
choice_result, _ = middot._tool_remember_preference(choice_sid, choice_payload)
assert not choice_result["ok"], choice_result
conn = middot._db_connect()
assert not conn.execute(
    "SELECT 1 FROM agent_memories WHERE device_id=? AND category='transport' AND memory_key='default_mode'",
    (choice_did,),
).fetchone()
conn.close()
print("pure_choice_no_does_not_authorize", choice_result["error"])

print("ALL_MEMORY_BACKEND_TESTS_OK")
