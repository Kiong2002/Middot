"""Regression: room AI writes use field-level CAS instead of a global cooldown."""

import importlib.util
import json
import os
import tempfile
from types import SimpleNamespace


root = tempfile.mkdtemp(prefix="middot-room-write-guard-")
os.environ["MIDDOT_DB_PATH"] = os.path.join(root, "middot.db")
os.environ["MIDDOT_DEVICE_SECRET"] = "room-write-guard-smoke-secret-20260814"
app_path = os.environ.get(
    "MIDDOT_APP_TEST_PATH", os.path.join(os.path.dirname(__file__), "app_v2.py")
)
spec = importlib.util.spec_from_file_location("middot_room_write_guard", app_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


device_id = "a" * 32
client = module.app.test_client()
client.set_cookie(module.DEVICE_COOKIE, module._device_cookie_encode(device_id))

created = client.post("/api/v2/rooms", json={
    "nickname": "我",
    "keyword": "咖啡",
    "location": {"lng": 120.1, "lat": 30.2, "address": "旧位置"},
})
assert created.status_code == 200, created.get_data(as_text=True)
code = created.get_json()["snapshot"]["code"]


def update(payload, expected_status=200):
    response = client.post(f"/api/v2/rooms/{code}/update", json=payload)
    assert response.status_code == expected_status, response.get_data(as_text=True)
    return response.get_json()


# The first AI write sees the expected value and succeeds atomically with attribution.
first = update({
    "keyword": "火锅",
    "expected": {"keyword": "咖啡"},
    "ai_action": {
        "tool": "set_keyword",
        "before": {"keyword": "咖啡"},
        "after": {"keyword": "火锅"},
    },
})
first_revision = first["revision"]

# A stale AI write to the same field must not overwrite the newer value.
conflict = update({
    "keyword": "安静咖啡",
    "expected": {"keyword": "咖啡"},
    "ai_action": {
        "tool": "set_keyword",
        "before": {"keyword": "咖啡"},
        "after": {"keyword": "安静咖啡"},
    },
}, 409)
assert conflict["conflict"] is True
assert conflict["conflict_fields"] == ["keyword"]
assert conflict["snapshot"]["keyword"] == "火锅"
assert len(conflict["snapshot"]["last_ai_actions"]) == 1

# A revision change on another field is harmless: keyword CAS still succeeds.
anchor = {"lng": 120.2, "lat": 30.3, "name": "新锚点", "radius_m": 5000}
unrelated = update({"anchor": anchor})
assert unrelated["revision"] > first_revision
parallel = update({
    "keyword": "烧烤",
    "expected": {"keyword": "火锅"},
    "ai_action": {
        "tool": "set_keyword",
        "before": {"keyword": "火锅"},
        "after": {"keyword": "烧烤"},
    },
})
assert parallel["revision"] > unrelated["revision"]

# The same protection applies to the member's own location.
old_location = {"lng": 120.1, "lat": 30.2, "address": "旧位置"}
new_location = {"lng": 120.3, "lat": 30.4, "address": "新位置"}
update({
    "my_location": new_location,
    "expected": {"my_location": old_location},
    "ai_action": {
        "tool": "set_participant_location",
        "before": {"location": old_location},
        "after": {"location": new_location},
    },
})
location_conflict = update({
    "my_location": {"lng": 120.5, "lat": 30.6, "address": "过期建议"},
    "expected": {"my_location": old_location},
}, 409)
me = next(member for member in location_conflict["snapshot"]["members"] if member["is_me"])
assert me["location"] == new_location

# The old blanket cooldown no longer exists; solo assistant writes are not delayed.
assert not hasattr(module, "_AI_WRITE_LAST")
assert not hasattr(module, "_ai_write_gate")

# Two immediate solo assistant requests both execute. There is no time-based lock.
stream_round = 0


def completion_create(**kwargs):
    global stream_round
    if not kwargs.get("stream"):
        parsed = {
            "intent": "other", "activity": "", "city_context": "",
            "locations": [], "ignored_text": [],
        }
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(parsed, ensure_ascii=False)
        ))])
    stream_round += 1
    if stream_round % 2:
        call = SimpleNamespace(
            index=0,
            id=f"call_keyword_{stream_round}",
            function=SimpleNamespace(
                name="set_keyword",
                arguments=json.dumps({"keyword": "安静 适合聊天"}, ensure_ascii=False),
            ),
        )
        delta = SimpleNamespace(content=None, tool_calls=[call])
    else:
        delta = SimpleNamespace(content="已经改好了。", tool_calls=None)
    return iter([SimpleNamespace(choices=[SimpleNamespace(delta=delta)])])


module.DEEPSEEK_API_KEY = "test"
module.llm_client = SimpleNamespace(
    chat=SimpleNamespace(completions=SimpleNamespace(create=completion_create))
)
assistant_payload = {
    "message": "关键词换成安静适合聊天",
    "bootstrap": {
        "participants": [{"id": "me", "name": "我", "lng": 120.1, "lat": 30.2}],
        "query": "咖啡",
    },
}
for _ in range(2):
    response = client.post("/api/v2/assistant/stream", json=assistant_payload)
    body = response.get_data(as_text=True)
    assert response.status_code == 200, body
    assert '"type": "tool_call"' in body, body
    assert "AI 写限流" not in body, body
    assert "达到工具调用上限" not in body, body

assert stream_round == 4, stream_round

print("ROOM_WRITE_GUARD_SMOKE_OK")
