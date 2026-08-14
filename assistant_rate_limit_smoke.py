"""Regression: an AI write cooldown must end once, not retry to MAX_ITERS."""

import importlib.util
import json
import os
import tempfile
import time
from types import SimpleNamespace


root = tempfile.mkdtemp(prefix="middot-assistant-rate-limit-")
os.environ["MIDDOT_DB_PATH"] = os.path.join(root, "middot.db")
os.environ["MIDDOT_DEVICE_SECRET"] = "assistant-rate-limit-smoke-secret-20260814"
app_path = os.environ.get(
    "MIDDOT_APP_TEST_PATH", os.path.join(os.path.dirname(__file__), "app_v2.py")
)
spec = importlib.util.spec_from_file_location("middot_assistant_rate_limit", app_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


calls = []


def completion_create(**kwargs):
    calls.append(kwargs)
    if not kwargs.get("stream"):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({
                "intent": "other", "activity": "", "city_context": "",
                "locations": [], "ignored_text": [],
            }, ensure_ascii=False)
        ))])
    tool_calls = [
        SimpleNamespace(
            index=0, id="call_keyword",
            function=SimpleNamespace(
                name="set_keyword",
                arguments=json.dumps({"keyword": "安静 适合聊天"}, ensure_ascii=False),
            ),
        ),
        SimpleNamespace(
            index=1, id="call_location",
            function=SimpleNamespace(
                name="set_participant_location",
                arguments=json.dumps({
                    "index": 2, "place_name": "浙江大学紫金港校区",
                }, ensure_ascii=False),
            ),
        ),
    ]
    return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=None, tool_calls=tool_calls,
    ))])])


module.DEEPSEEK_API_KEY = "test"
module.llm_client = SimpleNamespace(
    chat=SimpleNamespace(completions=SimpleNamespace(create=completion_create))
)
device_id = "a" * 32
module._AI_WRITE_LAST[device_id] = time.monotonic()
client = module.app.test_client()
client.set_cookie(module.DEVICE_COOKIE, module._device_cookie_encode(device_id))
response = client.post("/api/v2/assistant/stream", json={
    "message": "把第二位改到浙大紫金港，关键词换成安静适合聊天",
    "bootstrap": {
        "participants": [
            {"id": "me", "name": "我", "lng": 120.1, "lat": 30.2},
            {"id": "friend", "name": "朋友", "lng": 120.2, "lat": 30.3},
        ],
        "query": "咖啡",
    },
})
body = response.get_data(as_text=True)
assert response.status_code == 200, body
assert "达到工具调用上限" not in body, body
assert '"type": "tool_call"' not in body, body
assert body.count("刚刚才更新过方案") == 1, body
assert len(calls) == 2, len(calls)  # one utterance parse + one assistant call
print("ASSISTANT_RATE_LIMIT_SMOKE_OK")
