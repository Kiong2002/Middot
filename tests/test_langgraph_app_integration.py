from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace


def _load_app(monkeypatch, tmp_path):
    monkeypatch.setenv("MIDDOT_DB_PATH", str(tmp_path / "middot-test.db"))
    monkeypatch.setenv("MIDDOT_LANGGRAPH_DB_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.setenv("MIDDOT_AGENT_RUNTIME", "langgraph")
    monkeypatch.setenv("MIDDOT_DEVICE_SECRET", "a" * 64)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("MIDDOT_LANGSMITH_TRACING", "false")
    import amap_client

    monkeypatch.setattr(amap_client, "DEEPSEEK_API_KEY", "test-key")
    sys.modules.pop("app_v2", None)
    module = importlib.import_module("app_v2")
    module.MIDDOT_AGENT_RUNTIME = "langgraph"
    module.MIDDOT_LANGGRAPH_DB_PATH = str(tmp_path / "checkpoints.db")
    return module


def _session(module, device_id="device-a"):
    return module.session_create(
        {
            "participants": [
                {"id": "me", "name": "我", "lng": None, "lat": None, "address": ""}
            ],
            "city": "北京",
            "memory_did": device_id,
            "my_did": device_id,
            "current_user_message": "我从清华出发",
            "current_utterance_parse": {
                "city_context": "北京",
                "locations": [
                    {"participant_index": 1, "expression": "清华", "owner": "我"}
                ],
            },
            "agent_task": {
                "id": "task-one",
                "status": "running",
                "completed": [],
                "failures": [],
            },
        }
    )


def _ambiguous_resolution():
    return {
        "success": True,
        "status": "ambiguous",
        "query": "清华",
        "provider": "amap_inputtips",
        "confidence": 0.62,
        "candidates": [
            {
                "id": "tsinghua-main",
                "label": "清华大学",
                "address": "北京市海淀区双清路30号",
                "lng": 116.326,
                "lat": 40.003,
                "source": "amap_inputtips",
            },
            {
                "id": "tsinghua-garden",
                "label": "清华园",
                "address": "北京市海淀区成府路",
                "lng": 116.333,
                "lat": 39.999,
                "source": "amap_inputtips",
            },
        ],
    }


def test_real_tool_waits_and_resumes_after_runtime_restart(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    resolver_calls = []

    def fake_resolver(*args, **kwargs):
        resolver_calls.append((args, kwargs))
        return _ambiguous_resolution()

    monkeypatch.setattr(module, "_resolve_place_candidates", fake_resolver)
    first_sid = _session(module)
    tool_result, patch = module._tool_set_participant_location(
        first_sid, {"index": 1, "place_name": "清华"}
    )
    assert tool_result["ok"] is True
    assert tool_result["runtime"] == "langgraph"
    assert patch["type"] == "location_choices"
    assert patch["token"]
    assert len(resolver_calls) == 1

    # 模拟部署/进程重建：内存 session 和 graph 对象丢失，SQLite checkpoint 保留。
    module._sessions.clear()
    module._location_graph_conn.close()
    module._location_graph_runtime = None
    module._location_graph_conn = None
    resumed_sid = _session(module)

    ok, message, canonical = module._apply_location_choice(
        resumed_sid,
        {"token": patch["token"], "candidate_id": "tsinghua-main"},
    )
    assert ok is True
    assert canonical == "清华大学"
    assert "清华大学" in message
    participant = module.session_get(resumed_sid)["participants"][0]
    assert participant["lng"] == 116.326
    assert len(resolver_calls) == 1

    conn = module._db_connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM agent_operations").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM place_alias_evidence").fetchone()[0] == 1
        status = conn.execute(
            "SELECT status FROM agent_interrupts WHERE interrupt_id=?", (patch["token"],)
        ).fetchone()[0]
        assert status == "consumed"
    finally:
        conn.close()

    repeated = module._apply_location_choice(
        resumed_sid,
        {"token": patch["token"], "candidate_id": "tsinghua-main"},
    )
    assert repeated[0] is False


def test_confident_graph_result_stays_a_visible_draft(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    resolved = _ambiguous_resolution()
    resolved.update(
        status="resolved",
        confidence=0.97,
        reason="简称与大学实体高度一致",
        candidate=resolved["candidates"][0],
    )
    resolved.pop("candidates")
    monkeypatch.setattr(module, "_resolve_place_candidates", lambda *args, **kwargs: resolved)
    sid = _session(module)
    tool_result, patch = module._tool_set_participant_location(
        sid, {"index": 1, "place_name": "清华"}
    )
    assert tool_result["ok"] is True
    assert patch["type"] == "draft"
    assert patch["data"]["place_resolution"]["label"] == "清华大学"
    # AI 可以替用户选候选，但仍遵守现有草稿确认边界，不直接修改设置。
    assert module.session_get(sid)["participants"][0]["lng"] is None


def test_legacy_flag_keeps_original_location_path(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    module.MIDDOT_AGENT_RUNTIME = "legacy"
    monkeypatch.setattr(
        module, "_resolve_place_candidates", lambda *args, **kwargs: _ambiguous_resolution()
    )
    sid = _session(module)
    result, patch = module._tool_set_participant_location(
        sid, {"index": 1, "place_name": "清华"}
    )
    assert result["ok"] is True
    assert "runtime" not in result
    assert patch["type"] == "location_choices"
    assert len(patch["token"]) == 12
    conn = module._db_connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM agent_interrupts").fetchone()[0] == 0
    finally:
        conn.close()


def test_assistant_stream_uses_main_langgraph_orchestrator(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    module.MIDDOT_AGENT_ORCHESTRATOR = "langgraph"
    monkeypatch.setattr(
        module,
        "_parse_meeting_utterance",
        lambda *args, **kwargs: {
            "city_context": "北京",
            "locations": [],
            "ignored_text": [],
        },
    )
    monkeypatch.setattr(module, "_memory_context", lambda *args, **kwargs: "[记忆] 无")

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            delta = SimpleNamespace(content="已经准备好了。", tool_calls=None)
            return [SimpleNamespace(choices=[SimpleNamespace(delta=delta)])]

    module.llm_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    client = module.app.test_client()
    response = client.post(
        "/api/v2/assistant/stream",
        json={
            "message": "帮我找咖啡馆",
            "bootstrap": {
                "city": "北京",
                "participants": [
                    {
                        "id": "me",
                        "name": "我",
                        "lng": 116.3,
                        "lat": 39.9,
                        "address": "北京市",
                    }
                ],
                "pois": [],
                "query": "咖啡馆",
            },
        },
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '"type": "session"' in body
    assert '"type": "token"' in body
    assert "已经准备好了。" in body
    assert '"type": "done"' in body

    conn = module._db_connect()
    try:
        row = conn.execute(
            "SELECT payload_json FROM agent_trace_steps WHERE step_type='llm_call' "
            "ORDER BY created_at_ms DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert '"runtime": "langgraph"' in row["payload_json"]
        assert '"node": "planner"' in row["payload_json"]
    finally:
        conn.close()


def test_normal_choice_survives_session_and_runtime_restart(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    sid = module.session_create(
        {
            "participants": [],
            "memory_did": "device-a",
            "my_did": "device-a",
            "agent_task": {
                "id": "task-choice",
                "status": "running",
                "completed": [],
                "failures": [],
            },
        }
    )
    result, patch = module._tool_offer_choices(
        sid,
        {
            "question": "你想怎么过去？",
            "mode": "single",
            "options": [{"label": "坐地铁"}, {"label": "打车"}],
        },
    )
    assert result["ok"] is True
    token = patch["token"]

    module._sessions.clear()
    wrong_sid = module.session_create(
        {
            "participants": [],
            "memory_did": "device-b",
            "my_did": "device-b",
        }
    )
    assert module._consume_offer_choice_answers(
        wrong_sid, [{"token": token, "label": "坐地铁"}]
    ) == ("", [])

    resumed_sid = module.session_create(
        {
            "participants": [],
            "memory_did": "device-a",
            "my_did": "device-a",
        }
    )
    question, labels = module._consume_offer_choice_answers(
        resumed_sid,
        [{"token": token, "label": "坐地铁"}],
    )

    assert question == "你想怎么过去？"
    assert labels == ["坐地铁"]
    assert module.session_get(resumed_sid)["agent_task"]["id"] == "task-choice"
    conn = module._db_connect()
    try:
        status = conn.execute(
            "SELECT status FROM agent_choice_interrupts WHERE interrupt_id=?",
            (token,),
        ).fetchone()[0]
        assert status == "consumed"
    finally:
        conn.close()

    assert module._consume_offer_choice_answers(
        resumed_sid, [{"token": token, "label": "坐地铁"}]
    ) == ("", [])
