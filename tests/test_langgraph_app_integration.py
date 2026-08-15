from __future__ import annotations

import importlib
import sys


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
