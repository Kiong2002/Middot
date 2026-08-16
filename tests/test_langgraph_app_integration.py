from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

from flask import g


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
    tracked_authorizations = []
    monkeypatch.setattr(
        module,
        "_memory_track_authorization",
        lambda sid, text: tracked_authorizations.append((sid, text)),
    )

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
    assert tracked_authorizations[0][1] == "帮我找咖啡馆"

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


def test_history_continue_restores_state_summary_operations_and_pending_choice(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    module.MIDDOT_AGENT_ORCHESTRATOR = "langgraph"
    conversation_id = module._conversation_create("device-a", "继续规划")
    sid = module.session_create(
        {
            "conversation_id": conversation_id,
            "participants": [
                {
                    "id": "me",
                    "name": "我",
                    "lng": 116.3,
                    "lat": 39.9,
                    "address": "国贸",
                    "prefer": "transit",
                }
            ],
            "anchor": {"name": "国贸", "lng": 116.3, "lat": 39.9, "radius_m": 3000},
            "last_pois": [{"id": "poi-1", "name": "安静咖啡馆", "legs": []}],
            "query": "安静咖啡馆",
            "city": "北京",
            "memory_did": "device-a",
            "my_did": "device-a",
            "agent_task": {
                "id": "task-recovery",
                "status": "running",
                "completed": [],
                "failures": [],
            },
        }
    )
    module._conversation_append_event(conversation_id, "device-a", "user", "继续规划")
    module._conversation_append_event(conversation_id, "device-a", "assistant", "可以")
    module._conversation_save_summary(sid, "用户正在规划国贸附近的安静咖啡馆。")
    module._conversation_save_state(sid)
    module._conversation_record_operation(
        sid, "set_keyword", {"ok": True, "summary": "关键词已改为安静咖啡馆"}
    )
    _, patch = module._tool_offer_choices(
        sid,
        {
            "question": "你想怎么过去？",
            "mode": "single",
            "options": [{"label": "坐地铁"}, {"label": "打车"}],
        },
    )

    module._sessions.clear()
    module._choice_graph_conn.close()
    module._choice_graph_runtime = None
    module._choice_graph_conn = None
    with module.app.test_request_context():
        g.device_id = "device-a"
        response = module.api_conversation_continue(conversation_id).get_json()

    assert response["state"]["query"] == "安静咖啡馆"
    assert response["state"]["participants"][0]["address"] == "国贸"
    assert response["has_summary"] is True
    assert "关键词已改为安静咖啡馆" in response["operation_summaries"][0]
    assert response["pending_interaction"]["patch"]["token"] == patch["token"]

    resumed_sid = response["session_id"]
    question, labels = module._consume_offer_choice_answers(
        resumed_sid, [{"token": patch["token"], "label": "坐地铁"}]
    )
    assert question == "你想怎么过去？"
    assert labels == ["坐地铁"]
    conn = module._db_connect()
    try:
        recovery = conn.execute(
            "SELECT pending_interrupt_id FROM conversation_recovery WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()
        assert recovery["pending_interrupt_id"] is None
    finally:
        conn.close()


def test_apply_drafts_does_not_reference_assistant_message(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    sid = module.session_create(
        {
            "participants": [],
            "query": "咖啡",
            "memory_did": "device-a",
            "my_did": "device-a",
        }
    )
    response = module.app.test_client().post(
        "/api/v2/session/apply-drafts",
        json={
            "session_id": sid,
            "drafts": [{"kind": "set_keyword", "data": {"keyword": "日料"}}],
        },
    )

    assert response.status_code == 200
    assert response.get_json()["query"] == "日料"


def test_agent_exposes_unified_participant_tools(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    names = {(tool.get("function") or {}).get("name") for tool in module.ASSISTANT_TOOLS}
    assert "ensure_participant" in names
    assert "remove_participant" in names
    assert "set_participant_location" not in names
    assert "add_participant" not in names


def test_ensure_participant_creates_when_list_is_empty(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    sid = module.session_create(
        {
            "participants": [],
            "city": "北京",
            "memory_did": "device-a",
            "my_did": "device-a",
        }
    )
    result, patch = module._tool_ensure_participant(
        sid,
        {"index": 1, "participant_name": "我", "lng": 116.326, "lat": 40.003},
    )
    assert result["ok"] is True
    assert result["action"] == "created"
    assert patch["kind"] == "add_participant"
    assert patch["data"]["nickname"] == "我"
    assert module.session_get(sid)["agent_task"]["participant_drafts_pending"] is True


def test_search_waits_for_participant_draft_confirmation(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    sid = module.session_create(
        {
            "participants": [{"id": "a", "name": "A", "lng": 1.0, "lat": 1.0}],
            "query": "咖啡",
            "memory_did": "device-a",
            "my_did": "device-a",
        }
    )
    module._tool_ensure_participant(sid, {"index": 1, "participant_name": "E"})

    result, patch = module._tool_search_pois(sid, {"keyword": "咖啡"})

    assert result["ok"] is True
    assert result["skipped"] is True
    assert patch is None


def test_replace_abc_with_ef_and_remove_c_in_one_draft_batch(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    sid = module.session_create(
        {
            "participants": [
                {"id": "a", "name": "A", "lng": 1.0, "lat": 1.0, "prefer": "auto"},
                {"id": "b", "name": "B", "lng": 2.0, "lat": 2.0, "prefer": "auto"},
                {"id": "c", "name": "C", "lng": 3.0, "lat": 3.0, "prefer": "auto"},
            ],
            "city": "北京",
            "memory_did": "device-a",
            "my_did": "device-a",
        }
    )
    _, e_patch = module._tool_ensure_participant(sid, {"index": 1, "participant_name": "E"})
    _, f_patch = module._tool_ensure_participant(sid, {"index": 2, "participant_name": "F"})
    _, remove_patch = module._tool_remove_participant(sid, {"index": 3})
    response = module.app.test_client().post(
        "/api/v2/session/apply-drafts",
        json={
            "session_id": sid,
            "drafts": [
                {"kind": e_patch["kind"], "data": e_patch["data"]},
                {"kind": f_patch["kind"], "data": f_patch["data"]},
                {"kind": remove_patch["kind"], "data": remove_patch["data"]},
            ],
        },
    )
    assert response.status_code == 200
    assert [item["name"] for item in response.get_json()["participants"]] == ["E", "F"]


def test_participant_batch_automatically_searches_after_confirmation(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    sid = module.session_create(
        {
            "participants": [{"id": "a", "name": "A", "lng": 1.0, "lat": 1.0}],
            "query": "咖啡",
            "memory_did": "device-a",
            "my_did": "device-a",
        }
    )
    _, draft = module._tool_ensure_participant(
        sid, {"index": 1, "participant_name": "E", "lng": 2.0, "lat": 2.0}
    )
    calls = []

    def fake_search(search_sid, args):
        calls.append((search_sid, args))
        pois = [{"id": "poi-1", "name": "测试咖啡馆", "legs": []}]
        module.session_update(search_sid, {"last_pois": pois, "pois_base": pois})
        return {"ok": True, "count": 1, "summary": "找到 1 家"}, None

    monkeypatch.setattr(module, "_tool_search_pois", fake_search)
    response = module.app.test_client().post(
        "/api/v2/session/apply-drafts",
        json={"session_id": sid, "drafts": [{"kind": draft["kind"], "data": draft["data"]}]},
    )

    assert response.status_code == 200
    assert calls == [(sid, {"keyword": "咖啡"})]
    assert response.get_json()["pois"][0]["name"] == "测试咖啡馆"


def test_participant_draft_batch_rejects_partial_application(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    sid = module.session_create(
        {
            "participants": [{"id": "a", "name": "A", "lng": 1.0, "lat": 1.0}],
            "memory_did": "device-a",
            "my_did": "device-a",
        }
    )
    response = module.app.test_client().post(
        "/api/v2/session/apply-drafts",
        json={
            "session_id": sid,
            "drafts": [
                {"kind": "set_participant_location", "data": {"participant_id": "a", "new_nickname": "E"}},
                {"kind": "remove_participant", "data": {"participant_id": "missing"}},
            ],
        },
    )
    assert response.status_code == 409
    assert module.session_get(sid)["participants"][0]["name"] == "A"


def test_conversation_detail_replays_sanitized_tool_timeline(monkeypatch, tmp_path):
    module = _load_app(monkeypatch, tmp_path)
    conversation_id = module._conversation_create("device-a", "规划会面")
    trace_id = module._trace_start(conversation_id, "device-a", "sid-a", "我从清华出发")
    module._trace_step(
        trace_id,
        "tool_call",
        "调用 set_participant_location",
        tool_name="set_participant_location",
        payload={"runtime": "langgraph", "arguments": {"index": 1, "place_name": "清华"}},
    )
    module._trace_step(
        trace_id,
        "tool_result",
        "set_participant_location · 失败",
        tool_name="set_participant_location",
        summary="当前没有参与者，用 add_participant 先加人",
        payload={"runtime": "langgraph", "result": {"ok": False, "error": "当前没有参与者"}},
    )
    module._trace_step(
        trace_id,
        "tool_call",
        "调用 add_participant",
        tool_name="add_participant",
        payload={"runtime": "langgraph", "arguments": {"nickname": "我", "place_name": "清华"}},
    )
    module._trace_step(
        trace_id,
        "tool_result",
        "add_participant · 成功",
        tool_name="add_participant",
        summary="已准备新增我",
        payload={"runtime": "langgraph", "result": {"ok": True, "summary": "已准备新增我"}},
    )
    module._trace_step(
        trace_id,
        "assistant",
        "阿觅回复",
        summary="已经准备好了",
        payload={"content": "已经准备好了"},
    )
    module._trace_finish(trace_id, "done")
    with module.app.test_request_context():
        g.device_id = "device-a"
        detail = module.api_conversation_detail(conversation_id).get_json()
    assert len(detail["turns"]) == 1
    assert detail["turns"][0]["tools"][0]["status"] == "recovered"
    assert detail["turns"][0]["tools"][1]["status"] == "ok"
    assert detail["turns"][0]["assistant_content"] == "已经准备好了"
