from __future__ import annotations

from collections import Counter

from langgraph.checkpoint.memory import InMemorySaver

from middot.agent_runtime.main_graph import (
    MainAgentHooks,
    MainAgentRuntime,
    build_main_agent_graph,
)


class FakeHooks:
    def __init__(self, planner_outputs):
        self.planner_outputs = list(planner_outputs)
        self.calls = Counter()
        self.history = []
        self.tool_results = {}
        self.pois = False

    def call_model(self, state):
        self.calls["planner"] += 1
        return self.planner_outputs.pop(0)

    def execute_tool(self, state, name, args):
        self.calls[f"tool:{name}"] += 1
        return self.tool_results.get(
            name, ({"ok": True, "summary": f"{name} 完成", "args": args}, None)
        )

    def append_history(self, sid, message):
        self.history.append((sid, message))

    def verify(self, sid, names):
        self.calls["verify"] += 1
        return []

    def has_pois(self, sid):
        return self.pois

    def auto_recompute(self, state):
        self.calls["auto_recompute"] += 1
        return {"ok": True, "summary": "路线已重算"}, {"type": "routes"}

    def finalize(self, state, content):
        self.calls["finalize"] += 1
        return content

    def mark_waiting(self, state, kind):
        self.calls[f"waiting:{kind}"] += 1

    def mark_failed(self, state, error):
        self.calls["failed"] += 1

    def hooks(self):
        return MainAgentHooks(
            call_model=self.call_model,
            execute_tool=self.execute_tool,
            append_history=self.append_history,
            verify=self.verify,
            has_pois=self.has_pois,
            auto_recompute_routes=self.auto_recompute,
            finalize=self.finalize,
            mark_waiting=self.mark_waiting,
            mark_failed=self.mark_failed,
        )


def _state(**updates):
    state = {
        "request_id": "request-1",
        "thread_id": "agent:thread-1",
        "session_id": "session-1",
        "trace_id": "trace-1",
        "conversation_id": "conversation-1",
        "caller_device_id": "device-1",
        "messages": [{"role": "user", "content": "找咖啡馆"}],
        "tools": [],
        "iteration": 0,
        "max_iterations": 7,
        "successful_tool_signatures": [],
        "routes_recomputed_after_prefer": False,
        "me_has_location": True,
    }
    state.update(updates)
    return state


def _runtime(fake):
    graph = build_main_agent_graph(
        hooks=fake.hooks(),
        checkpointer=InMemorySaver(),
    )
    return MainAgentRuntime(graph)


def _tool_call(name, arguments="{}", call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_main_graph_runs_planner_tool_verify_and_finalize():
    fake = FakeHooks(
        [
            {"content": "", "tool_calls": [_tool_call("set_keyword", '{"keyword":"咖啡"}')]},
            {"content": "已经帮你找好了。", "tool_calls": []},
        ]
    )
    events = list(_runtime(fake).stream(_state(), thread_id="agent:thread-1"))

    assert [event["type"] for event in events] == [
        "tool_call",
        "tool_result",
        "token",
        "done",
    ]
    assert fake.calls["planner"] == 2
    assert fake.calls["tool:set_keyword"] == 1
    assert fake.calls["verify"] == 1
    assert fake.calls["finalize"] == 1


def test_main_graph_stops_at_location_choice_without_running_later_tools():
    fake = FakeHooks(
        [
            {
                "content": "",
                "tool_calls": [
                    _tool_call("set_participant_location", '{"index":1,"place_name":"清华"}', "loc"),
                    _tool_call("search_pois", '{"keyword":"咖啡"}', "search"),
                ],
            }
        ]
    )
    fake.tool_results["set_participant_location"] = (
        {"ok": True, "summary": "等待确认"},
        {"type": "location_choices", "options": []},
    )
    events = list(_runtime(fake).stream(_state(), thread_id="agent:thread-wait"))

    assert fake.calls["tool:set_participant_location"] == 1
    assert fake.calls["tool:search_pois"] == 0
    assert events[-2:] == [
        {"type": "waiting", "kind": "location_choice", "label": "等待你选择"},
        {"type": "done"},
    ]


def test_main_graph_deduplicates_successful_tool_signature_across_iterations():
    repeated = _tool_call("set_keyword", '{"keyword":"咖啡"}')
    fake = FakeHooks(
        [
            {"content": "", "tool_calls": [repeated]},
            {"content": "", "tool_calls": [repeated]},
            {"content": "完成", "tool_calls": []},
        ]
    )
    events = list(_runtime(fake).stream(_state(), thread_id="agent:thread-dedupe"))

    assert fake.calls["tool:set_keyword"] == 1
    assert [event["type"] for event in events].count("tool_call") == 1
    assert events[-1] == {"type": "done"}


def test_main_graph_auto_recomputes_routes_after_transport_change():
    fake = FakeHooks(
        [
            {
                "content": "",
                "tool_calls": [
                    _tool_call(
                        "set_participant_location",
                        '{"index":1,"place_name":"清华","prefer":"transit"}',
                    )
                ],
            },
            {"content": "路线更新好了", "tool_calls": []},
        ]
    )
    fake.pois = True
    events = list(_runtime(fake).stream(_state(), thread_id="agent:thread-routes"))

    assert fake.calls["auto_recompute"] == 1
    assert any(event.get("id") == "auto_recompute_1" for event in events)


def test_main_graph_reports_iteration_limit():
    fake = FakeHooks(
        [{"content": "", "tool_calls": [_tool_call("get_current_result")]}]
    )
    events = list(
        _runtime(fake).stream(
            _state(max_iterations=1), thread_id="agent:thread-limit"
        )
    )

    assert events[-2]["type"] == "error"
    assert "工具调用上限" in events[-2]["msg"]
    assert events[-1] == {"type": "done"}
    assert fake.calls["failed"] == 1
