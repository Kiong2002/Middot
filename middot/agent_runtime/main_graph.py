"""Middot 主 Agent 的 LangGraph 编排层。

业务工具仍由应用提供；本模块只负责 planner -> tools -> compensation -> verify ->
finalize 的控制流、checkpoint 状态和 SSE 业务事件。这样可以独立回放编排，同时不把
高德、记忆或房间业务重写进框架。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from .contracts import MainAgentState, PlannerResult
from .trace import NullTraceSink, TraceSink

ToolExecution = tuple[dict[str, Any], dict[str, Any] | None]


@dataclass(frozen=True)
class MainAgentHooks:
    call_model: Callable[[MainAgentState], PlannerResult]
    execute_tool: Callable[[MainAgentState, str, dict[str, Any]], ToolExecution]
    append_history: Callable[[str, dict[str, Any]], None]
    verify: Callable[[str, set[str]], list[str]]
    has_pois: Callable[[str], bool]
    auto_recompute_routes: Callable[[MainAgentState], ToolExecution]
    finalize: Callable[[MainAgentState, str], str]
    mark_waiting: Callable[[MainAgentState, str], None]
    mark_failed: Callable[[MainAgentState, str], None]


def _signature(name: str, args: Mapping[str, Any]) -> str:
    return name + "\x1f" + json.dumps(
        args, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _tool_message(call_id: str, name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def build_main_agent_graph(
    *,
    hooks: MainAgentHooks,
    checkpointer: Any,
    trace_sink: TraceSink | None = None,
):
    sink = trace_sink or NullTraceSink()

    def planner(state: MainAgentState) -> Mapping[str, Any]:
        iteration = int(state.get("iteration") or 0) + 1
        if iteration > int(state.get("max_iterations") or 7):
            return {"status": "failed", "error": "达到工具调用上限"}
        with sink.span(
            "agent.planner",
            inputs={
                "iteration": iteration,
                "message_count": len(state.get("messages") or []),
                "tool_count": len(state.get("tools") or []),
            },
            metadata={"request_id": state["request_id"]},
        ) as span:
            output = hooks.call_model({**state, "iteration": iteration})
            tool_calls = list(output.get("tool_calls") or [])
            content = str(output.get("content") or "")
            span.set_outputs({"content": content, "tool_calls": tool_calls})

        if not tool_calls:
            return {
                "iteration": iteration,
                "planner_content": content,
                "pending_tool_calls": [],
                "status": "finalizing",
            }

        assistant_message = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": tool_calls,
        }
        hooks.append_history(state["session_id"], assistant_message)
        return {
            "iteration": iteration,
            "planner_content": content,
            "pending_tool_calls": tool_calls,
            "messages": [*(state.get("messages") or []), assistant_message],
            "status": "executing_tools",
        }

    def route_after_planner(state: MainAgentState) -> str:
        if state.get("status") == "failed":
            return "fail"
        if state.get("pending_tool_calls"):
            return "execute_tools"
        return "finalize"

    def execute_tools(state: MainAgentState) -> Mapping[str, Any]:
        writer = get_stream_writer()
        messages = list(state.get("messages") or [])
        successful = set(state.get("successful_tool_signatures") or [])
        called_names: set[str] = set()
        location_targets: set[str] = set()
        prefer_changed = False
        waiting_kind = ""

        for call in state.get("pending_tool_calls") or []:
            function = dict(call.get("function") or {})
            name = str(function.get("name") or "")
            call_id = str(call.get("id") or "")
            called_names.add(name)
            try:
                args = json.loads(str(function.get("arguments") or "{}"))
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}

            signature = _signature(name, args)
            if signature in successful:
                result = {
                    "ok": True,
                    "summary": "相同动作本轮已经完成，无需重复执行",
                    "duplicate": True,
                }
                message = _tool_message(call_id, name, result)
                messages.append(message)
                hooks.append_history(state["session_id"], message)
                continue

            if name == "set_participant_location":
                target = str(args.get("index") or args.get("participant_name") or "")
                if target in location_targets:
                    result = {"ok": False, "error": "同一人物本轮已有位置动作，重复调用已忽略"}
                    message = _tool_message(call_id, name, result)
                    messages.append(message)
                    hooks.append_history(state["session_id"], message)
                    continue
                location_targets.add(target)

            if waiting_kind:
                result = {
                    "ok": False,
                    "error": (
                        "正在等待用户确认具体位置"
                        if waiting_kind == "location_choice"
                        else "正在等待用户选择"
                    ),
                }
                message = _tool_message(call_id, name, result)
                messages.append(message)
                hooks.append_history(state["session_id"], message)
                continue

            writer({"type": "tool_call", "id": call_id, "name": name, "args": args})
            with sink.span(
                f"agent.tool.{name or 'unknown'}",
                inputs={"arguments": args},
                metadata={"request_id": state["request_id"]},
            ) as span:
                result, patch = hooks.execute_tool(state, name, args)
                span.set_outputs({"result": result, "state_patch": patch})

            writer(
                {
                    "type": "tool_result",
                    "id": call_id,
                    "name": name,
                    "ok": bool(result.get("ok")),
                    "summary": result.get("summary") or result.get("error") or "",
                    "data": result,
                }
            )
            if result.get("ok"):
                successful.add(signature)
                if name == "set_participant_prefer" or (
                    name in {"set_participant_location", "ensure_participant"}
                    and args.get("prefer")
                ):
                    prefer_changed = True
            if patch:
                writer({"type": "state_patch", "patch": patch})
                if patch.get("type") == "location_choices":
                    waiting_kind = "location_choice"
                elif patch.get("type") == "choices":
                    waiting_kind = "choice"

            message = _tool_message(call_id, name, result)
            messages.append(message)
            hooks.append_history(state["session_id"], message)

        if (
            not waiting_kind
            and prefer_changed
            and "recompute_routes" not in called_names
            and not state.get("routes_recomputed_after_prefer")
            and hooks.has_pois(state["session_id"])
        ):
            auto_id = f"auto_recompute_{state.get('iteration', 0)}"
            writer({"type": "tool_call", "id": auto_id, "name": "recompute_routes", "args": {}})
            result, patch = hooks.auto_recompute_routes(state)
            writer(
                {
                    "type": "tool_result",
                    "id": auto_id,
                    "name": "recompute_routes",
                    "ok": bool(result.get("ok")),
                    "summary": result.get("summary") or result.get("error") or "",
                    "data": result,
                }
            )
            if patch:
                writer({"type": "state_patch", "patch": patch})
            routes_recomputed = bool(result.get("ok"))
        else:
            routes_recomputed = bool(state.get("routes_recomputed_after_prefer"))

        issues = hooks.verify(state["session_id"], called_names)
        if issues:
            messages.append(
                {
                    "role": "system",
                    "content": "执行校验发现尚未闭环："
                    + "；".join(issues)
                    + "。请修复后再向用户宣称完成。",
                }
            )
        if state.get("me_has_location"):
            messages.append(
                {
                    "role": "system",
                    "content": "最终回复校验：‘我’在本轮开始时已有设备定位。不得说‘你的位置没填/没设’，也不得让用户再次定位。",
                }
            )
        return {
            "messages": messages,
            "pending_tool_calls": [],
            "successful_tool_signatures": sorted(successful),
            "called_names": sorted(called_names),
            "verification_issues": issues,
            "waiting_kind": waiting_kind,
            "routes_recomputed_after_prefer": routes_recomputed,
            "status": "waiting_user" if waiting_kind else "planning",
        }

    def route_after_tools(state: MainAgentState) -> str:
        if state.get("waiting_kind"):
            return "wait"
        if int(state.get("iteration") or 0) >= int(state.get("max_iterations") or 7):
            return "fail"
        return "planner"

    def wait(state: MainAgentState) -> Mapping[str, Any]:
        kind = str(state.get("waiting_kind") or "choice")
        hooks.mark_waiting(state, kind)
        writer = get_stream_writer()
        writer(
            {
                "type": "waiting",
                "kind": kind,
                "label": "等待你选择",
            }
        )
        writer({"type": "done", "outcome": "waiting"})
        return {"status": "waiting_user"}

    def finalize(state: MainAgentState) -> Mapping[str, Any]:
        with sink.span(
            "agent.finalize",
            inputs={"content": state.get("planner_content") or ""},
            metadata={"request_id": state["request_id"]},
        ) as span:
            content = hooks.finalize(state, str(state.get("planner_content") or ""))
            span.set_outputs({"content": content})
        writer = get_stream_writer()
        if content:
            writer({"type": "token", "delta": content})
        writer({"type": "done", "outcome": "completed"})
        return {"final_response": content, "status": "completed"}

    def fail(state: MainAgentState) -> Mapping[str, Any]:
        error = str(state.get("error") or "达到工具调用上限")
        if error == "达到工具调用上限":
            error = f"达到工具调用上限 ({state.get('max_iterations', 7)})"
        hooks.mark_failed(state, error)
        writer = get_stream_writer()
        writer({"type": "error", "msg": error})
        writer({"type": "done", "outcome": "failed"})
        return {"status": "failed", "error": error}

    builder = StateGraph(MainAgentState)
    builder.add_node("planner", planner)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("wait", wait)
    builder.add_node("finalize", finalize)
    builder.add_node("fail", fail)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"execute_tools": "execute_tools", "finalize": "finalize", "fail": "fail"},
    )
    builder.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {"planner": "planner", "wait": "wait", "fail": "fail"},
    )
    builder.add_edge("wait", END)
    builder.add_edge("finalize", END)
    builder.add_edge("fail", END)
    return builder.compile(checkpointer=checkpointer)


class MainAgentRuntime:
    def __init__(self, graph: Any):
        self._graph = graph

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "main_agent",
            }
        }

    def stream(
        self, state: MainAgentState, *, thread_id: str
    ) -> Iterator[dict[str, Any]]:
        for event in self._graph.stream(
            state,
            config=self._config(thread_id),
            stream_mode="custom",
        ):
            if isinstance(event, Mapping):
                yield dict(event)
