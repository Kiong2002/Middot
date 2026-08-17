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
    needs_search: Callable[[str, str], bool]
    auto_search: Callable[[MainAgentState, str], ToolExecution]
    finalize: Callable[[MainAgentState, str], str]
    mark_waiting: Callable[[MainAgentState, str], None]
    mark_failed: Callable[[MainAgentState, str], None]


def _signature(name: str, args: Mapping[str, Any]) -> str:
    return name + "\x1f" + json.dumps(
        args, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _failure_key(name: str, args: Mapping[str, Any]) -> str:
    if name in {"ensure_participant", "set_participant_location"}:
        target = args.get("index") or args.get("participant_id") or args.get("participant_name")
        return f"{name}\x1fparticipant:{target or ''}"
    return _signature(name, args)


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
            desired_keyword = str(state.get("desired_search_keyword") or "").strip()
            if (
                desired_keyword
                and not state.get("search_compensated")
                and hooks.needs_search(state["session_id"], desired_keyword)
            ):
                return {
                    "iteration": iteration,
                    "planner_content": content,
                    "pending_tool_calls": [],
                    "status": "compensating_search",
                }
            issues = hooks.verify(state["session_id"], set())
            repair_attempts = int(state.get("repair_attempts") or 0)
            if issues and repair_attempts < 2 and iteration < int(state.get("max_iterations") or 7):
                repair_message = {
                    "role": "system",
                    "content": "结束前校验发现尚未闭环："
                    + "；".join(issues)
                    + "。不得只用文字声称完成；请立即调用必要工具，或如实说明阻塞原因。",
                }
                return {
                    "iteration": iteration,
                    "planner_content": "",
                    "pending_tool_calls": [],
                    "messages": [*(state.get("messages") or []), repair_message],
                    "verification_issues": issues,
                    "repair_attempts": repair_attempts + 1,
                    "status": "repairing",
                }
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
        if state.get("status") == "compensating_search":
            return "compensate_search"
        if state.get("status") == "repairing":
            return "planner"
        return "finalize"

    def compensate_search(state: MainAgentState) -> Mapping[str, Any]:
        writer = get_stream_writer()
        keyword = str(state.get("desired_search_keyword") or "").strip()
        call_id = f"auto_search_{state.get('iteration', 0)}"
        args = {"keyword": keyword}
        writer({"type": "tool_call", "id": call_id, "name": "search_pois", "args": args})
        with sink.span(
            "agent.tool.search_pois.compensation",
            inputs={"arguments": args},
            metadata={"request_id": state["request_id"]},
        ) as span:
            result, patch = hooks.auto_search(state, keyword)
            span.set_outputs({"result": result, "state_patch": patch})
        writer(
            {
                "type": "tool_result",
                "id": call_id,
                "name": "search_pois",
                "ok": bool(result.get("ok")) and not bool(result.get("skipped")),
                "summary": result.get("summary") or result.get("error") or "",
                "data": result,
            }
        )
        if patch:
            writer({"type": "state_patch", "patch": patch})
        messages = list(state.get("messages") or [])
        messages.append(
            {
                "role": "system",
                "content": "系统已按结构化搜索目标补做地点搜索。真实结果："
                + json.dumps(result, ensure_ascii=False)
                + "。最终回复只能依据这个结果，不得引用旧关键词的推荐。",
            }
        )
        return {
            "messages": messages,
            "search_compensated": True,
            "called_names": sorted(set(state.get("called_names") or []) | {"search_pois"}),
            "status": "planning" if result.get("ok") and not result.get("skipped") else "repairing",
        }

    def execute_tools(state: MainAgentState) -> Mapping[str, Any]:
        writer = get_stream_writer()
        messages = list(state.get("messages") or [])
        successful = set(state.get("successful_tool_signatures") or [])
        non_retryable_failures = set(state.get("non_retryable_tool_failures") or [])
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
            failure_key = _failure_key(name, args)
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

            if failure_key in non_retryable_failures:
                result = {
                    "ok": False,
                    "error": "相同内部错误本轮已经记录，已停止重复尝试",
                    "error_code": "non_retryable_failure_already_reported",
                    "retryable": False,
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
            elif result.get("retryable") is False:
                non_retryable_failures.add(failure_key)
                messages.append({
                    "role": "system",
                    "content": (
                        f"工具 {name} 对当前目标发生不可重试的系统内部错误。"
                        "本轮不得再次调用同一工具处理同一目标；请如实简短说明一次，"
                        "不要让用户反复重试同一个故障。"
                    ),
                })
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
            "non_retryable_tool_failures": sorted(non_retryable_failures),
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
    builder.add_node("compensate_search", compensate_search)
    builder.add_node("wait", wait)
    builder.add_node("finalize", finalize)
    builder.add_node("fail", fail)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "execute_tools": "execute_tools",
            "compensate_search": "compensate_search",
            "planner": "planner",
            "finalize": "finalize",
            "fail": "fail",
        },
    )
    builder.add_edge("compensate_search", "planner")
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
