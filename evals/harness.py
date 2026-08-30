"""Deterministic scenario evaluation for the Middot LangGraph orchestrator.

This suite deliberately separates orchestration quality from live-model quality.
Planner outputs are replayed from a versioned dataset, while the real graph decides
deduplication, compensation, verification, waiting, failure and finalization.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from middot.agent_runtime.main_graph import (
    MainAgentHooks,
    MainAgentRuntime,
    build_main_agent_graph,
)


DATASET_PATH = Path(__file__).with_name("scenarios.jsonl")


def load_scenarios(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        if not item.get("id") or not item.get("category"):
            raise ValueError(f"{path}:{line_number}: id/category are required")
        scenarios.append(item)
    return scenarios


class ScenarioHooks:
    def __init__(self, scenario: dict[str, Any]):
        self.scenario = scenario
        self.planner_outputs = deepcopy(scenario.get("planner_outputs") or [])
        self.config = deepcopy(scenario.get("hook_config") or {})
        self.calls: Counter[str] = Counter()
        self.executed_tools: Counter[str] = Counter()
        self.tool_arguments: dict[str, list[dict[str, Any]]] = {}
        self.executed_names: set[str] = set()
        self.history: list[tuple[str, dict[str, Any]]] = []
        self.business = deepcopy(scenario.get("initial_business") or {})

    def call_model(self, _state):
        self.calls["planner"] += 1
        if not self.planner_outputs:
            return {"content": "评测脚本没有更多模型返回", "tool_calls": []}
        return self.planner_outputs.pop(0)

    def _configured_tool_result(self, name: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        configured = (self.config.get("tool_results") or {}).get(name)
        if not configured:
            return {"ok": True, "summary": f"{name} 完成"}, None
        result = deepcopy(configured.get("result") or {"ok": True, "summary": f"{name} 完成"})
        patch = deepcopy(configured.get("patch"))
        return result, patch

    def _apply_business_effect(self, name: str, args: dict[str, Any]) -> None:
        if name == "set_keyword":
            self.business["query"] = str(args.get("keyword") or "")
        elif name == "search_pois":
            self.business["query"] = str(args.get("keyword") or "")
            self.business["last_search_keyword"] = str(args.get("keyword") or "")
            self.business["pois_count"] = int(self.config.get("search_result_count", 3))
        elif name == "set_radius":
            self.business["radius_m"] = int(args.get("radius_m") or 0)
        elif name == "set_participant_prefer":
            index = int(args.get("index") or 0)
            participants = self.business.setdefault("participants", [])
            if 0 < index <= len(participants):
                participants[index - 1]["prefer"] = args.get("prefer")
        elif name in {"ensure_participant", "set_participant_location"}:
            index = int(args.get("index") or 0)
            if index <= 0:
                return
            participants = self.business.setdefault("participants", [])
            while len(participants) < index:
                participants.append({"name": "小伙伴"})
            current = participants[index - 1]
            if args.get("identity_action") == "replace":
                current = {}
                participants[index - 1] = current
            if args.get("participant_name"):
                current["name"] = args["participant_name"]
            for key in ("place_name", "city", "lng", "lat", "prefer"):
                if key in args:
                    current[key] = args[key]
        elif name == "remove_participant":
            participants = self.business.setdefault("participants", [])
            index = int(args.get("index") or 0)
            if 0 < index <= len(participants):
                participants.pop(index - 1)
            elif args.get("participant_name"):
                target = str(args["participant_name"])
                self.business["participants"] = [
                    item for item in participants if str(item.get("name")) != target
                ]
        elif name == "recompute_routes":
            self.business["route_revision"] = int(self.business.get("route_revision") or 0) + 1

    def execute_tool(self, _state, name, args):
        self.calls[f"tool:{name}"] += 1
        self.executed_tools[name] += 1
        self.executed_names.add(name)
        self.tool_arguments.setdefault(name, []).append(deepcopy(args))
        result, patch = self._configured_tool_result(name)
        if result.get("ok"):
            self._apply_business_effect(name, args)
        return result, patch

    def append_history(self, sid, message):
        self.history.append((sid, deepcopy(message)))

    def verify(self, _sid, _names):
        self.calls["verify"] += 1
        required = set(self.config.get("verify_requires_tools") or [])
        missing = sorted(required - self.executed_names)
        return [f"尚未执行 {name}" for name in missing]

    def has_pois(self, _sid):
        return bool(self.config.get("has_pois"))

    def auto_recompute(self, _state):
        self.calls["auto_recompute"] += 1
        self.business["route_revision"] = int(self.business.get("route_revision") or 0) + 1
        return {"ok": True, "summary": "路线已自动重算"}, {"type": "routes"}

    def needs_search(self, _sid, _keyword):
        self.calls["needs_search"] += 1
        return bool(self.config.get("search_needed"))

    def auto_search(self, _state, keyword):
        self.calls["auto_search"] += 1
        self.executed_names.add("search_pois")
        self.business["query"] = keyword
        self.business["last_search_keyword"] = keyword
        self.business["pois_count"] = int(self.config.get("search_result_count", 3))
        return (
            {"ok": True, "summary": f"已搜索{keyword}", "count": self.business["pois_count"]},
            {"type": "pois_replaced", "pois": [{"name": f"{keyword}候选"}]},
        )

    def finalize(self, _state, content):
        self.calls["finalize"] += 1
        return content

    def mark_waiting(self, _state, kind):
        self.calls[f"waiting:{kind}"] += 1

    def mark_failed(self, _state, _error):
        self.calls["failed"] += 1

    def hooks(self) -> MainAgentHooks:
        return MainAgentHooks(
            call_model=self.call_model,
            execute_tool=self.execute_tool,
            append_history=self.append_history,
            verify=self.verify,
            has_pois=self.has_pois,
            auto_recompute_routes=self.auto_recompute,
            needs_search=self.needs_search,
            auto_search=self.auto_search,
            finalize=self.finalize,
            mark_waiting=self.mark_waiting,
            mark_failed=self.mark_failed,
        )


def tool_call(name: str, args: dict[str, Any] | None = None, call_id: str | None = None) -> dict[str, Any]:
    return {
        "id": call_id or f"call-{name}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args or {}, ensure_ascii=False, separators=(",", ":")),
        },
    }


def _state(scenario: dict[str, Any]) -> dict[str, Any]:
    base = {
        "request_id": f"eval:{scenario['id']}",
        "thread_id": f"eval:{scenario['id']}",
        "session_id": f"eval-session:{scenario['id']}",
        "trace_id": f"eval-trace:{scenario['id']}",
        "conversation_id": f"eval-conversation:{scenario['id']}",
        "caller_device_id": "eval-device",
        "messages": [{"role": "user", "content": scenario.get("user_message") or scenario["id"]}],
        "tools": [],
        "iteration": 0,
        "max_iterations": 7,
        "successful_tool_signatures": [],
        "non_retryable_tool_failures": [],
        "routes_recomputed_after_prefer": False,
        "me_has_location": True,
        "desired_search_keyword": "",
        "search_compensated": False,
        "repair_attempts": 0,
        "status": "planning",
    }
    base.update(deepcopy(scenario.get("state") or {}))
    return base


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _event_tool_counts(events: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        str(event.get("name"))
        for event in events
        if event.get("type") == "tool_call" and event.get("name")
    )


def _check_expectations(
    scenario: dict[str, Any], hooks: ScenarioHooks, events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected = scenario.get("expect") or {}
    checks: list[tuple[str, bool, Any, Any]] = []
    done_events = [event for event in events if event.get("type") == "done"]
    actual_outcome = done_events[-1].get("outcome") if done_events else None
    event_tools = _event_tool_counts(events)

    if "outcome" in expected:
        checks.append(("outcome", actual_outcome == expected["outcome"], expected["outcome"], actual_outcome))
    for name, minimum in (expected.get("required_tools") or {}).items():
        checks.append((f"tool:{name}:min", event_tools[name] >= int(minimum), int(minimum), event_tools[name]))
    for name in expected.get("forbidden_tools") or []:
        checks.append((f"tool:{name}:forbidden", event_tools[name] == 0, 0, event_tools[name]))
    for name, maximum in (expected.get("max_tool_counts") or {}).items():
        checks.append((f"tool:{name}:max", event_tools[name] <= int(maximum), int(maximum), event_tools[name]))
    for name, count in (expected.get("execution_counts") or {}).items():
        checks.append((f"execution:{name}", hooks.executed_tools[name] == int(count), int(count), hooks.executed_tools[name]))
    for counter_name in ("planner", "verify", "auto_search", "auto_recompute", "failed"):
        key = f"{counter_name}_calls"
        if key in expected:
            checks.append((key, hooks.calls[counter_name] == int(expected[key]), int(expected[key]), hooks.calls[counter_name]))
    if "waiting_kind" in expected:
        kind = str(expected["waiting_kind"])
        actual = hooks.calls[f"waiting:{kind}"]
        checks.append(("waiting_kind", actual == 1, kind, actual))
    if "final_contains" in expected:
        final_text = "".join(str(event.get("delta") or "") for event in events if event.get("type") == "token")
        needle = str(expected["final_contains"])
        checks.append(("final_contains", needle in final_text, needle, final_text))
    if "business" in expected:
        for key, value in expected["business"].items():
            checks.append((f"business:{key}", hooks.business.get(key) == value, value, hooks.business.get(key)))

    return [
        {"name": name, "passed": passed, "expected": wanted, "actual": actual}
        for name, passed, wanted, actual in checks
    ]


@dataclass
class ScenarioResult:
    id: str
    category: str
    description: str
    passed: bool
    duration_ms: float
    checks: list[dict[str, Any]]
    events: list[dict[str, Any]]

    def compact(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "passed": self.passed,
            "duration_ms": round(self.duration_ms, 3),
            "failed_checks": [check for check in self.checks if not check["passed"]],
        }


def evaluate_scenario(scenario: dict[str, Any]) -> ScenarioResult:
    hooks = ScenarioHooks(scenario)
    graph = build_main_agent_graph(hooks=hooks.hooks(), checkpointer=InMemorySaver())
    runtime = MainAgentRuntime(graph)
    started = time.perf_counter()
    events = list(runtime.stream(_state(scenario), thread_id=f"eval:{scenario['id']}"))
    duration_ms = (time.perf_counter() - started) * 1000
    checks = _check_expectations(scenario, hooks, events)
    return ScenarioResult(
        id=str(scenario["id"]),
        category=str(scenario["category"]),
        description=str(scenario.get("description") or ""),
        passed=bool(checks) and all(check["passed"] for check in checks),
        duration_ms=duration_ms,
        checks=checks,
        events=events,
    )


def evaluate_all(scenarios: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    items = scenarios or load_scenarios()
    results = [evaluate_scenario(item) for item in items]
    categories: dict[str, dict[str, Any]] = {}
    for category in sorted({result.category for result in results}):
        selected = [result for result in results if result.category == category]
        passed = sum(result.passed for result in selected)
        categories[category] = {
            "passed": passed,
            "total": len(selected),
            "pass_rate": round(passed / len(selected) * 100, 2),
        }
    assertion_total = sum(len(result.checks) for result in results)
    assertion_passed = sum(
        check["passed"] for result in results for check in result.checks
    )
    durations = [result.duration_ms for result in results]
    passed = sum(result.passed for result in results)
    return {
        "evaluation_mode": "deterministic_orchestration_replay",
        "dataset_version": 1,
        "scenario_total": len(results),
        "scenario_passed": passed,
        "scenario_pass_rate": round(passed / len(results) * 100, 2) if results else 0.0,
        "assertion_total": assertion_total,
        "assertion_passed": assertion_passed,
        "assertion_pass_rate": round(assertion_passed / assertion_total * 100, 2) if assertion_total else 0.0,
        "duration_ms": {
            "p50": round(_percentile(durations, 0.50), 3),
            "p95": round(_percentile(durations, 0.95), 3),
        },
        "categories": categories,
        "results": [result.compact() for result in results],
    }

