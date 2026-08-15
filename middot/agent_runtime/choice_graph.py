"""普通选择卡的 durable interrupt/resume 子图。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ChoiceGraphState(TypedDict, total=False):
    thread_id: str
    request_id: str
    question: str
    mode: str
    options: list[dict[str, str]]
    purpose: str
    payload: dict[str, Any]
    labels: list[str]
    choice_error: str


@dataclass(frozen=True)
class ChoiceRunOutcome:
    status: str
    thread_id: str
    request_id: str
    interrupt_id: str = ""
    prompt: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


def build_choice_graph(*, checkpointer: Any):
    def request_choice(state: ChoiceGraphState) -> Mapping[str, Any]:
        answer = interrupt(
            {
                "kind": "choice",
                "request_id": state["request_id"],
                "question": state["question"],
                "mode": state.get("mode") or "single",
                "options": state.get("options") or [],
                "purpose": state.get("purpose") or "",
                "error": state.get("choice_error") or "",
            }
        )
        labels = [str(item).strip() for item in ((answer or {}).get("labels") or [])]
        allowed = {
            str(item.get("label") or "").strip()
            for item in (state.get("options") or [])
            if str(item.get("label") or "").strip()
        }
        valid = bool(labels) and len(labels) == len(set(labels)) and all(
            label in allowed for label in labels
        )
        if (state.get("mode") or "single") == "single" and len(labels) != 1:
            valid = False
        return {
            "labels": labels if valid else [],
            "choice_error": "" if valid else "invalid_choice",
        }

    def route_after_choice(state: ChoiceGraphState) -> str:
        return "complete" if state.get("labels") else "request_choice"

    def complete(state: ChoiceGraphState) -> Mapping[str, Any]:
        return {"choice_error": ""}

    builder = StateGraph(ChoiceGraphState)
    builder.add_node("request_choice", request_choice)
    builder.add_node("complete", complete)
    builder.add_edge(START, "request_choice")
    builder.add_conditional_edges(
        "request_choice",
        route_after_choice,
        {"request_choice": "request_choice", "complete": "complete"},
    )
    builder.add_edge("complete", END)
    return builder.compile(checkpointer=checkpointer)


class ChoiceResolutionRuntime:
    def __init__(self, graph: Any):
        self._graph = graph

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        # checkpoint_ns 由父图调用子图时使用；这里本身就是顶层图，thread_id
        # 已带 choice: 前缀。给顶层图硬塞 namespace 会让 get_state 将它误认
        # 为一个并不存在的子图。
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _outcome(result: Mapping[str, Any]) -> ChoiceRunOutcome:
        interrupted = result.get("__interrupt__") or ()
        if interrupted:
            item = interrupted[0]
            return ChoiceRunOutcome(
                status="waiting_user",
                thread_id=result["thread_id"],
                request_id=result["request_id"],
                interrupt_id=item.id,
                prompt=dict(item.value),
            )
        return ChoiceRunOutcome(
            status="completed",
            thread_id=result["thread_id"],
            request_id=result["request_id"],
            result={
                "labels": list(result.get("labels") or []),
                "purpose": result.get("purpose") or "",
                "payload": dict(result.get("payload") or {}),
            },
        )

    def start(
        self,
        *,
        thread_id: str,
        request_id: str,
        question: str,
        mode: str,
        options: list[dict[str, str]],
        purpose: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> ChoiceRunOutcome:
        state: ChoiceGraphState = {
            "thread_id": thread_id,
            "request_id": request_id,
            "question": question,
            "mode": mode,
            "options": options,
            "purpose": purpose,
            "payload": dict(payload or {}),
        }
        result = self._graph.invoke(state, config=self._config(thread_id))
        return self._outcome(result)

    def resume(
        self, *, thread_id: str, interrupt_id: str, labels: list[str]
    ) -> ChoiceRunOutcome:
        config = self._config(thread_id)
        snapshot = self._graph.get_state(config)
        current = tuple(snapshot.interrupts or ())
        if not current:
            raise ValueError("choice graph is not waiting")
        if current[0].id != interrupt_id:
            raise ValueError("stale or mismatched interrupt id")
        result = self._graph.invoke(Command(resume={"labels": labels}), config=config)
        return self._outcome(result)
