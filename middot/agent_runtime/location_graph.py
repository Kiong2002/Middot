"""地点消歧的 LangGraph 纵切试验。

查询、自动判断、等待用户和写入被拆为独立节点。interrupt 所在节点没有
外部副作用；写入节点必须使用稳定 operation_id 实现恰好一次语义。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .contracts import (
    AutoSelection,
    CandidateResolver,
    CandidateSelector,
    LocationCandidate,
    LocationCommitter,
    LocationGraphState,
    LocationRunOutcome,
)
from .trace import NullTraceSink, TraceSink


def _candidate_by_id(
    candidates: Sequence[LocationCandidate], candidate_id: str
) -> LocationCandidate | None:
    return next((item for item in candidates if item["id"] == candidate_id), None)


def _operation_id(state: LocationGraphState, candidate_id: str) -> str:
    material = "\x1f".join(
        [state["thread_id"], state["request_id"], state["participant_id"], candidate_id]
    )
    return "loc_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def build_location_graph(
    *,
    resolver: CandidateResolver,
    selector: CandidateSelector,
    committer: LocationCommitter,
    checkpointer: Any,
    trace_sink: TraceSink | None = None,
    auto_select_threshold: float = 0.90,
):
    sink = trace_sink or NullTraceSink()

    def resolve_candidates(state: LocationGraphState) -> Mapping[str, Any]:
        with sink.span(
            "location.resolve_candidates",
            inputs={"query": state["query"], "city": state.get("city", "")},
            metadata={"request_id": state["request_id"]},
        ) as span:
            candidates = list(resolver(state))
            if not candidates:
                raise ValueError("no location candidates")
            if len({item["id"] for item in candidates}) != len(candidates):
                raise ValueError("candidate ids must be unique")
            span.set_outputs({"candidate_count": len(candidates)})
            return {"candidates": candidates}

    def auto_select(state: LocationGraphState) -> Mapping[str, Any]:
        candidates = state["candidates"]
        with sink.span(
            "location.auto_select",
            inputs={"query": state["query"], "candidates": candidates},
            metadata={"request_id": state["request_id"]},
        ) as span:
            decision: AutoSelection = selector(state, candidates) or {
                "candidate_id": "",
                "confidence": 0.0,
                "reason": "uncertain",
            }
            span.set_outputs({"decision": decision})
        candidate = _candidate_by_id(candidates, decision.get("candidate_id", ""))
        confident = (
            candidate is not None
            and float(decision.get("confidence", 0)) >= auto_select_threshold
        )
        update: dict[str, Any] = {"auto_selection": decision}
        if confident:
            update.update(
                selected_candidate_id=candidate["id"], selection_source="auto"
            )
        return update

    def route_after_auto(state: LocationGraphState) -> str:
        return (
            "commit_location"
            if state.get("selected_candidate_id")
            else "request_choice"
        )

    def request_choice(state: LocationGraphState) -> Mapping[str, Any]:
        answer = interrupt(
            {
                "kind": "location_choice",
                "request_id": state["request_id"],
                "participant_id": state["participant_id"],
                "query": state["query"],
                "candidates": state["candidates"],
                "error": state.get("choice_error", ""),
            }
        )
        candidate_id = str((answer or {}).get("candidate_id", ""))
        metadata = {
            **dict(state.get("metadata") or {}),
            **dict((answer or {}).get("metadata") or {}),
        }
        if _candidate_by_id(state["candidates"], candidate_id) is None:
            return {
                "selected_candidate_id": "",
                "selection_source": "",
                "choice_error": "invalid_candidate",
                "metadata": metadata,
            }
        return {
            "selected_candidate_id": candidate_id,
            "selection_source": "user",
            "choice_error": "",
            "metadata": metadata,
        }

    def route_after_choice(state: LocationGraphState) -> str:
        return (
            "commit_location"
            if state.get("selected_candidate_id")
            else "request_choice"
        )

    def commit_location(state: LocationGraphState) -> Mapping[str, Any]:
        candidate_id = state["selected_candidate_id"]
        candidate = _candidate_by_id(state["candidates"], candidate_id)
        if candidate is None:
            raise ValueError("selected candidate disappeared before commit")
        operation_id = _operation_id(state, candidate_id)
        with sink.span(
            "location.commit",
            inputs={
                "operation_id": operation_id,
                "participant_id": state["participant_id"],
                "candidate": candidate,
                "selection_source": state["selection_source"],
            },
            metadata={"request_id": state["request_id"]},
        ) as span:
            result = dict(
                committer(
                    operation_id=operation_id,
                    request_id=state["request_id"],
                    participant_id=state["participant_id"],
                    candidate=candidate,
                    selection_source=state["selection_source"],
                    state=state,
                )
            )
            span.set_outputs({"operation_id": operation_id, "result": result})
            return {"operation_id": operation_id, "commit_result": result}

    builder = StateGraph(LocationGraphState)
    builder.add_node("resolve_candidates", resolve_candidates)
    builder.add_node("auto_select", auto_select)
    builder.add_node("request_choice", request_choice)
    builder.add_node("commit_location", commit_location)
    builder.add_edge(START, "resolve_candidates")
    builder.add_edge("resolve_candidates", "auto_select")
    builder.add_conditional_edges(
        "auto_select",
        route_after_auto,
        {"request_choice": "request_choice", "commit_location": "commit_location"},
    )
    builder.add_conditional_edges(
        "request_choice",
        route_after_choice,
        {"request_choice": "request_choice", "commit_location": "commit_location"},
    )
    builder.add_edge("commit_location", END)
    return builder.compile(checkpointer=checkpointer)


class LocationResolutionRuntime:
    def __init__(self, graph: Any):
        self._graph = graph

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _outcome(result: Mapping[str, Any]) -> LocationRunOutcome:
        interrupted = result.get("__interrupt__") or ()
        if interrupted:
            item = interrupted[0]
            return LocationRunOutcome(
                status="waiting_user",
                thread_id=result["thread_id"],
                request_id=result["request_id"],
                interrupt_id=item.id,
                prompt=dict(item.value),
            )
        return LocationRunOutcome(
            status="completed",
            thread_id=result["thread_id"],
            request_id=result["request_id"],
            result={
                "operation_id": result["operation_id"],
                "selected_candidate_id": result["selected_candidate_id"],
                "selection_source": result["selection_source"],
                "commit_result": result["commit_result"],
            },
        )

    def start(
        self,
        *,
        thread_id: str,
        request_id: str,
        participant_id: str,
        query: str,
        city: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> LocationRunOutcome:
        state: LocationGraphState = {
            "thread_id": thread_id,
            "request_id": request_id,
            "participant_id": participant_id,
            "query": query,
            "city": city,
            "metadata": dict(metadata or {}),
        }
        result = self._graph.invoke(state, config=self._config(thread_id))
        return self._outcome(result)

    def resume(
        self,
        *,
        thread_id: str,
        interrupt_id: str,
        candidate_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> LocationRunOutcome:
        config = self._config(thread_id)
        snapshot = self._graph.get_state(config)
        current_interrupts = tuple(snapshot.interrupts or ())
        if not current_interrupts:
            raise ValueError("thread is not waiting for a location choice")
        if current_interrupts[0].id != interrupt_id:
            raise ValueError("stale or mismatched interrupt id")
        result = self._graph.invoke(
            Command(
                resume={"candidate_id": candidate_id, "metadata": dict(metadata or {})}
            ),
            config=config,
        )
        return self._outcome(result)
