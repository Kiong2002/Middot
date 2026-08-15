"""地点消歧纵切使用的显式数据契约。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict


class LocationCandidate(TypedDict):
    id: str
    label: str
    address: str
    lng: float
    lat: float


class AutoSelection(TypedDict):
    candidate_id: str
    confidence: float
    reason: str


class LocationGraphState(TypedDict, total=False):
    request_id: str
    thread_id: str
    participant_id: str
    query: str
    city: str
    metadata: dict[str, Any]
    candidates: list[LocationCandidate]
    auto_selection: AutoSelection
    selected_candidate_id: str
    selection_source: str
    choice_error: str
    operation_id: str
    commit_result: dict[str, Any]


class CandidateResolver(Protocol):
    def __call__(self, state: LocationGraphState) -> Sequence[LocationCandidate]: ...


class CandidateSelector(Protocol):
    def __call__(
        self,
        state: LocationGraphState,
        candidates: Sequence[LocationCandidate],
    ) -> AutoSelection | None: ...


class LocationCommitter(Protocol):
    def __call__(
        self,
        *,
        operation_id: str,
        request_id: str,
        participant_id: str,
        candidate: LocationCandidate,
        selection_source: str,
        state: LocationGraphState,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class LocationRunOutcome:
    status: str
    thread_id: str
    request_id: str
    interrupt_id: str = ""
    prompt: dict[str, Any] | None = None
    result: dict[str, Any] | None = None


class PlannerResult(TypedDict):
    content: str
    tool_calls: list[dict[str, Any]]


class MainAgentState(TypedDict, total=False):
    """单次 Agent turn 的可持久化状态。

    这里不放数据库连接、Flask request 或函数对象；checkpoint 只保存恢复执行所需的
    纯数据。页面 session 是这个状态的投影，不再承担编排游标的职责。
    """

    request_id: str
    thread_id: str
    session_id: str
    trace_id: str
    conversation_id: str
    caller_device_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    iteration: int
    max_iterations: int
    successful_tool_signatures: list[str]
    routes_recomputed_after_prefer: bool
    me_has_location: bool
    planner_content: str
    pending_tool_calls: list[dict[str, Any]]
    called_names: list[str]
    verification_issues: list[str]
    waiting_kind: str
    final_response: str
    status: str
    error: str
