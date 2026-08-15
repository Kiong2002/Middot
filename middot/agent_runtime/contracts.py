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
