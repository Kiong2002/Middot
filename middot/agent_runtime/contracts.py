"""地点消歧纵切使用的显式数据契约。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict


class LocationCandidate(TypedDict):
    id: str
    name: str
    address: str
    longitude: float
    latitude: float


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
    candidates: list[LocationCandidate]
    auto_selection: AutoSelection
    selected_candidate_id: str
    selection_source: str
    choice_error: str
    operation_id: str
    commit_result: dict[str, Any]


class CandidateResolver(Protocol):
    def __call__(self, query: str, city: str) -> Sequence[LocationCandidate]: ...


class CandidateSelector(Protocol):
    def __call__(
        self,
        query: str,
        city: str,
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
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class LocationRunOutcome:
    status: str
    thread_id: str
    request_id: str
    interrupt_id: str = ""
    prompt: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
