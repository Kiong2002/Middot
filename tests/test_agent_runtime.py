from __future__ import annotations

from collections import Counter

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from middot.agent_runtime.location_graph import (
    LocationResolutionRuntime,
    build_location_graph,
)
from middot.agent_runtime.runtime import RuntimeSettings, load_runtime_settings
from middot.agent_runtime.trace import (
    LangSmithTraceSink,
    NullTraceSink,
    sanitize_payload,
)

CANDIDATES = [
    {
        "id": "tsinghua-main",
        "label": "清华大学",
        "address": "北京市海淀区双清路30号",
        "lng": 116.326,
        "lat": 40.003,
    },
    {
        "id": "tsinghua-garden",
        "label": "清华园",
        "address": "北京市海淀区成府路",
        "lng": 116.333,
        "lat": 39.999,
    },
]


class FakeDependencies:
    def __init__(self, confidence: float = 0.40):
        self.calls = Counter()
        self.confidence = confidence
        self.commits = {}

    def resolve(self, state):
        self.calls["resolve"] += 1
        assert state["query"] == "清华"
        assert state["city"] == "北京"
        return CANDIDATES

    def select(self, state, candidates):
        self.calls["select"] += 1
        return {
            "candidate_id": candidates[0]["id"],
            "confidence": self.confidence,
            "reason": "semantic_match",
        }

    def commit(self, **payload):
        self.calls["commit_attempt"] += 1
        operation_id = payload["operation_id"]
        if operation_id not in self.commits:
            self.calls["commit_effect"] += 1
            self.commits[operation_id] = {
                "ok": True,
                "candidate_id": payload["candidate"]["id"],
            }
        return self.commits[operation_id]


def make_runtime(deps, saver=None):
    saver = saver or InMemorySaver()
    graph = build_location_graph(
        resolver=deps.resolve,
        selector=deps.select,
        committer=deps.commit,
        checkpointer=saver,
        trace_sink=NullTraceSink(),
    )
    return LocationResolutionRuntime(graph), saver


def test_default_runtime_is_legacy(monkeypatch):
    monkeypatch.delenv("MIDDOT_AGENT_RUNTIME", raising=False)
    monkeypatch.delenv("MIDDOT_AGENT_ORCHESTRATOR", raising=False)
    monkeypatch.delenv("MIDDOT_LANGSMITH_TRACING", raising=False)
    settings = load_runtime_settings()
    assert settings.agent_runtime == "legacy"
    assert settings.use_langgraph is False
    assert settings.use_langgraph_orchestrator is False
    assert settings.langsmith_tracing is False


def test_main_orchestrator_flag_is_independent(monkeypatch):
    monkeypatch.setenv("MIDDOT_AGENT_RUNTIME", "langgraph")
    monkeypatch.setenv("MIDDOT_AGENT_ORCHESTRATOR", "langgraph")
    settings = load_runtime_settings()
    assert settings.use_langgraph is True
    assert settings.use_langgraph_orchestrator is True


def test_uncertain_location_interrupts_then_resumes_without_requery():
    deps = FakeDependencies(confidence=0.40)
    runtime, saver = make_runtime(deps)
    waiting = runtime.start(
        thread_id="conversation-1",
        request_id="request-1",
        participant_id="me",
        query="清华",
        city="北京",
    )
    assert waiting.status == "waiting_user"
    assert waiting.interrupt_id
    assert [item["id"] for item in waiting.prompt["candidates"]] == [
        "tsinghua-main",
        "tsinghua-garden",
    ]
    assert deps.calls == Counter(resolve=1, select=1)

    # 模拟进程重建：新 graph 复用同一个持久化 checkpointer。
    resumed_runtime, _ = make_runtime(deps, saver=saver)
    completed = resumed_runtime.resume(
        thread_id="conversation-1",
        interrupt_id=waiting.interrupt_id,
        candidate_id="tsinghua-main",
    )
    assert completed.status == "completed"
    assert completed.result["selected_candidate_id"] == "tsinghua-main"
    assert completed.result["selection_source"] == "user"
    assert deps.calls["resolve"] == 1
    assert deps.calls["select"] == 1
    assert deps.calls["commit_effect"] == 1


def test_confident_ai_selection_skips_interrupt():
    deps = FakeDependencies(confidence=0.96)
    runtime, _ = make_runtime(deps)
    outcome = runtime.start(
        thread_id="conversation-auto",
        request_id="request-auto",
        participant_id="lisa",
        query="清华",
        city="北京",
    )
    assert outcome.status == "completed"
    assert outcome.result["selected_candidate_id"] == "tsinghua-main"
    assert outcome.result["selection_source"] == "auto"
    assert deps.calls["commit_effect"] == 1


def test_single_candidate_still_needs_selector_confidence():
    deps = FakeDependencies(confidence=0.20)

    def resolve_one(state):
        deps.calls["resolve"] += 1
        return CANDIDATES[:1]

    graph = build_location_graph(
        resolver=resolve_one,
        selector=deps.select,
        committer=deps.commit,
        checkpointer=InMemorySaver(),
    )
    outcome = LocationResolutionRuntime(graph).start(
        thread_id="conversation-one",
        request_id="request-one",
        participant_id="me",
        query="清华",
        city="北京",
    )
    assert outcome.status == "waiting_user"
    assert len(outcome.prompt["candidates"]) == 1
    assert deps.calls["commit_attempt"] == 0


def test_invalid_resume_never_commits_and_can_retry():
    deps = FakeDependencies(confidence=0.10)
    runtime, _ = make_runtime(deps)
    waiting = runtime.start(
        thread_id="conversation-invalid",
        request_id="request-invalid",
        participant_id="me",
        query="清华",
        city="北京",
    )
    rejected = runtime.resume(
        thread_id="conversation-invalid",
        interrupt_id=waiting.interrupt_id,
        candidate_id="forged",
    )
    assert rejected.status == "waiting_user"
    assert rejected.prompt["error"] == "invalid_candidate"
    assert deps.calls["commit_attempt"] == 0

    outcome = runtime.resume(
        thread_id="conversation-invalid",
        interrupt_id=rejected.interrupt_id,
        candidate_id="tsinghua-garden",
    )
    assert outcome.status == "completed"
    assert outcome.result["selected_candidate_id"] == "tsinghua-garden"
    assert deps.calls["commit_effect"] == 1


def test_commit_operation_id_is_stable_across_retries():
    deps = FakeDependencies(confidence=0.99)
    runtime, _ = make_runtime(deps)
    first = runtime.start(
        thread_id="conversation-stable",
        request_id="request-stable",
        participant_id="me",
        query="清华",
        city="北京",
    )
    operation_id = first.result["operation_id"]
    replay = deps.commit(
        operation_id=operation_id,
        request_id="request-stable",
        participant_id="me",
        candidate=CANDIDATES[0],
        selection_source="auto",
    )
    assert replay == first.result["commit_result"]
    assert deps.calls["commit_attempt"] == 2
    assert deps.calls["commit_effect"] == 1


def test_trace_payload_redacts_strings_by_default():
    safe = sanitize_payload(
        {"query": "清华", "candidate_count": 2}, include_content=False
    )
    assert safe["query"]["redacted"] is True
    assert safe["query"]["length"] == 2
    assert safe["candidate_count"] == 2
    assert "清华" not in repr(safe)


def test_stale_interrupt_is_rejected_before_resume():
    deps = FakeDependencies(confidence=0.10)
    runtime, _ = make_runtime(deps)
    runtime.start(
        thread_id="conversation-stale",
        request_id="request-stale",
        participant_id="me",
        query="清华",
        city="北京",
    )
    with pytest.raises(ValueError, match="stale or mismatched"):
        runtime.resume(
            thread_id="conversation-stale",
            interrupt_id="old-card",
            candidate_id="tsinghua-main",
        )
    assert deps.calls["commit_attempt"] == 0


def test_langsmith_failure_does_not_hide_business_exception(monkeypatch):
    import langsmith

    class FakeRun:
        def end(self, outputs):
            self.outputs = outputs

    class FakeManager:
        def __init__(self):
            self.run = FakeRun()

        def __enter__(self):
            return self.run

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(langsmith, "trace", lambda *args, **kwargs: FakeManager())
    sink = LangSmithTraceSink(
        RuntimeSettings(langsmith_tracing=True, langsmith_sample_rate=1.0)
    )
    with (
        pytest.raises(RuntimeError, match="business failed"),
        sink.span("test", inputs={"query": "清华"}),
    ):
        raise RuntimeError("business failed")
