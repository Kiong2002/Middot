"""Feature flag 和运行时选择。

默认值必须保持 legacy，避免安装依赖后意外切换生产链路。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSettings:
    agent_runtime: str = "legacy"
    langsmith_tracing: bool = False
    langsmith_project: str = "middot-staging"
    langsmith_sample_rate: float = 0.0
    langsmith_include_content: bool = False

    @property
    def use_langgraph(self) -> bool:
        return self.agent_runtime == "langgraph"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _sample_rate(raw: str | None) -> float:
    try:
        value = float(raw or "0")
    except ValueError:
        return 0.0
    return min(1.0, max(0.0, value))


def load_runtime_settings() -> RuntimeSettings:
    runtime = os.getenv("MIDDOT_AGENT_RUNTIME", "legacy").strip().lower()
    if runtime not in {"legacy", "langgraph"}:
        runtime = "legacy"
    return RuntimeSettings(
        agent_runtime=runtime,
        langsmith_tracing=_env_bool("MIDDOT_LANGSMITH_TRACING"),
        langsmith_project=os.getenv(
            "MIDDOT_LANGSMITH_PROJECT", "middot-staging"
        ).strip()
        or "middot-staging",
        langsmith_sample_rate=_sample_rate(os.getenv("MIDDOT_LANGSMITH_SAMPLE_RATE")),
        langsmith_include_content=_env_bool("MIDDOT_LANGSMITH_INCLUDE_CONTENT"),
    )
