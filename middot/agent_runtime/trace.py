"""可关闭、失败不影响主链路的 TraceSink。"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

from .runtime import RuntimeSettings

logger = logging.getLogger(__name__)


class Span(Protocol):
    def set_outputs(self, outputs: Mapping[str, Any]) -> None: ...


class TraceSink(Protocol):
    def span(
        self,
        name: str,
        *,
        inputs: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> AbstractContextManager[Span]: ...


class _NullSpan:
    def set_outputs(self, outputs: Mapping[str, Any]) -> None:
        del outputs


class NullTraceSink:
    @contextmanager
    def span(
        self,
        name: str,
        *,
        inputs: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[_NullSpan]:
        del name, inputs, metadata
        yield _NullSpan()


def _private_string(value: str) -> dict[str, Any]:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return {"redacted": True, "length": len(value), "sha256_12": digest}


def sanitize_payload(value: Any, *, include_content: bool) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if include_content else _private_string(value)
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_payload(item, include_content=include_content)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            sanitize_payload(item, include_content=include_content) for item in value
        ]
    return sanitize_payload(str(value), include_content=include_content)


class _LangSmithSpan:
    def __init__(self, run_tree: Any, include_content: bool):
        self._run_tree = run_tree
        self._include_content = include_content

    def set_outputs(self, outputs: Mapping[str, Any]) -> None:
        self._run_tree.end(
            outputs=sanitize_payload(outputs, include_content=self._include_content)
        )


class LangSmithTraceSink:
    """研发可观测性适配器；网络或 SDK 错误不得打断业务。"""

    def __init__(self, settings: RuntimeSettings):
        self._settings = settings

    def _sampled(self, metadata: Mapping[str, Any] | None) -> bool:
        request_id = str((metadata or {}).get("request_id", ""))
        if not request_id:
            return False
        bucket = int.from_bytes(
            hashlib.sha256(request_id.encode("utf-8")).digest()[:8], "big"
        ) / float(2**64)
        return bucket < self._settings.langsmith_sample_rate

    @contextmanager
    def span(
        self,
        name: str,
        *,
        inputs: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[Span]:
        # 同一 request_id 使用确定性采样，保证一次调用链不会只留下半截 span。
        if not self._sampled(metadata):
            yield _NullSpan()
            return
        try:
            from langsmith import trace

            manager = trace(
                name,
                run_type="chain",
                inputs=sanitize_payload(
                    inputs, include_content=self._settings.langsmith_include_content
                ),
                metadata=sanitize_payload(
                    metadata or {},
                    include_content=self._settings.langsmith_include_content,
                ),
                project_name=self._settings.langsmith_project,
                exceptions_to_handle=(Exception,),
            )
            run_tree = manager.__enter__()
        except Exception:
            logger.debug("LangSmith span setup failed", exc_info=True)
            yield _NullSpan()
            return

        try:
            yield _LangSmithSpan(run_tree, self._settings.langsmith_include_content)
        except BaseException as exc:
            try:
                manager.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                logger.debug("LangSmith span teardown failed", exc_info=True)
            raise
        else:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                # Trace 是旁路能力，任何配置、网络或 SDK 故障都不能影响 Agent。
                logger.debug("LangSmith span teardown failed", exc_info=True)


def build_trace_sink(settings: RuntimeSettings) -> TraceSink:
    if not settings.langsmith_tracing or settings.langsmith_sample_rate <= 0:
        return NullTraceSink()
    return LangSmithTraceSink(settings)
