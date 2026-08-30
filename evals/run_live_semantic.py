"""Evaluate Middot's real semantic parser without importing the production app.

The prompts are extracted from app_v2.py via AST so the runner exercises the exact
checked-in parser/verifier instructions while avoiding database or Redis side effects.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


CASES_PATH = Path(__file__).with_name("semantic_cases.jsonl")


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_prompt(source_path: Path, variable_name: str) -> str:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_meeting_utterance"
    )
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError(f"cannot find {variable_name} in _parse_meeting_utterance")


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def _subset_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _score(case: dict[str, Any], parsed: dict[str, Any]) -> list[dict[str, Any]]:
    expected = case["expect"]
    checks: list[dict[str, Any]] = []

    def add(name: str, wanted: Any, actual: Any) -> None:
        checks.append({"name": name, "passed": wanted == actual, "expected": wanted, "actual": actual})

    for key in ("intent", "search_keyword", "city_context"):
        if key in expected:
            add(key, expected[key], parsed.get(key, ""))
    if "search_keyword_any" in expected:
        actual_keyword = parsed.get("search_keyword", "")
        allowed = expected["search_keyword_any"]
        checks.append({
            "name": "search_keyword_any",
            "passed": actual_keyword in allowed,
            "expected": allowed,
            "actual": actual_keyword,
        })
    change = parsed.get("participant_change") or {}
    if "participant_mode" in expected:
        add("participant_mode", expected["participant_mode"], change.get("mode"))
    if "ordered_names" in expected:
        add("ordered_names", expected["ordered_names"], change.get("ordered_names") or [])
    if "slot_changes" in expected:
        actual_changes = change.get("slot_changes") or []
        for wanted in expected["slot_changes"]:
            matched = any(_subset_match(item, wanted) for item in actual_changes if isinstance(item, dict))
            checks.append({"name": f"slot_change:{wanted.get('index')}", "passed": matched, "expected": wanted, "actual": actual_changes})
    if "locations" in expected:
        actual_locations = parsed.get("locations") or []
        add("location_count", len(expected["locations"]), len(actual_locations))
        for wanted in expected["locations"]:
            candidates = [
                item for item in actual_locations
                if int(item.get("participant_index") or 0) == int(wanted.get("participant_index") or 0)
            ]
            matched = False
            for item in candidates:
                ordinary = {
                    key: value for key, value in wanted.items()
                    if key not in {"canonical_contains"}
                }
                if not _subset_match(item, ordinary):
                    continue
                canonical = wanted.get("canonical_contains")
                if canonical and canonical not in (item.get("canonical_candidates") or []):
                    continue
                matched = True
                break
            checks.append({"name": f"location:{wanted.get('participant_index')}", "passed": matched, "expected": wanted, "actual": candidates})
    for ignored in expected.get("ignored_contains") or []:
        actual_ignored = parsed.get("ignored_text") or []
        matched = any(ignored in str(item) for item in actual_ignored)
        checks.append({"name": f"ignored:{ignored}", "passed": matched, "expected": ignored, "actual": actual_ignored})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("app_v2.py"))
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is required for live semantic evaluation")
    parse_prompt = _extract_prompt(args.source, "system")
    verify_prompt = _extract_prompt(args.source, "verify_system")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    results = []
    for case in _load_cases(args.cases):
        request_payload = {
            "message": case["message"],
            "me_index": 1,
            "participants": [
                {"index": index + 1, **participant}
                for index, participant in enumerate(case.get("participants") or [])
            ],
        }
        started = time.perf_counter()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": parse_prompt}, {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)}],
            response_format={"type": "json_object"},
            temperature=0,
            stream=False,
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        if parsed.get("locations"):
            initial_search_keyword = str(parsed.get("search_keyword") or "").strip()
            verify_payload = {"message": case["message"], "parsed": parsed}
            verified = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": verify_prompt}, {"role": "user", "content": json.dumps(verify_payload, ensure_ascii=False)}],
                response_format={"type": "json_object"},
                temperature=0,
                stream=False,
            )
            verified_value = json.loads(verified.choices[0].message.content or "{}")
            # Match production exactly: some models echo {message, parsed}
            # instead of returning the requested full schema.  Such a wrapper
            # must not erase the valid first-pass parse.
            if isinstance(verified_value, dict) and isinstance(verified_value.get("locations"), list):
                parsed = verified_value
                if not str(parsed.get("search_keyword") or "").strip():
                    parsed["search_keyword"] = initial_search_keyword
        duration_ms = (time.perf_counter() - started) * 1000
        checks = _score(case, parsed)
        results.append({
            "id": case["id"],
            "category": case["category"],
            "passed": bool(checks) and all(check["passed"] for check in checks),
            "duration_ms": round(duration_ms, 1),
            "failed_checks": [check for check in checks if not check["passed"]],
        })
    durations = [item["duration_ms"] for item in results]
    passed = sum(item["passed"] for item in results)
    report = {
        "evaluation_mode": "live_deepseek_semantic_parser",
        "model": "deepseek-chat",
        "case_total": len(results),
        "case_passed": passed,
        "case_pass_rate": round(passed / len(results) * 100, 2),
        "duration_ms": {"p50": round(_percentile(durations, 0.5), 1), "p95": round(_percentile(durations, 0.95), 1)},
        "results": results,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Live semantic cases: {passed}/{len(results)} ({report['case_pass_rate']:.2f}%)")
        print(f"Latency: p50={report['duration_ms']['p50']}ms, p95={report['duration_ms']['p95']}ms")
        for item in results:
            if not item["passed"]:
                print(f"FAILED {item['id']}: {json.dumps(item['failed_checks'], ensure_ascii=False)}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
