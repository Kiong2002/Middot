from __future__ import annotations

import argparse
import json

from evals.harness import evaluate_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Middot deterministic Agent evaluations")
    parser.add_argument("--json", action="store_true", help="print the complete JSON report")
    args = parser.parse_args()
    report = evaluate_all()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"Agent orchestration scenarios: {report['scenario_passed']}/{report['scenario_total']} "
            f"({report['scenario_pass_rate']:.2f}%)"
        )
        print(
            f"Assertions: {report['assertion_passed']}/{report['assertion_total']} "
            f"({report['assertion_pass_rate']:.2f}%)"
        )
        print(
            f"Runtime: p50={report['duration_ms']['p50']:.3f}ms, "
            f"p95={report['duration_ms']['p95']:.3f}ms"
        )
        for category, value in report["categories"].items():
            print(f"- {category}: {value['passed']}/{value['total']} ({value['pass_rate']:.2f}%)")
        failures = [item for item in report["results"] if not item["passed"]]
        for failure in failures:
            print(f"FAILED {failure['id']}: {failure['failed_checks']}")
    return 0 if report["scenario_passed"] == report["scenario_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

