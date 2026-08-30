from __future__ import annotations

import pytest

from evals.harness import evaluate_scenario, load_scenarios


SCENARIOS = load_scenarios()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[item["id"] for item in SCENARIOS])
def test_agent_orchestration_scenario(scenario):
    result = evaluate_scenario(scenario)
    failures = [check for check in result.checks if not check["passed"]]
    assert result.passed, failures

