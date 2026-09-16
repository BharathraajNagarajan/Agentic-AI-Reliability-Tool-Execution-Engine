"""Tests for running evaluation scenarios against the real AnthropicProposeProvider wiring.

All tests here use a plain duck-typed fake Anthropic client (same
technique as tests/test_providers_anthropic.py) — none of them make a
network call, require an API key, or even require `anthropic` to be
installed. `run_scenario`/`run_scenarios_with_real_provider` are exercised
only with an injected fake `client`, so `AnthropicProposeProvider`'s lazy
`import anthropic` branch is never reached.
"""

import json
import os
from types import SimpleNamespace

import pytest

from src.execution.schemas import ExecutionStatus
from src.graph.evaluation import (
    EvalScenario,
    RealProviderRunOutcome,
    build_default_scenarios,
    persist_results,
    run_default_scenarios_against_real_provider,
    run_scenario,
    run_scenarios_with_real_provider,
)
from src.graph.state import RunStatus
from src.tools.schemas import ProposedAction


class FakeMessages:
    def __init__(self, response) -> None:
        self._response = response
        self.create_calls: list[dict] = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self._response


class FakeAnthropicClient:
    def __init__(self, response) -> None:
        self.messages = FakeMessages(response)


def make_tool_use_response(tool_name: str, arguments: dict):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=tool_name, input=arguments)])


class TestRunScenarioWithPropseFnOverride:
    def test_default_propose_fn_still_uses_scripted_provider(self):
        """Backward compatibility: run_scenario() with no propose_fn behaves exactly as before."""
        scenario = build_default_scenarios()[0]
        result = run_scenario(scenario)
        assert result.passed is True
        assert result.actual_run_status is RunStatus.COMPLETED

    def test_explicit_propose_fn_overrides_scripted_proposals(self):
        """Passing a propose_fn means scenario.proposals is never consulted."""
        scenario = EvalScenario(
            name="override-test",
            ticket=build_default_scenarios()[0].ticket,
            task_description="Add a note.",
            proposals=[ProposedAction(tool_name="issue_refund", arguments={})],  # would fail if actually used
            expected_run_status=RunStatus.COMPLETED,
        )
        fake_client = FakeAnthropicClient(
            make_tool_use_response("add_ticket_note", {"ticket_id": scenario.ticket.id, "note": "Handled."})
        )
        from src.graph.providers.anthropic_provider import AnthropicProposeProvider

        provider = AnthropicProposeProvider(client=fake_client)
        result = run_scenario(scenario, propose_fn=provider)

        assert result.passed is True
        assert result.actual_run_status is RunStatus.COMPLETED


class TestRunScenariosWithRealProvider:
    def test_happy_path_scenario_passes_with_matching_fake_response(self):
        happy_path = build_default_scenarios()[0]
        fake_client = FakeAnthropicClient(
            make_tool_use_response(
                "issue_refund",
                {"ticket_id": happy_path.ticket.id, "amount": "20.00", "currency": "USD", "reason": "damaged item"},
            )
        )
        results = run_scenarios_with_real_provider([happy_path], client=fake_client)

        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].actual_run_status is RunStatus.COMPLETED
        assert results[0].actual_execution_status is ExecutionStatus.SUCCESS

    def test_mismatch_between_scripted_expectation_and_model_decision_is_reported_not_raised(self):
        """A 'model' that does something different from what the scenario expected is a reported mismatch, not a crash."""
        policy_denied_scenario = build_default_scenarios()[1]  # expects DENIED for an over-limit refund
        # Simulate the "model" instead adding a note - completely different tool than expected.
        fake_client = FakeAnthropicClient(
            make_tool_use_response(
                "add_ticket_note", {"ticket_id": policy_denied_scenario.ticket.id, "note": "Looked into it."}
            )
        )
        results = run_scenarios_with_real_provider([policy_denied_scenario], client=fake_client)

        assert len(results) == 1
        result = results[0]
        # The scenario expected DENIED (from an over-limit refund); the "model" did something else entirely,
        # so the actual outcome is COMPLETED, and the harness reports this as a plain, non-fatal mismatch.
        assert result.expected_run_status is RunStatus.DENIED
        assert result.actual_run_status is RunStatus.COMPLETED
        assert result.passed is False

    def test_each_scenario_gets_its_own_provider_and_client_is_reused(self):
        """The same injected fake client can be shared across scenarios without cross-contamination."""
        scenarios = build_default_scenarios()[:2]
        fake_client = FakeAnthropicClient(
            make_tool_use_response("add_ticket_note", {"ticket_id": "T-1", "note": "same response for all"})
        )
        results = run_scenarios_with_real_provider(scenarios, client=fake_client)
        assert len(results) == 2
        # Both scenarios got a call, proving the client (and its .messages.create) was invoked per scenario.
        assert len(fake_client.messages.create_calls) == 2

    def test_passes_model_argument_through_to_client(self):
        scenario = build_default_scenarios()[0]
        fake_client = FakeAnthropicClient(make_tool_use_response("add_ticket_note", {"ticket_id": "T-1", "note": "x"}))
        run_scenarios_with_real_provider([scenario], client=fake_client, model="claude-haiku-4-5")
        assert fake_client.messages.create_calls[0]["model"] == "claude-haiku-4-5"

    def test_never_touches_real_anthropic_sdk_when_client_injected(self):
        import sys

        was_already_imported = "anthropic" in sys.modules
        scenario = build_default_scenarios()[0]
        fake_client = FakeAnthropicClient(make_tool_use_response("add_ticket_note", {"ticket_id": "T-1", "note": "x"}))
        run_scenarios_with_real_provider([scenario], client=fake_client)
        if not was_already_imported:
            assert "anthropic" not in sys.modules


class TestPersistResults:
    def test_writes_a_json_file_with_expected_shape(self, tmp_path):
        scenario = build_default_scenarios()[0]
        fake_client = FakeAnthropicClient(
            make_tool_use_response(
                "issue_refund",
                {"ticket_id": "T-1", "amount": "20.00", "currency": "USD", "reason": "damaged item"},
            )
        )
        results = run_scenarios_with_real_provider([scenario], client=fake_client)

        output_path = persist_results(results, output_dir=tmp_path, model="claude-haiku-4-5")

        assert output_path.exists()
        assert output_path.parent == tmp_path
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["model"] == "claude-haiku-4-5"
        assert "timestamp" in payload
        assert len(payload["results"]) == 1
        assert payload["results"][0]["scenario_name"] == scenario.name
        assert payload["results"][0]["passed"] is True
        assert isinstance(payload["results"][0]["events"], list)
        assert len(payload["results"][0]["events"]) > 0

    def test_filename_is_windows_safe_and_timestamped(self, tmp_path):
        from datetime import datetime, timezone

        results = []
        fixed_time = datetime(2026, 9, 16, 7, 30, 0, tzinfo=timezone.utc)
        output_path = persist_results(results, output_dir=tmp_path, timestamp=fixed_time)

        assert output_path.name == "run-20260916T073000Z.json"
        assert ":" not in output_path.name

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested_dir = tmp_path / "does" / "not" / "exist" / "yet"
        output_path = persist_results([], output_dir=nested_dir)
        assert output_path.exists()
        assert nested_dir.exists()

    def test_events_are_json_serializable_including_decimal_and_enum_fields(self, tmp_path):
        """A regression guard: EvalResult.events holds pydantic models with Decimal/enum fields."""
        scenario = build_default_scenarios()[0]  # an issue_refund happy path, so events carry a Decimal amount
        fake_client = FakeAnthropicClient(
            make_tool_use_response(
                "issue_refund",
                {"ticket_id": "T-1", "amount": "20.00", "currency": "USD", "reason": "damaged item"},
            )
        )
        results = run_scenarios_with_real_provider([scenario], client=fake_client)
        output_path = persist_results(results, output_dir=tmp_path)
        # json.loads succeeding at all (no TypeError from write_text/json.dumps) is the real assertion.
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["results"][0]["events"]


class TestRunDefaultScenariosAgainstRealProvider:
    def test_skips_cleanly_when_api_key_is_not_set(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        outcome = run_default_scenarios_against_real_provider(output_dir=tmp_path)

        assert isinstance(outcome, RealProviderRunOutcome)
        assert outcome.skipped is True
        assert outcome.reason is not None
        assert "ANTHROPIC_API_KEY" in outcome.reason
        assert outcome.results == []
        assert outcome.output_path is None
        # Nothing should have been written.
        assert list(tmp_path.iterdir()) == []

    def test_does_not_import_anthropic_when_skipping(self, monkeypatch, tmp_path):
        import sys

        was_already_imported = "anthropic" in sys.modules
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        run_default_scenarios_against_real_provider(output_dir=tmp_path)
        if not was_already_imported:
            assert "anthropic" not in sys.modules

    def test_does_not_raise_when_key_missing(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        try:
            run_default_scenarios_against_real_provider(output_dir=tmp_path)
        except Exception as exc:  # pragma: no cover - failure path
            pytest.fail(f"run_default_scenarios_against_real_provider() raised unexpectedly: {exc}")
