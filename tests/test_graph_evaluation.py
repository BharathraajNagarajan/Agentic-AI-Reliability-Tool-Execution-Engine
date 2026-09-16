"""Tests for the minimal in-memory evaluation harness (src.graph.evaluation).

Covers the fixed default scenario set (happy path, policy-denied,
verification-failure, execution-failure-retries-exhausted) all reporting
PASS, plus a small custom scenario set with one expected-pass and one
deliberately-mismatched (expected-fail) scenario, confirming the harness
correctly distinguishes the two.
"""

from datetime import datetime, timezone
from decimal import Decimal

from src.execution.schemas import ExecutionStatus
from src.graph.evaluation import EvalScenario, build_default_scenarios, format_report, run_scenarios
from src.graph.state import RunStatus
from src.observability.tracer import EventType
from src.tools.schemas import AuthorizedUpdateTicketStatusCommand, ProposedAction, Ticket, TicketStatus


def make_ticket(**overrides) -> Ticket:
    defaults = dict(
        id="T-1",
        status=TicketStatus.OPEN,
        customer_ref="cust-123",
        subject="Package arrived damaged",
        order_ref="ORD-456",
        refund_eligible_amount=Decimal("50.00"),
        currency="USD",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return Ticket(**defaults)


class TestDefaultScenarioSet:
    def test_includes_the_four_required_scenarios(self):
        scenarios = build_default_scenarios()
        names = {s.name for s in scenarios}
        assert len(scenarios) == 4
        assert any("happy" in n or "completed" in n for n in names)
        assert any("denied" in n for n in names)
        assert any("verification" in n for n in names)
        assert any("execution-failure" in n for n in names)

    def test_all_default_scenarios_pass(self):
        results = run_scenarios(build_default_scenarios())
        assert all(r.passed for r in results), format_report(results)

    def test_happy_path_scenario_reports_completed(self):
        results = run_scenarios(build_default_scenarios())
        happy = next(r for r in results if "happy" in r.scenario_name or "completed" in r.scenario_name)
        assert happy.passed is True
        assert happy.actual_run_status is RunStatus.COMPLETED
        assert happy.actual_execution_status is ExecutionStatus.SUCCESS

    def test_policy_denied_scenario_reports_denied_with_correct_rule(self):
        results = run_scenarios(build_default_scenarios())
        denied = next(r for r in results if "denied" in r.scenario_name)
        assert denied.passed is True
        assert denied.actual_run_status is RunStatus.DENIED
        assert denied.actual_rule_id == "refund-amount-limit"

    def test_verification_failure_scenario_reports_failed_with_verification_failed_status(self):
        """Closes the gap flagged in the prior phase: VERIFICATION_FAILED was never exercised with tracing on."""
        results = run_scenarios(build_default_scenarios())
        verification_failed = next(r for r in results if "verification" in r.scenario_name)

        assert verification_failed.passed is True
        assert verification_failed.actual_run_status is RunStatus.FAILED
        assert verification_failed.actual_execution_status is ExecutionStatus.VERIFICATION_FAILED

        executed_events = [e for e in verification_failed.events if e.event_type is EventType.EXECUTED]
        assert len(executed_events) == 1
        assert executed_events[0].execution_result.status is ExecutionStatus.VERIFICATION_FAILED

    def test_verification_failure_scenario_traced_event_sequence(self):
        results = run_scenarios(build_default_scenarios())
        verification_failed = next(r for r in results if "verification" in r.scenario_name)
        assert [e.event_type for e in verification_failed.events] == [
            EventType.PROPOSED,
            EventType.AUTHORIZED,
            EventType.EXECUTED,
            EventType.RUN_FINISHED,
        ]

    def test_execution_failure_scenario_reports_failed_with_failure_status(self):
        """Retries exhausted via should_fail injection: expected RunStatus.FAILED / ExecutionStatus.FAILURE."""
        results = run_scenarios(build_default_scenarios())
        execution_failed = next(r for r in results if "execution-failure" in r.scenario_name)

        assert execution_failed.passed is True
        assert execution_failed.actual_run_status is RunStatus.FAILED
        assert execution_failed.actual_execution_status is ExecutionStatus.FAILURE

        executed_events = [e for e in execution_failed.events if e.event_type is EventType.EXECUTED]
        assert len(executed_events) == 1
        assert executed_events[0].execution_result.status is ExecutionStatus.FAILURE

    def test_execution_failure_scenario_traced_event_sequence(self):
        results = run_scenarios(build_default_scenarios())
        execution_failed = next(r for r in results if "execution-failure" in r.scenario_name)
        assert [e.event_type for e in execution_failed.events] == [
            EventType.PROPOSED,
            EventType.AUTHORIZED,
            EventType.EXECUTED,
            EventType.RUN_FINISHED,
        ]

    def test_tool_dispatch_table_is_restored_after_verification_failure_scenario(self):
        """The verification-failure scenario's tool_function_overrides must not leak into later runs."""
        from src.execution import executor as executor_module
        from src.tools.functions import update_ticket_status

        before = executor_module._TOOL_FUNCTIONS[AuthorizedUpdateTicketStatusCommand]
        run_scenarios(build_default_scenarios())
        after = executor_module._TOOL_FUNCTIONS[AuthorizedUpdateTicketStatusCommand]
        assert after is update_ticket_status is before


class TestHarnessReportsPassAndFailCorrectly:
    def test_expected_pass_scenario_reports_pass(self):
        scenario = EvalScenario(
            name="expect-pass",
            ticket=make_ticket(),
            task_description="Add a note.",
            proposals=[ProposedAction(tool_name="add_ticket_note", arguments={"ticket_id": "T-1", "note": "hi"})],
            expected_run_status=RunStatus.COMPLETED,
        )
        results = run_scenarios([scenario])
        assert results[0].passed is True
        assert results[0].scenario_name == "expect-pass"

    def test_deliberately_mismatched_scenario_reports_fail(self):
        """Script a proposal that will actually be DENIED, but assert we expected COMPLETED."""
        scenario = EvalScenario(
            name="expect-fail",
            ticket=make_ticket(refund_eligible_amount=Decimal("50.00")),
            task_description="Refund way too much.",
            proposals=[
                ProposedAction(
                    tool_name="issue_refund",
                    arguments={"ticket_id": "T-1", "amount": "999.00", "currency": "USD", "reason": "x"},
                )
            ],
            expected_run_status=RunStatus.COMPLETED,  # deliberately wrong: this will actually be DENIED
        )
        results = run_scenarios([scenario])
        result = results[0]
        assert result.passed is False
        assert result.expected_run_status is RunStatus.COMPLETED
        assert result.actual_run_status is RunStatus.DENIED

    def test_mixed_scenario_set_reports_each_independently(self):
        pass_scenario = EvalScenario(
            name="expect-pass",
            ticket=make_ticket(id="T-pass"),
            task_description="Add a note.",
            proposals=[ProposedAction(tool_name="add_ticket_note", arguments={"ticket_id": "T-pass", "note": "hi"})],
            expected_run_status=RunStatus.COMPLETED,
        )
        fail_scenario = EvalScenario(
            name="expect-fail",
            ticket=make_ticket(id="T-fail", refund_eligible_amount=Decimal("50.00")),
            task_description="Refund way too much.",
            proposals=[
                ProposedAction(
                    tool_name="issue_refund",
                    arguments={"ticket_id": "T-fail", "amount": "999.00", "currency": "USD", "reason": "x"},
                )
            ],
            expected_run_status=RunStatus.COMPLETED,
        )

        results = run_scenarios([pass_scenario, fail_scenario])

        assert results[0].scenario_name == "expect-pass"
        assert results[0].passed is True
        assert results[1].scenario_name == "expect-fail"
        assert results[1].passed is False

    def test_mismatched_expected_rule_id_fails_even_if_run_status_matches(self):
        """A right RunStatus for the wrong reason (wrong rule_id) must still be reported as a failure."""
        scenario = EvalScenario(
            name="wrong-rule-id",
            ticket=make_ticket(refund_eligible_amount=Decimal("50.00")),
            task_description="Refund way too much.",
            proposals=[
                ProposedAction(
                    tool_name="issue_refund",
                    arguments={"ticket_id": "T-1", "amount": "999.00", "currency": "USD", "reason": "x"},
                )
            ],
            expected_run_status=RunStatus.DENIED,
            expected_rule_id="some-other-rule-that-did-not-fire",
        )
        results = run_scenarios([scenario])
        assert results[0].actual_run_status is RunStatus.DENIED
        assert results[0].passed is False
        assert results[0].actual_rule_id == "refund-amount-limit"


class TestFormatReport:
    def test_report_includes_events_only_for_failing_scenarios(self):
        pass_scenario = EvalScenario(
            name="expect-pass",
            ticket=make_ticket(),
            task_description="Add a note.",
            proposals=[ProposedAction(tool_name="add_ticket_note", arguments={"ticket_id": "T-1", "note": "hi"})],
            expected_run_status=RunStatus.COMPLETED,
        )
        fail_scenario = EvalScenario(
            name="expect-fail",
            ticket=make_ticket(refund_eligible_amount=Decimal("50.00")),
            task_description="Refund way too much.",
            proposals=[
                ProposedAction(
                    tool_name="issue_refund",
                    arguments={"ticket_id": "T-1", "amount": "999.00", "currency": "USD", "reason": "x"},
                )
            ],
            expected_run_status=RunStatus.COMPLETED,
        )
        results = run_scenarios([pass_scenario, fail_scenario])
        report = format_report(results)

        assert "[PASS] expect-pass" in report
        assert "[FAIL] expect-fail" in report
        assert "traced events" in report
        pass_section, fail_section = report.split("[FAIL]")
        assert "traced events" not in pass_section
