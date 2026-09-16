"""Tests for the in-memory structured trace recorder (src.observability.tracer).

Covers recording a full event sequence for one run, retrieving events by
run_id, and confirming the recorded events accurately distinguish a
denied path from a completed path.
"""

from datetime import datetime, timezone
from decimal import Decimal

from src.execution.schemas import ExecutionResult, ExecutionStatus
from src.graph.state import RunStatus
from src.observability.tracer import EventType, Tracer
from src.policy.schemas import AuthorizationContext, AuthorizationDecision, AuthorizationResult
from src.policy.interface import authorize
from src.policy.rules import refund_amount_rule, status_transition_rule
from src.tools.schemas import (
    AuthorizedIssueRefundCommand,
    IssueRefundArgs,
    ProposedAction,
    Ticket,
    TicketStatus,
)


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


class TestRecordingAFullSequence:
    def test_full_sequence_for_one_run(self):
        tracer = Tracer()
        run_id = "run-1"

        proposal = ProposedAction(
            tool_name="issue_refund",
            arguments={"ticket_id": "T-1", "amount": "20.00", "currency": "USD", "reason": "damaged item"},
        )
        tracer.record_proposal(run_id, proposal)

        authorization_result = AuthorizationResult(
            decision=AuthorizationDecision.APPROVED,
            authorized_command=AuthorizedIssueRefundCommand(
                args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("20.00"), currency="USD", reason="damaged item")
            ),
            reason="All policy rules passed and arguments validated for the requested tool.",
        )
        tracer.record_authorization(run_id, authorization_result)

        execution_result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            output={"ticket": {"id": "T-1"}},
            verified=True,
            attempt=1,
        )
        tracer.record_execution(run_id, execution_result)

        tracer.record_run_finished(run_id, RunStatus.COMPLETED)

        events = tracer.events_for_run(run_id)
        assert [e.event_type for e in events] == [
            EventType.PROPOSED,
            EventType.AUTHORIZED,
            EventType.EXECUTED,
            EventType.RUN_FINISHED,
        ]

    def test_sequence_numbers_are_ordered_and_zero_indexed(self):
        tracer = Tracer()
        run_id = "run-1"
        tracer.record_proposal(run_id, ProposedAction(tool_name="add_ticket_note", arguments={}))
        tracer.record_authorization(
            run_id, AuthorizationResult(decision=AuthorizationDecision.DENIED, reason="denied")
        )
        tracer.record_run_finished(run_id, RunStatus.DENIED)

        sequences = [e.sequence for e in tracer.events_for_run(run_id)]
        assert sequences == [0, 1, 2]

    def test_multiple_execution_attempts_are_each_recorded_distinctly(self):
        """A retried execution should be visible as multiple EXECUTED events with distinct attempt numbers."""
        tracer = Tracer()
        run_id = "run-1"
        tracer.record_execution(
            run_id, ExecutionResult(status=ExecutionStatus.FAILURE, error="transient", attempt=1)
        )
        tracer.record_execution(
            run_id, ExecutionResult(status=ExecutionStatus.SUCCESS, verified=True, attempt=2)
        )

        executed_events = [e for e in tracer.events_for_run(run_id) if e.event_type is EventType.EXECUTED]
        assert [e.execution_result.attempt for e in executed_events] == [1, 2]
        assert executed_events[0].execution_result.status is ExecutionStatus.FAILURE
        assert executed_events[1].execution_result.status is ExecutionStatus.SUCCESS


class TestRetrievalByRunId:
    def test_events_are_isolated_per_run_id(self):
        tracer = Tracer()
        tracer.record_proposal("run-a", ProposedAction(tool_name="add_ticket_note", arguments={}))
        tracer.record_proposal("run-b", ProposedAction(tool_name="issue_refund", arguments={}))
        tracer.record_proposal("run-b", ProposedAction(tool_name="issue_refund", arguments={}))

        assert len(tracer.events_for_run("run-a")) == 1
        assert len(tracer.events_for_run("run-b")) == 2

    def test_unknown_run_id_returns_empty_list_not_error(self):
        tracer = Tracer()
        assert tracer.events_for_run("never-seen") == []

    def test_run_ids_lists_all_runs_with_events(self):
        tracer = Tracer()
        tracer.record_proposal("run-a", ProposedAction(tool_name="add_ticket_note", arguments={}))
        tracer.record_proposal("run-b", ProposedAction(tool_name="issue_refund", arguments={}))

        assert set(tracer.run_ids()) == {"run-a", "run-b"}


class TestDeniedVsCompletedPathsReflectedAccurately:
    def test_denied_path_via_real_authorize_and_rules(self):
        """Use the real authorize() + refund_amount_rule to produce a genuine denial, then trace it."""
        tracer = Tracer()
        run_id = "run-denied"
        ticket = make_ticket(refund_eligible_amount=Decimal("50.00"))
        proposal = ProposedAction(
            tool_name="issue_refund",
            arguments={"ticket_id": "T-1", "amount": "999.00", "currency": "USD", "reason": "damaged item"},
        )

        tracer.record_proposal(run_id, proposal)
        result = authorize(
            proposed_action=proposal,
            ticket=ticket,
            context=AuthorizationContext(),
            rules=[refund_amount_rule, status_transition_rule],
        )
        tracer.record_authorization(run_id, result)
        tracer.record_run_finished(run_id, RunStatus.DENIED)

        events = tracer.events_for_run(run_id)
        auth_event = next(e for e in events if e.event_type is EventType.AUTHORIZED)
        finished_event = next(e for e in events if e.event_type is EventType.RUN_FINISHED)

        assert auth_event.authorization_result.decision is AuthorizationDecision.DENIED
        assert auth_event.authorization_result.rule_id == "refund-amount-limit"
        assert auth_event.authorization_result.authorized_command is None
        assert finished_event.run_status is RunStatus.DENIED
        assert not any(e.event_type is EventType.EXECUTED for e in events)

    def test_completed_path_via_real_authorize_and_rules(self):
        """Use the real authorize() with an in-limit refund to produce a genuine approval, then trace it."""
        tracer = Tracer()
        run_id = "run-completed"
        ticket = make_ticket(refund_eligible_amount=Decimal("50.00"))
        proposal = ProposedAction(
            tool_name="issue_refund",
            arguments={"ticket_id": "T-1", "amount": "20.00", "currency": "USD", "reason": "damaged item"},
        )

        tracer.record_proposal(run_id, proposal)
        result = authorize(
            proposed_action=proposal,
            ticket=ticket,
            context=AuthorizationContext(),
            rules=[refund_amount_rule, status_transition_rule],
        )
        tracer.record_authorization(run_id, result)
        tracer.record_execution(
            run_id, ExecutionResult(status=ExecutionStatus.SUCCESS, verified=True, attempt=1)
        )
        tracer.record_run_finished(run_id, RunStatus.COMPLETED)

        events = tracer.events_for_run(run_id)
        auth_event = next(e for e in events if e.event_type is EventType.AUTHORIZED)
        exec_event = next(e for e in events if e.event_type is EventType.EXECUTED)
        finished_event = next(e for e in events if e.event_type is EventType.RUN_FINISHED)

        assert auth_event.authorization_result.decision is AuthorizationDecision.APPROVED
        assert auth_event.authorization_result.authorized_command is not None
        assert exec_event.execution_result.status is ExecutionStatus.SUCCESS
        assert exec_event.execution_result.verified is True
        assert finished_event.run_status is RunStatus.COMPLETED
