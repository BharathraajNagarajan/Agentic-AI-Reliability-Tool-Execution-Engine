"""Tests for the retry/idempotency execution wrapper (src.execution.executor).

Covers: a successful first execution (for each of the three tool
dispatches), a retried-then-successful execution via the injectable
failure predicate, exhausting retries returning a FAILURE result rather
than raising, idempotent re-execution not reapplying an already-applied
command, verification wiring (both the passing case and a forced failing
case via a deliberately "buggy" tool-function stub), and idempotency
holding for a VERIFICATION_FAILED result too.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.execution import executor as executor_module
from src.execution.executor import ExecutionWrapper
from src.graph.state import ExecutionStatus
from src.tools.schemas import (
    AddTicketNoteArgs,
    AuthorizedAddTicketNoteCommand,
    AuthorizedIssueRefundCommand,
    AuthorizedUpdateTicketStatusCommand,
    IssueRefundArgs,
    Ticket,
    TicketStatus,
    UpdateTicketStatusArgs,
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


class TestSuccessfulFirstExecution:
    def test_update_ticket_status(self):
        wrapper = ExecutionWrapper()
        ticket = make_ticket(status=TicketStatus.OPEN)
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.IN_PROGRESS, reason="starting work")
        )
        result = wrapper.execute(command, ticket)
        assert result.status is ExecutionStatus.SUCCESS
        assert result.attempt == 1
        assert result.output["ticket"]["status"] == "in_progress"

    def test_issue_refund(self):
        wrapper = ExecutionWrapper()
        ticket = make_ticket(refund_eligible_amount=Decimal("50.00"))
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("20.00"), currency="USD", reason="damaged item")
        )
        result = wrapper.execute(command, ticket)
        assert result.status is ExecutionStatus.SUCCESS
        assert result.attempt == 1
        assert result.output["ticket"]["refund_eligible_amount"] == "30.00"

    def test_add_ticket_note(self):
        wrapper = ExecutionWrapper()
        ticket = make_ticket()
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Called customer."))
        result = wrapper.execute(command, ticket)
        assert result.status is ExecutionStatus.SUCCESS
        assert result.attempt == 1
        assert result.output["ticket"]["notes"] == ["Called customer."]


class TestRetryBehavior:
    def test_retried_then_successful_execution(self):
        wrapper = ExecutionWrapper()
        ticket = make_ticket()
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Called customer."))

        attempts_seen: list[int] = []

        def fail_first_two_attempts(attempt: int) -> bool:
            attempts_seen.append(attempt)
            return attempt < 3

        result = wrapper.execute(command, ticket, max_attempts=5, should_fail=fail_first_two_attempts)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.attempt == 3
        assert attempts_seen == [1, 2, 3]
        assert result.output["ticket"]["notes"] == ["Called customer."]

    def test_exhausting_retries_returns_failure_not_exception(self):
        wrapper = ExecutionWrapper()
        ticket = make_ticket()
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Called customer."))

        result = wrapper.execute(command, ticket, max_attempts=3, should_fail=lambda attempt: True)

        assert result.status is ExecutionStatus.FAILURE
        assert result.attempt == 3
        assert result.error is not None
        assert result.output is None

    def test_failed_execution_is_not_recorded_as_idempotent(self):
        """A failed run must not block a later, successful retry of the same command."""
        wrapper = ExecutionWrapper()
        ticket = make_ticket()
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Called customer."))

        failed = wrapper.execute(command, ticket, max_attempts=2, should_fail=lambda attempt: True)
        assert failed.status is ExecutionStatus.FAILURE

        succeeded = wrapper.execute(command, ticket, max_attempts=2, should_fail=lambda attempt: False)
        assert succeeded.status is ExecutionStatus.SUCCESS
        assert succeeded.output["ticket"]["notes"] == ["Called customer."]


class TestIdempotency:
    def test_idempotent_reexecution_does_not_reapply_command(self, monkeypatch):
        ticket = make_ticket()
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Called customer."))

        call_count = {"n": 0}
        original_fn = executor_module._TOOL_FUNCTIONS[AuthorizedAddTicketNoteCommand]

        def counting_add_note(cmd, tkt):
            call_count["n"] += 1
            return original_fn(cmd, tkt)

        monkeypatch.setitem(executor_module._TOOL_FUNCTIONS, AuthorizedAddTicketNoteCommand, counting_add_note)

        wrapper = ExecutionWrapper()
        first_result = wrapper.execute(command, ticket)
        second_result = wrapper.execute(command, ticket)

        assert call_count["n"] == 1
        assert second_result == first_result
        assert first_result.output["ticket"]["notes"] == ["Called customer."]

    def test_idempotency_key_distinguishes_different_commands(self):
        wrapper = ExecutionWrapper()
        ticket = make_ticket()
        note_a = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Note A."))
        note_b = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Note B."))

        result_a = wrapper.execute(note_a, ticket)
        result_b = wrapper.execute(note_b, ticket)

        assert result_a.output["ticket"]["notes"] == ["Note A."]
        assert result_b.output["ticket"]["notes"] == ["Note B."]
        assert len(wrapper.idempotency_store) == 2

    def test_max_attempts_below_one_raises(self):
        wrapper = ExecutionWrapper()
        ticket = make_ticket()
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Note."))
        with pytest.raises(ValueError):
            wrapper.execute(command, ticket, max_attempts=0)


class TestVerificationWiring:
    def test_successful_execution_is_verified(self):
        """Normal case: the tool function's effect matches the command, so verification passes."""
        wrapper = ExecutionWrapper()
        ticket = make_ticket(status=TicketStatus.OPEN)
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.IN_PROGRESS, reason="starting work")
        )
        result = wrapper.execute(command, ticket)
        assert result.status is ExecutionStatus.SUCCESS
        assert result.verified is True
        assert result.output["ticket"]["status"] == "in_progress"

    def test_execution_with_failed_verification_returns_verification_failed(self, monkeypatch):
        """Force execution to succeed but verification to fail via a deliberately buggy tool-function stub.

        The stub simulates a tool function that runs without error but does
        NOT actually apply the authorized effect (status stays unchanged) —
        exactly the discrepancy verify() exists to catch.
        """
        ticket = make_ticket(status=TicketStatus.OPEN)
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.IN_PROGRESS, reason="starting work")
        )

        def buggy_update_ticket_status(cmd, tkt):
            return tkt.model_copy()  # status left untouched: does not reflect cmd.args.new_status

        monkeypatch.setitem(
            executor_module._TOOL_FUNCTIONS, AuthorizedUpdateTicketStatusCommand, buggy_update_ticket_status
        )

        wrapper = ExecutionWrapper()
        result = wrapper.execute(command, ticket, max_attempts=5, should_fail=lambda attempt: False)

        assert result.status is ExecutionStatus.VERIFICATION_FAILED
        assert result.verified is False
        assert result.error is not None
        assert "in_progress" in result.error
        assert result.attempt == 1  # no retry was attempted after the (non-transient) verification failure

    def test_verification_failed_result_is_idempotent(self, monkeypatch):
        """Re-executing a command whose prior run was VERIFICATION_FAILED must not re-run the tool function."""
        ticket = make_ticket(status=TicketStatus.OPEN)
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.IN_PROGRESS, reason="starting work")
        )

        call_count = {"n": 0}

        def buggy_update_ticket_status(cmd, tkt):
            call_count["n"] += 1
            return tkt.model_copy()  # status left untouched, forcing VERIFICATION_FAILED every time it runs

        monkeypatch.setitem(
            executor_module._TOOL_FUNCTIONS, AuthorizedUpdateTicketStatusCommand, buggy_update_ticket_status
        )

        wrapper = ExecutionWrapper()
        first_result = wrapper.execute(command, ticket)
        second_result = wrapper.execute(command, ticket)

        assert call_count["n"] == 1
        assert first_result.status is ExecutionStatus.VERIFICATION_FAILED
        assert second_result == first_result
