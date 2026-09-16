"""Tests for the shared LangGraph state schema in src.graph.state.

Focus is on constructing valid instances of the state object and its
component result models, and confirming the state's `proposed_action` /
`authorization_result` slots correctly preserve the ProposedAction vs.
AuthorizedCommand trust distinction from src.tools.schemas.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.graph.state import (
    AgentState,
    AuthorizationDecision,
    AuthorizationResult,
    ConversationMessage,
    ExecutionResult,
    ExecutionStatus,
    RunStatus,
)
from src.tools.schemas import (
    AuthorizedIssueRefundCommand,
    IssueRefundArgs,
    ProposedAction,
    Ticket,
    TicketStatus,
)


def make_ticket() -> Ticket:
    return Ticket(
        id="T-1",
        status=TicketStatus.OPEN,
        customer_ref="cust-123",
        subject="Package arrived damaged",
        order_ref="ORD-456",
        refund_eligible_amount=Decimal("49.99"),
        currency="USD",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )


class TestConversationMessage:
    def test_valid_message(self):
        msg = ConversationMessage(
            role="user", content="My order arrived broken.", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        assert msg.role == "user"

    def test_rejects_unknown_role(self):
        with pytest.raises(ValidationError):
            ConversationMessage(
                role="narrator", content="...", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc)
            )


class TestAuthorizationResult:
    def test_denied_result_needs_no_authorized_command(self):
        result = AuthorizationResult(decision=AuthorizationDecision.DENIED, reason="Refund exceeds policy limit")
        assert result.authorized_command is None
        assert result.decision is AuthorizationDecision.DENIED

    def test_approved_result_carries_authorized_command(self):
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("10.00"), currency="USD", reason="damaged item")
        )
        result = AuthorizationResult(
            decision=AuthorizationDecision.APPROVED,
            authorized_command=command,
            reason="Within refund policy limit",
            rule_id="refund-under-50",
        )
        assert result.authorized_command.args.amount == Decimal("10.00")


class TestExecutionResult:
    def test_valid_success_result(self):
        result = ExecutionResult(status=ExecutionStatus.SUCCESS, output={"refund_id": "R-1"}, verified=True)
        assert result.attempt == 1
        assert result.verified is True

    def test_valid_failure_result(self):
        result = ExecutionResult(status=ExecutionStatus.FAILURE, error="downstream timeout", attempt=2)
        assert result.error == "downstream timeout"

    def test_rejects_non_positive_attempt(self):
        with pytest.raises(ValidationError):
            ExecutionResult(status=ExecutionStatus.SUCCESS, attempt=0)


class TestAgentState:
    def test_minimal_valid_state_defaults(self):
        state = AgentState(
            run_id="run-1",
            ticket=make_ticket(),
            task_description="Customer wants a refund for a damaged item.",
        )
        assert state.run_status is RunStatus.PENDING
        assert state.retry_count == 0
        assert state.proposed_action is None
        assert state.authorization_result is None
        assert state.execution_result is None
        assert state.conversation_history == []

    def test_full_state_through_one_loop_iteration(self):
        proposal = ProposedAction(
            tool_name="issue_refund",
            arguments={"ticket_id": "T-1", "amount": "10.00", "currency": "USD", "reason": "damaged item"},
        )
        authorized = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("10.00"), currency="USD", reason="damaged item")
        )
        state = AgentState(
            run_id="run-2",
            ticket=make_ticket(),
            task_description="Customer wants a refund for a damaged item.",
            conversation_history=[
                ConversationMessage(
                    role="user",
                    content="My order arrived broken, please refund me.",
                    timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                )
            ],
            proposed_action=proposal,
            authorization_result=AuthorizationResult(
                decision=AuthorizationDecision.APPROVED,
                authorized_command=authorized,
                reason="Within refund policy limit",
            ),
            execution_result=ExecutionResult(status=ExecutionStatus.SUCCESS, verified=True),
            retry_count=0,
            run_status=RunStatus.COMPLETED,
        )
        assert state.run_status is RunStatus.COMPLETED
        assert state.proposed_action.tool_name == "issue_refund"
        assert state.authorization_result.authorized_command.args.amount == Decimal("10.00")

    def test_rejects_missing_required_ticket(self):
        with pytest.raises(ValidationError):
            AgentState(run_id="run-3", task_description="No ticket provided")

    def test_rejects_negative_retry_count(self):
        with pytest.raises(ValidationError):
            AgentState(
                run_id="run-4",
                ticket=make_ticket(),
                task_description="Should not accept negative retries",
                retry_count=-1,
            )
