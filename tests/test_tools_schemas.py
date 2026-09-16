"""Tests for the ticket and tool-call data contracts in src.tools.schemas.

These are pure data-contract tests: they construct valid instances of each
model, and verify the trust-boundary property that `ProposedAction` (the
untrusted LLM output shape) tolerates malformed/incomplete data while
`AuthorizedCommand` (the trusted, validated shape) rejects it.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.tools.schemas import (
    AddTicketNoteArgs,
    AuthorizedAddTicketNoteCommand,
    AuthorizedCommandAdapter,
    AuthorizedIssueRefundCommand,
    AuthorizedUpdateTicketStatusCommand,
    IssueRefundArgs,
    ProposedAction,
    Ticket,
    TicketPriority,
    TicketStatus,
    ToolName,
    UpdateTicketStatusArgs,
)


def make_valid_ticket(**overrides) -> Ticket:
    defaults = dict(
        id="T-1",
        status=TicketStatus.OPEN,
        priority=TicketPriority.NORMAL,
        customer_ref="cust-123",
        subject="Package arrived damaged",
        order_ref="ORD-456",
        refund_eligible_amount=Decimal("49.99"),
        currency="USD",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return Ticket(**defaults)


class TestTicket:
    def test_valid_ticket_constructs(self):
        ticket = make_valid_ticket()
        assert ticket.id == "T-1"
        assert ticket.status is TicketStatus.OPEN
        assert ticket.refund_eligible_amount == Decimal("49.99")

    def test_minimal_ticket_without_refund_fields(self):
        ticket = Ticket(
            id="T-2",
            status=TicketStatus.CLOSED,
            customer_ref="cust-999",
            subject="General question",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        assert ticket.order_ref is None
        assert ticket.refund_eligible_amount is None
        assert ticket.priority is TicketPriority.NORMAL  # default

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            Ticket(
                status=TicketStatus.OPEN,
                customer_ref="cust-123",
                subject="Missing id",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            Ticket(
                id="T-3",
                status="not_a_real_status",
                customer_ref="cust-123",
                subject="Bad status",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )


class TestProposedAction:
    def test_valid_proposed_action_constructs(self):
        action = ProposedAction(
            tool_name="issue_refund",
            arguments={"ticket_id": "T-1", "amount": "49.99", "currency": "USD", "reason": "damaged item"},
        )
        assert action.tool_name == "issue_refund"
        assert action.arguments["amount"] == "49.99"

    def test_accepts_unknown_tool_name(self):
        """The LLM can hallucinate a tool name that doesn't exist; this must not raise."""
        action = ProposedAction(tool_name="delete_all_tickets_forever", arguments={})
        assert action.tool_name == "delete_all_tickets_forever"

    def test_accepts_missing_arguments(self):
        """Arguments are optional/defaulted since the LLM may omit them entirely."""
        action = ProposedAction(tool_name="issue_refund")
        assert action.arguments == {}

    def test_accepts_malformed_argument_types(self):
        """Wrong types, missing keys, and nonsense values must all be tolerated."""
        action = ProposedAction(
            tool_name="issue_refund",
            arguments={"amount": {"nested": "not a number"}, "unexpected_key": 12345, "ticket_id": None},
        )
        assert action.arguments["amount"] == {"nested": "not a number"}
        assert action.arguments["ticket_id"] is None

    def test_accepts_extra_top_level_fields(self):
        """Extra fields outside the known schema (e.g. LLM chatter) are allowed, not rejected."""
        action = ProposedAction(
            tool_name="issue_refund",
            arguments={},
            confidence="high",
            thoughts="I believe this ticket qualifies for a refund",
        )
        assert action.model_extra["confidence"] == "high"

    def test_accepts_completely_empty_arguments_dict_with_only_tool_name(self):
        action = ProposedAction.model_validate({"tool_name": "totally_made_up"})
        assert action.arguments == {}

    def test_rejects_missing_tool_name(self):
        """Even the untrusted shape needs to know what tool was (allegedly) requested."""
        with pytest.raises(ValidationError):
            ProposedAction(arguments={"foo": "bar"})


class TestAuthorizedCommandArgs:
    def test_valid_update_ticket_status_args(self):
        args = UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.RESOLVED, reason="Issue fixed")
        assert args.new_status is TicketStatus.RESOLVED

    def test_valid_issue_refund_args(self):
        args = IssueRefundArgs(ticket_id="T-1", amount=Decimal("10.00"), currency="USD", reason="damaged item")
        assert args.amount == Decimal("10.00")

    def test_issue_refund_rejects_non_positive_amount(self):
        with pytest.raises(ValidationError):
            IssueRefundArgs(ticket_id="T-1", amount=Decimal("0"), currency="USD", reason="damaged item")

    def test_issue_refund_rejects_negative_amount(self):
        with pytest.raises(ValidationError):
            IssueRefundArgs(ticket_id="T-1", amount=Decimal("-5"), currency="USD", reason="damaged item")

    def test_issue_refund_rejects_bad_currency_length(self):
        with pytest.raises(ValidationError):
            IssueRefundArgs(ticket_id="T-1", amount=Decimal("10"), currency="US", reason="damaged item")

    def test_valid_add_ticket_note_args(self):
        args = AddTicketNoteArgs(ticket_id="T-1", note="Customer confirmed receipt of refund.")
        assert args.note.startswith("Customer")

    def test_add_ticket_note_rejects_empty_note(self):
        with pytest.raises(ValidationError):
            AddTicketNoteArgs(ticket_id="T-1", note="")


class TestAuthorizedCommand:
    def test_valid_update_ticket_status_command_via_concrete_class(self):
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.ESCALATED, reason="Needs manager")
        )
        assert command.tool_name is ToolName.UPDATE_TICKET_STATUS

    def test_valid_issue_refund_command_via_discriminated_union_adapter(self):
        validated = AuthorizedCommandAdapter.validate_python(
            {
                "tool_name": "issue_refund",
                "args": {"ticket_id": "T-1", "amount": "25.00", "currency": "USD", "reason": "damaged item"},
            }
        )
        assert isinstance(validated, AuthorizedIssueRefundCommand)
        assert validated.args.amount == Decimal("25.00")

    def test_valid_add_ticket_note_command_via_discriminated_union_adapter(self):
        validated = AuthorizedCommandAdapter.validate_python(
            {"tool_name": "add_ticket_note", "args": {"ticket_id": "T-1", "note": "Called customer back."}}
        )
        assert isinstance(validated, AuthorizedAddTicketNoteCommand)

    def test_rejects_unknown_tool_name(self):
        """Unlike ProposedAction, an unrecognized tool name must be rejected outright."""
        with pytest.raises(ValidationError):
            AuthorizedCommandAdapter.validate_python({"tool_name": "delete_all_tickets_forever", "args": {}})

    def test_rejects_missing_arguments(self):
        with pytest.raises(ValidationError):
            AuthorizedCommandAdapter.validate_python({"tool_name": "issue_refund"})

    def test_rejects_malformed_argument_types(self):
        """The same malformed payload ProposedAction happily accepted must be rejected here."""
        with pytest.raises(ValidationError):
            AuthorizedCommandAdapter.validate_python(
                {
                    "tool_name": "issue_refund",
                    "args": {"amount": {"nested": "not a number"}, "unexpected_key": 12345, "ticket_id": None},
                }
            )

    def test_rejects_args_from_wrong_tool(self):
        """update_ticket_status args must not validate under the issue_refund tool name."""
        with pytest.raises(ValidationError):
            AuthorizedCommandAdapter.validate_python(
                {
                    "tool_name": "issue_refund",
                    "args": {"ticket_id": "T-1", "new_status": "resolved", "reason": "Issue fixed"},
                }
            )

    def test_rejects_non_positive_refund_amount_end_to_end(self):
        with pytest.raises(ValidationError):
            AuthorizedCommandAdapter.validate_python(
                {
                    "tool_name": "issue_refund",
                    "args": {"ticket_id": "T-1", "amount": "-1.00", "currency": "USD", "reason": "damaged item"},
                }
            )

    def test_proposed_action_arguments_do_not_directly_satisfy_authorized_command(self):
        """A raw ProposedAction that "looks fine" still must not be usable as an AuthorizedCommand
        without going through explicit re-validation of its arguments."""
        proposal = ProposedAction(
            tool_name="issue_refund",
            arguments={"ticket_id": "T-1", "amount": "not-a-number", "currency": "USD", "reason": "damaged item"},
        )
        with pytest.raises(ValidationError):
            AuthorizedCommandAdapter.validate_python(
                {"tool_name": proposal.tool_name, "args": proposal.arguments}
            )
