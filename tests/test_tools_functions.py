"""Tests for the pure tool functions in src.tools.functions.

Each function takes an already-authorized command plus a Ticket and
returns the updated Ticket. Tests here only exercise the valid-command
path (the trust boundary / validation is exercised elsewhere in
tests/test_tools_schemas.py and tests/test_policy_interface.py); the
focus here is the resulting ticket state.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.tools.functions import add_ticket_note, issue_refund, update_ticket_status
from src.tools.schemas import (
    AuthorizedAddTicketNoteCommand,
    AuthorizedIssueRefundCommand,
    AuthorizedUpdateTicketStatusCommand,
    AddTicketNoteArgs,
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


class TestUpdateTicketStatus:
    def test_sets_new_status(self):
        ticket = make_ticket(status=TicketStatus.OPEN)
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.IN_PROGRESS, reason="starting work")
        )
        updated = update_ticket_status(command, ticket)
        assert updated.status is TicketStatus.IN_PROGRESS

    def test_bumps_updated_at(self):
        ticket = make_ticket()
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.ESCALATED, reason="needs manager")
        )
        updated = update_ticket_status(command, ticket)
        assert updated.updated_at > ticket.updated_at

    def test_does_not_mutate_input_ticket(self):
        ticket = make_ticket(status=TicketStatus.OPEN)
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.CLOSED, reason="resolved")
        )
        update_ticket_status(command, ticket)
        assert ticket.status is TicketStatus.OPEN

    def test_other_fields_unchanged(self):
        ticket = make_ticket()
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.CLOSED, reason="resolved")
        )
        updated = update_ticket_status(command, ticket)
        assert updated.id == ticket.id
        assert updated.customer_ref == ticket.customer_ref
        assert updated.refund_eligible_amount == ticket.refund_eligible_amount

    def test_mismatched_ticket_id_raises(self):
        ticket = make_ticket(id="T-1")
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-999", new_status=TicketStatus.CLOSED, reason="resolved")
        )
        with pytest.raises(ValueError):
            update_ticket_status(command, ticket)


class TestIssueRefund:
    def test_reduces_refund_eligible_amount(self):
        ticket = make_ticket(refund_eligible_amount=Decimal("50.00"))
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("20.00"), currency="USD", reason="damaged item")
        )
        updated = issue_refund(command, ticket)
        assert updated.refund_eligible_amount == Decimal("30.00")

    def test_full_amount_refund_leaves_zero_remaining(self):
        ticket = make_ticket(refund_eligible_amount=Decimal("50.00"))
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("50.00"), currency="USD", reason="damaged item")
        )
        updated = issue_refund(command, ticket)
        assert updated.refund_eligible_amount == Decimal("0")

    def test_remaining_amount_never_goes_negative(self):
        """Defensive floor: even if a refund somehow exceeds the eligible amount, it clamps at zero."""
        ticket = make_ticket(refund_eligible_amount=Decimal("10.00"))
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("10.00"), currency="USD", reason="damaged item")
        )
        updated = issue_refund(command, ticket)
        assert updated.refund_eligible_amount >= Decimal("0")

    def test_ticket_with_no_refund_eligible_amount_stays_none(self):
        ticket = make_ticket(refund_eligible_amount=None)
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("5.00"), currency="USD", reason="damaged item")
        )
        updated = issue_refund(command, ticket)
        assert updated.refund_eligible_amount is None

    def test_bumps_updated_at(self):
        ticket = make_ticket()
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("5.00"), currency="USD", reason="damaged item")
        )
        updated = issue_refund(command, ticket)
        assert updated.updated_at > ticket.updated_at

    def test_does_not_mutate_input_ticket(self):
        ticket = make_ticket(refund_eligible_amount=Decimal("50.00"))
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("20.00"), currency="USD", reason="damaged item")
        )
        issue_refund(command, ticket)
        assert ticket.refund_eligible_amount == Decimal("50.00")

    def test_does_not_change_status(self):
        ticket = make_ticket(status=TicketStatus.OPEN)
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("5.00"), currency="USD", reason="damaged item")
        )
        updated = issue_refund(command, ticket)
        assert updated.status is TicketStatus.OPEN

    def test_mismatched_ticket_id_raises(self):
        ticket = make_ticket(id="T-1")
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-999", amount=Decimal("5.00"), currency="USD", reason="damaged item")
        )
        with pytest.raises(ValueError):
            issue_refund(command, ticket)


class TestAddTicketNote:
    def test_appends_note(self):
        ticket = make_ticket()
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Called customer."))
        updated = add_ticket_note(command, ticket)
        assert updated.notes == ["Called customer."]

    def test_appends_to_existing_notes_preserving_order(self):
        ticket = make_ticket(notes=["First note."])
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Second note."))
        updated = add_ticket_note(command, ticket)
        assert updated.notes == ["First note.", "Second note."]

    def test_bumps_updated_at(self):
        ticket = make_ticket()
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Note."))
        updated = add_ticket_note(command, ticket)
        assert updated.updated_at > ticket.updated_at

    def test_does_not_mutate_input_ticket(self):
        ticket = make_ticket()
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Note."))
        add_ticket_note(command, ticket)
        assert ticket.notes == []

    def test_does_not_change_status_or_refund_amount(self):
        ticket = make_ticket(status=TicketStatus.OPEN, refund_eligible_amount=Decimal("50.00"))
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Note."))
        updated = add_ticket_note(command, ticket)
        assert updated.status is TicketStatus.OPEN
        assert updated.refund_eligible_amount == Decimal("50.00")

    def test_mismatched_ticket_id_raises(self):
        ticket = make_ticket(id="T-1")
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-999", note="Note."))
        with pytest.raises(ValueError):
            add_ticket_note(command, ticket)
