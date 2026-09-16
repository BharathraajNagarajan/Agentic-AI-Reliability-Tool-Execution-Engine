"""Tests for post-execution verification (src.execution.verification).

Covers a verification pass and a verification failure for each of the
three tools. Failure cases use a `ticket_after` that has been tampered
with / doesn't reflect the command's effect, to prove `verify()` actually
catches a mismatch rather than rubber-stamping everything.
"""

from datetime import datetime, timezone
from decimal import Decimal

from src.execution.verification import verify
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


class TestVerifyUpdateTicketStatus:
    def test_pass_when_status_matches_command(self):
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.IN_PROGRESS, reason="starting work")
        )
        before = make_ticket(status=TicketStatus.OPEN)
        after = make_ticket(status=TicketStatus.IN_PROGRESS)
        outcome = verify(command, before, after)
        assert outcome.verified is True

    def test_fail_when_status_was_not_actually_changed(self):
        """ticket_after still shows the old status: the execution didn't take effect."""
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.IN_PROGRESS, reason="starting work")
        )
        before = make_ticket(status=TicketStatus.OPEN)
        after = make_ticket(status=TicketStatus.OPEN)
        outcome = verify(command, before, after)
        assert outcome.verified is False
        assert "in_progress" in outcome.reason

    def test_fail_when_status_was_tampered_with_a_different_value(self):
        """ticket_after shows some other status entirely, not what was authorized."""
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.IN_PROGRESS, reason="starting work")
        )
        before = make_ticket(status=TicketStatus.OPEN)
        after = make_ticket(status=TicketStatus.CLOSED)
        outcome = verify(command, before, after)
        assert outcome.verified is False


class TestVerifyIssueRefund:
    def test_pass_when_amount_decreased_exactly(self):
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("20.00"), currency="USD", reason="damaged item")
        )
        before = make_ticket(refund_eligible_amount=Decimal("50.00"))
        after = make_ticket(refund_eligible_amount=Decimal("30.00"))
        outcome = verify(command, before, after)
        assert outcome.verified is True

    def test_pass_respects_floor_at_zero_clamp(self):
        """Refunding more than the remaining eligible amount should floor at 0, matching src.tools.functions."""
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("999.00"), currency="USD", reason="damaged item")
        )
        before = make_ticket(refund_eligible_amount=Decimal("50.00"))
        after = make_ticket(refund_eligible_amount=Decimal("0"))
        outcome = verify(command, before, after)
        assert outcome.verified is True

    def test_pass_when_ticket_had_no_refund_eligible_amount(self):
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("5.00"), currency="USD", reason="damaged item")
        )
        before = make_ticket(refund_eligible_amount=None)
        after = make_ticket(refund_eligible_amount=None)
        outcome = verify(command, before, after)
        assert outcome.verified is True

    def test_fail_when_amount_was_not_actually_decreased(self):
        """ticket_after still shows the pre-refund amount: the execution didn't take effect."""
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("20.00"), currency="USD", reason="damaged item")
        )
        before = make_ticket(refund_eligible_amount=Decimal("50.00"))
        after = make_ticket(refund_eligible_amount=Decimal("50.00"))
        outcome = verify(command, before, after)
        assert outcome.verified is False

    def test_fail_when_amount_decreased_by_the_wrong_number(self):
        """ticket_after was tampered with: decreased by a different amount than the command authorized."""
        command = AuthorizedIssueRefundCommand(
            args=IssueRefundArgs(ticket_id="T-1", amount=Decimal("20.00"), currency="USD", reason="damaged item")
        )
        before = make_ticket(refund_eligible_amount=Decimal("50.00"))
        after = make_ticket(refund_eligible_amount=Decimal("10.00"))  # decreased by 40, not 20
        outcome = verify(command, before, after)
        assert outcome.verified is False
        assert "30.00" in outcome.reason  # expected value should be reported


class TestVerifyAddTicketNote:
    def test_pass_when_note_present(self):
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Called customer."))
        before = make_ticket(notes=[])
        after = make_ticket(notes=["Called customer."])
        outcome = verify(command, before, after)
        assert outcome.verified is True

    def test_pass_when_note_appended_after_existing_notes(self):
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Second note."))
        before = make_ticket(notes=["First note."])
        after = make_ticket(notes=["First note.", "Second note."])
        outcome = verify(command, before, after)
        assert outcome.verified is True

    def test_fail_when_note_missing_entirely(self):
        """ticket_after's notes were never updated: the execution didn't take effect."""
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Called customer."))
        before = make_ticket(notes=[])
        after = make_ticket(notes=[])
        outcome = verify(command, before, after)
        assert outcome.verified is False

    def test_fail_when_notes_were_tampered_with_different_text(self):
        """ticket_after has a note, but not the one that was authorized."""
        command = AuthorizedAddTicketNoteCommand(args=AddTicketNoteArgs(ticket_id="T-1", note="Called customer."))
        before = make_ticket(notes=[])
        after = make_ticket(notes=["Something else entirely."])
        outcome = verify(command, before, after)
        assert outcome.verified is False


class TestVerifyTicketIdentityGuard:
    def test_fail_when_ticket_after_is_a_different_ticket(self):
        command = AuthorizedUpdateTicketStatusCommand(
            args=UpdateTicketStatusArgs(ticket_id="T-1", new_status=TicketStatus.IN_PROGRESS, reason="starting work")
        )
        before = make_ticket(id="T-1", status=TicketStatus.OPEN)
        after = make_ticket(id="T-2", status=TicketStatus.IN_PROGRESS)
        outcome = verify(command, before, after)
        assert outcome.verified is False
        assert "does not match" in outcome.reason
