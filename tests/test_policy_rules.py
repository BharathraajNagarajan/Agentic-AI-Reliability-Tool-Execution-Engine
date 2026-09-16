"""Tests for the first real policy rules (src.policy.rules).

Covers the approve and deny paths for each rule in isolation, boundary
cases (refund amount exactly equal to the eligible amount, closed tickets
as a terminal status), and both rules wired together into `authorize()`.
"""

from datetime import datetime, timezone
from decimal import Decimal

from src.policy.interface import authorize
from src.policy.rules import refund_amount_rule, status_transition_rule
from src.policy.schemas import AuthorizationContext, AuthorizationDecision
from src.tools.schemas import ProposedAction, Ticket, TicketStatus


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


def make_refund_proposal(**overrides) -> ProposedAction:
    arguments = dict(ticket_id="T-1", amount="10.00", currency="USD", reason="damaged item")
    arguments.update(overrides)
    return ProposedAction(tool_name="issue_refund", arguments=arguments)


def make_status_proposal(**overrides) -> ProposedAction:
    arguments = dict(ticket_id="T-1", new_status="in_progress", reason="starting work")
    arguments.update(overrides)
    return ProposedAction(tool_name="update_ticket_status", arguments=arguments)


CONTEXT = AuthorizationContext()


class TestRefundAmountRule:
    def test_amount_under_limit_passes(self):
        outcome = refund_amount_rule(make_refund_proposal(amount="10.00"), make_ticket(), CONTEXT)
        assert outcome.passed is True
        assert outcome.rule_id == "refund-amount-limit"

    def test_amount_exactly_equal_to_limit_passes(self):
        """Boundary case: amount == refund_eligible_amount must pass, not be treated as 'exceeding'."""
        outcome = refund_amount_rule(
            make_refund_proposal(amount="50.00"), make_ticket(refund_eligible_amount=Decimal("50.00")), CONTEXT
        )
        assert outcome.passed is True

    def test_amount_over_limit_denies(self):
        outcome = refund_amount_rule(
            make_refund_proposal(amount="50.01"), make_ticket(refund_eligible_amount=Decimal("50.00")), CONTEXT
        )
        assert outcome.passed is False
        assert "exceeds" in outcome.reason

    def test_amount_one_cent_over_limit_denies(self):
        """Boundary case just past the limit, adjacent to the exact-equality boundary above."""
        outcome = refund_amount_rule(
            make_refund_proposal(amount="50.01"), make_ticket(refund_eligible_amount=Decimal("50.00")), CONTEXT
        )
        assert outcome.passed is False

    def test_ticket_with_no_refund_eligible_amount_denies(self):
        outcome = refund_amount_rule(
            make_refund_proposal(amount="1.00"), make_ticket(refund_eligible_amount=None), CONTEXT
        )
        assert outcome.passed is False
        assert "no refund-eligible amount" in outcome.reason

    def test_non_refund_proposal_is_not_applicable_and_passes(self):
        outcome = refund_amount_rule(make_status_proposal(), make_ticket(), CONTEXT)
        assert outcome.passed is True
        assert "not applicable" in outcome.reason

    def test_malformed_refund_arguments_defer_and_pass(self):
        """Rule defers to authorize()'s schema validation rather than erroring on bad args."""
        malformed = ProposedAction(tool_name="issue_refund", arguments={"amount": "not-a-number"})
        outcome = refund_amount_rule(malformed, make_ticket(), CONTEXT)
        assert outcome.passed is True
        assert "deferring" in outcome.reason


class TestStatusTransitionRule:
    def test_allowed_transition_passes(self):
        outcome = status_transition_rule(
            make_status_proposal(new_status="in_progress"), make_ticket(status=TicketStatus.OPEN), CONTEXT
        )
        assert outcome.passed is True
        assert outcome.rule_id == "status-transition"

    def test_disallowed_transition_denies(self):
        outcome = status_transition_rule(
            make_status_proposal(new_status="resolved"), make_ticket(status=TicketStatus.OPEN), CONTEXT
        )
        assert outcome.passed is False
        assert "not an allowed transition" in outcome.reason

    def test_transition_from_terminal_closed_status_denies(self):
        """Boundary case: CLOSED is terminal, so every proposed transition out of it is denied."""
        outcome = status_transition_rule(
            make_status_proposal(new_status="open"), make_ticket(status=TicketStatus.CLOSED), CONTEXT
        )
        assert outcome.passed is False

    def test_transition_to_same_status_denies(self):
        """Boundary case: re-proposing the current status is not a valid transition."""
        outcome = status_transition_rule(
            make_status_proposal(new_status="open"), make_ticket(status=TicketStatus.OPEN), CONTEXT
        )
        assert outcome.passed is False

    def test_non_status_proposal_is_not_applicable_and_passes(self):
        outcome = status_transition_rule(make_refund_proposal(), make_ticket(), CONTEXT)
        assert outcome.passed is True
        assert "not applicable" in outcome.reason

    def test_malformed_status_arguments_defer_and_pass(self):
        malformed = ProposedAction(tool_name="update_ticket_status", arguments={"new_status": "not_a_real_status"})
        outcome = status_transition_rule(malformed, make_ticket(), CONTEXT)
        assert outcome.passed is True
        assert "deferring" in outcome.reason


class TestBothRulesWiredIntoAuthorize:
    def test_refund_within_limit_and_valid_status_proposal_both_approve(self):
        result = authorize(
            proposed_action=make_refund_proposal(amount="25.00"),
            ticket=make_ticket(refund_eligible_amount=Decimal("50.00")),
            context=CONTEXT,
            rules=[refund_amount_rule, status_transition_rule],
        )
        assert result.decision is AuthorizationDecision.APPROVED
        assert result.authorized_command.args.amount == Decimal("25.00")

    def test_status_transition_approves_when_refund_rule_not_applicable(self):
        result = authorize(
            proposed_action=make_status_proposal(new_status="escalated"),
            ticket=make_ticket(status=TicketStatus.OPEN),
            context=CONTEXT,
            rules=[refund_amount_rule, status_transition_rule],
        )
        assert result.decision is AuthorizationDecision.APPROVED
        assert result.authorized_command.args.new_status.value == "escalated"

    def test_refund_over_limit_denied_by_refund_rule_before_status_rule_runs(self):
        result = authorize(
            proposed_action=make_refund_proposal(amount="999.00"),
            ticket=make_ticket(refund_eligible_amount=Decimal("50.00")),
            context=CONTEXT,
            rules=[refund_amount_rule, status_transition_rule],
        )
        assert result.decision is AuthorizationDecision.DENIED
        assert result.rule_id == "refund-amount-limit"
        assert result.authorized_command is None

    def test_disallowed_status_transition_denied_by_status_rule(self):
        result = authorize(
            proposed_action=make_status_proposal(new_status="resolved"),
            ticket=make_ticket(status=TicketStatus.CLOSED),
            context=CONTEXT,
            rules=[refund_amount_rule, status_transition_rule],
        )
        assert result.decision is AuthorizationDecision.DENIED
        assert result.rule_id == "status-transition"
        assert result.authorized_command is None
