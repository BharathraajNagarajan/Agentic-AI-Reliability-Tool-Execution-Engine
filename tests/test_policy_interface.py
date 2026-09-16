"""Tests for the policy layer's authorization interface (src.policy.interface).

These tests prove the `authorize()` signature and `PolicyRule` extension
point work end-to-end using hand-constructed FAKE rules only (a rule that
always approves, a rule that always denies with a reason). No real
business rules (refund thresholds, status-transition tables, role checks)
are exercised here — that is a later phase.
"""

from datetime import datetime, timezone
from decimal import Decimal

from src.policy.interface import authorize
from src.policy.schemas import AuthorizationContext, AuthorizationDecision, RuleOutcome
from src.tools.schemas import AuthorizedIssueRefundCommand, ProposedAction, Ticket, TicketStatus


def make_ticket(**overrides) -> Ticket:
    defaults = dict(
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
    defaults.update(overrides)
    return Ticket(**defaults)


def make_refund_proposal(**overrides) -> ProposedAction:
    arguments = dict(ticket_id="T-1", amount="10.00", currency="USD", reason="damaged item")
    arguments.update(overrides)
    return ProposedAction(tool_name="issue_refund", arguments=arguments)


def always_approve_rule(proposed_action: ProposedAction, ticket: Ticket, context: AuthorizationContext) -> RuleOutcome:
    """Fake rule used only to test the interface: unconditionally passes."""
    return RuleOutcome(rule_id="always-approve", description="Always approves.", passed=True, reason="Fake rule always passes.")


def always_deny_rule(proposed_action: ProposedAction, ticket: Ticket, context: AuthorizationContext) -> RuleOutcome:
    """Fake rule used only to test the interface: unconditionally denies."""
    return RuleOutcome(
        rule_id="always-deny", description="Always denies.", passed=False, reason="Fake rule always denies for testing."
    )


class TestAuthorizeSignature:
    def test_empty_rule_list_with_valid_proposal_approves(self):
        """With zero rules and a proposal whose args satisfy the tool schema, authorize() approves."""
        result = authorize(
            proposed_action=make_refund_proposal(),
            ticket=make_ticket(),
            context=AuthorizationContext(),
            rules=[],
        )
        assert result.decision is AuthorizationDecision.APPROVED
        assert isinstance(result.authorized_command, AuthorizedIssueRefundCommand)
        assert result.authorized_command.args.amount == Decimal("10.00")
        assert result.rule_id is None

    def test_all_passing_rules_approves(self):
        result = authorize(
            proposed_action=make_refund_proposal(),
            ticket=make_ticket(),
            context=AuthorizationContext(),
            rules=[always_approve_rule, always_approve_rule],
        )
        assert result.decision is AuthorizationDecision.APPROVED
        assert result.authorized_command is not None

    def test_single_denying_rule_denies_with_reason_and_rule_id(self):
        result = authorize(
            proposed_action=make_refund_proposal(),
            ticket=make_ticket(),
            context=AuthorizationContext(),
            rules=[always_deny_rule],
        )
        assert result.decision is AuthorizationDecision.DENIED
        assert result.authorized_command is None
        assert result.rule_id == "always-deny"
        assert "always denies" in result.reason

    def test_first_failing_rule_short_circuits_remaining_rules(self):
        calls: list[str] = []

        def tracking_pass(pa, t, c) -> RuleOutcome:
            calls.append("pass")
            return RuleOutcome(rule_id="tracker", description="tracks calls", passed=True, reason="ok")

        def tracking_fail(pa, t, c) -> RuleOutcome:
            calls.append("fail")
            return RuleOutcome(rule_id="tracker-fail", description="tracks calls", passed=False, reason="denied for test")

        def should_not_run(pa, t, c) -> RuleOutcome:
            calls.append("should_not_run")
            return RuleOutcome(rule_id="unreachable", description="unreachable", passed=True, reason="ok")

        result = authorize(
            proposed_action=make_refund_proposal(),
            ticket=make_ticket(),
            context=AuthorizationContext(),
            rules=[tracking_pass, tracking_fail, should_not_run],
        )
        assert result.decision is AuthorizationDecision.DENIED
        assert calls == ["pass", "fail"]

    def test_rules_receive_the_actual_proposed_action_and_ticket(self):
        """Confirms the interface actually threads proposed_action/ticket/context through to rules."""
        seen = {}

        def capturing_rule(pa: ProposedAction, t: Ticket, c: AuthorizationContext) -> RuleOutcome:
            seen["tool_name"] = pa.tool_name
            seen["ticket_id"] = t.id
            seen["requested_by"] = c.requested_by
            return RuleOutcome(rule_id="capture", description="captures inputs", passed=True, reason="ok")

        authorize(
            proposed_action=make_refund_proposal(),
            ticket=make_ticket(id="T-42"),
            context=AuthorizationContext(requested_by="operator-1"),
            rules=[capturing_rule],
        )
        assert seen == {"tool_name": "issue_refund", "ticket_id": "T-42", "requested_by": "operator-1"}

    def test_valid_proposal_that_all_rules_approve_still_denies_on_bad_schema(self):
        """Rules passing does not bypass the mechanical tool-argument validation step."""
        malformed_proposal = ProposedAction(
            tool_name="issue_refund",
            arguments={"ticket_id": "T-1", "amount": "not-a-number", "currency": "USD", "reason": "damaged item"},
        )
        result = authorize(
            proposed_action=malformed_proposal,
            ticket=make_ticket(),
            context=AuthorizationContext(),
            rules=[always_approve_rule],
        )
        assert result.decision is AuthorizationDecision.DENIED
        assert result.authorized_command is None

    def test_unknown_tool_name_denies_even_with_no_rules(self):
        hallucinated_proposal = ProposedAction(tool_name="delete_all_tickets_forever", arguments={})
        result = authorize(
            proposed_action=hallucinated_proposal,
            ticket=make_ticket(),
            context=AuthorizationContext(),
            rules=[],
        )
        assert result.decision is AuthorizationDecision.DENIED
        assert result.authorized_command is None
