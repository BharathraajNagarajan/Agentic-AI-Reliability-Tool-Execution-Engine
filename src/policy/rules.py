"""The first real policy rules for the support-ticket ops agent.

Each rule here is a `PolicyRule`-conforming callable (see
`src.policy.interface`): a pure function of `(proposed_action, ticket,
context) -> RuleOutcome`, usable in the `rules` list passed to
`authorize()` without any change to that function's signature.

Because `authorize()` runs rules BEFORE the mechanical tool-argument schema
validation described in `src.tools.schemas`, `proposed_action.arguments`
here is still the raw, untrusted dict — it has not yet been proven to
match the tool's argument schema. Each rule below re-validates the subset
of arguments it needs via the relevant `*Args` model and, if that
sub-validation fails, treats the rule as "not applicable" (passes) rather
than raising or denying itself — malformed arguments are `authorize()`'s
own schema-validation step's responsibility to reject, not this rule's.
"""

from __future__ import annotations

from pydantic import ValidationError

from src.policy.schemas import AuthorizationContext, RuleOutcome
from src.tools.schemas import IssueRefundArgs, ProposedAction, Ticket, TicketStatus, ToolName, UpdateTicketStatusArgs

REFUND_AMOUNT_RULE_ID = "refund-amount-limit"


def refund_amount_rule(proposed_action: ProposedAction, ticket: Ticket, context: AuthorizationContext) -> RuleOutcome:
    """Deny an `issue_refund` proposal whose amount exceeds the ticket's refund-eligible amount.

    An amount exactly equal to `ticket.refund_eligible_amount` passes. A
    ticket with no `refund_eligible_amount` set is treated as not eligible
    for any refund amount. Proposals for any other tool are not this
    rule's concern and pass through untouched.
    """
    description = "Denies issue_refund proposals whose amount exceeds the ticket's refund_eligible_amount."

    if proposed_action.tool_name != ToolName.ISSUE_REFUND.value:
        return RuleOutcome(
            rule_id=REFUND_AMOUNT_RULE_ID,
            description=description,
            passed=True,
            reason="Not an issue_refund proposal; rule not applicable.",
        )

    try:
        args = IssueRefundArgs.model_validate(proposed_action.arguments)
    except ValidationError:
        return RuleOutcome(
            rule_id=REFUND_AMOUNT_RULE_ID,
            description=description,
            passed=True,
            reason="Arguments do not match the issue_refund schema; deferring to argument validation.",
        )

    if ticket.refund_eligible_amount is None:
        return RuleOutcome(
            rule_id=REFUND_AMOUNT_RULE_ID,
            description=description,
            passed=False,
            reason=f"Ticket '{ticket.id}' has no refund-eligible amount; no refund may be issued against it.",
        )

    if args.amount > ticket.refund_eligible_amount:
        return RuleOutcome(
            rule_id=REFUND_AMOUNT_RULE_ID,
            description=description,
            passed=False,
            reason=(
                f"Requested refund amount {args.amount} exceeds ticket '{ticket.id}'s "
                f"refund-eligible amount {ticket.refund_eligible_amount}."
            ),
        )

    return RuleOutcome(
        rule_id=REFUND_AMOUNT_RULE_ID,
        description=description,
        passed=True,
        reason=(
            f"Requested refund amount {args.amount} is within ticket '{ticket.id}'s "
            f"refund-eligible amount {ticket.refund_eligible_amount}."
        ),
    )


STATUS_TRANSITION_RULE_ID = "status-transition"

ALLOWED_STATUS_TRANSITIONS: dict[TicketStatus, frozenset[TicketStatus]] = {
    TicketStatus.OPEN: frozenset({TicketStatus.IN_PROGRESS, TicketStatus.ESCALATED, TicketStatus.CLOSED}),
    TicketStatus.IN_PROGRESS: frozenset(
        {TicketStatus.PENDING_CUSTOMER, TicketStatus.ESCALATED, TicketStatus.RESOLVED, TicketStatus.CLOSED}
    ),
    TicketStatus.PENDING_CUSTOMER: frozenset(
        {TicketStatus.IN_PROGRESS, TicketStatus.ESCALATED, TicketStatus.CLOSED}
    ),
    TicketStatus.ESCALATED: frozenset({TicketStatus.IN_PROGRESS, TicketStatus.RESOLVED, TicketStatus.CLOSED}),
    TicketStatus.RESOLVED: frozenset({TicketStatus.IN_PROGRESS, TicketStatus.CLOSED}),
    TicketStatus.CLOSED: frozenset(),
}
"""Minimal allowed-transitions table: current status -> set of statuses it may move to.

`CLOSED` is terminal (empty set): no further transitions are allowed once
a ticket is closed. A status is never considered a valid "transition" to
itself, so re-proposing the current status is denied like any other
disallowed transition.
"""


def status_transition_rule(
    proposed_action: ProposedAction, ticket: Ticket, context: AuthorizationContext
) -> RuleOutcome:
    """Deny an `update_ticket_status` proposal whose `new_status` is not reachable from the ticket's current status.

    Reachability is defined by `ALLOWED_STATUS_TRANSITIONS`. Proposals for
    any other tool are not this rule's concern and pass through untouched.
    """
    description = (
        "Denies update_ticket_status proposals whose new_status is not a valid "
        "transition from the ticket's current status."
    )

    if proposed_action.tool_name != ToolName.UPDATE_TICKET_STATUS.value:
        return RuleOutcome(
            rule_id=STATUS_TRANSITION_RULE_ID,
            description=description,
            passed=True,
            reason="Not an update_ticket_status proposal; rule not applicable.",
        )

    try:
        args = UpdateTicketStatusArgs.model_validate(proposed_action.arguments)
    except ValidationError:
        return RuleOutcome(
            rule_id=STATUS_TRANSITION_RULE_ID,
            description=description,
            passed=True,
            reason="Arguments do not match the update_ticket_status schema; deferring to argument validation.",
        )

    allowed_next_statuses = ALLOWED_STATUS_TRANSITIONS.get(ticket.status, frozenset())
    if args.new_status not in allowed_next_statuses:
        return RuleOutcome(
            rule_id=STATUS_TRANSITION_RULE_ID,
            description=description,
            passed=False,
            reason=(
                f"Cannot transition ticket '{ticket.id}' from '{ticket.status.value}' to "
                f"'{args.new_status.value}': not an allowed transition."
            ),
        )

    return RuleOutcome(
        rule_id=STATUS_TRANSITION_RULE_ID,
        description=description,
        passed=True,
        reason=f"Transition from '{ticket.status.value}' to '{args.new_status.value}' is allowed.",
    )
