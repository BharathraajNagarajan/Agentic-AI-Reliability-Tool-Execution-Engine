"""Policy layer authorization interface.

This module defines the SHAPE of authorization — a generic rule-evaluation
engine — without encoding any real business rules. No refund thresholds,
no allowed-status-transition tables, no role/permission checks live here;
those are a later phase. What exists here is:

- `PolicyRule`: the protocol a rule callable must satisfy, so real rules
  can be written and plugged in later without changing `authorize()`.
- `authorize()`: takes an untrusted `ProposedAction`, the `Ticket` it
  targets, an `AuthorizationContext`, and a list of `PolicyRule`s to run.
  It runs every rule, denies on the first failing rule, and otherwise
  validates the proposal's raw tool name/arguments into a trusted
  `AuthorizedCommand` (the mechanical trust-boundary check every proposal
  must pass regardless of which business rules exist) before approving.

Because `rules` is a plain parameter, callers (initially: tests with fake
rules, later: the graph's authorize node with real rules) fully control
policy behavior without `authorize()`'s signature ever needing to change.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from pydantic import ValidationError

from src.policy.schemas import AuthorizationContext, AuthorizationDecision, AuthorizationResult, RuleOutcome
from src.tools.schemas import AuthorizedCommandAdapter, ProposedAction, Ticket


class PolicyRule(Protocol):
    """A single policy rule: a pure function of a proposed action, ticket, and context.

    A conforming rule must not depend on anything outside its three
    arguments and must return a `RuleOutcome` describing its verdict. This
    is the extension point real business rules will implement later.
    """

    def __call__(
        self, proposed_action: ProposedAction, ticket: Ticket, context: AuthorizationContext
    ) -> RuleOutcome: ...


def authorize(
    proposed_action: ProposedAction,
    ticket: Ticket,
    context: AuthorizationContext,
    rules: Sequence[PolicyRule],
) -> AuthorizationResult:
    """Evaluate `proposed_action` against `rules` and return an `AuthorizationResult`.

    Behavior (deliberately generic, with zero business logic):

    1. Run each rule in `rules`, in order, against the same
       (proposed_action, ticket, context) inputs.
    2. If any rule fails, deny immediately with that rule's id and reason.
       Later rules are not evaluated.
    3. If every rule passes (including the trivial case of an empty rule
       list), attempt to validate `proposed_action`'s raw tool name and
       arguments into a strongly-typed `AuthorizedCommand`. This is the
       mechanical trust-boundary check described in `src.tools.schemas` —
       not a business rule — and applies no matter what rules were passed.
    4. If that validation fails (unknown tool name, missing/malformed
       arguments), deny with a reason describing the schema mismatch.
    5. Otherwise, approve and attach the resulting `AuthorizedCommand`.
    """
    for rule in rules:
        outcome = rule(proposed_action, ticket, context)
        if not outcome.passed:
            return AuthorizationResult(
                decision=AuthorizationDecision.DENIED,
                reason=outcome.reason,
                rule_id=outcome.rule_id,
            )

    try:
        authorized_command = AuthorizedCommandAdapter.validate_python(
            {"tool_name": proposed_action.tool_name, "args": proposed_action.arguments}
        )
    except ValidationError as exc:
        return AuthorizationResult(
            decision=AuthorizationDecision.DENIED,
            reason=f"Proposed action failed tool argument validation: {exc}",
            rule_id=None,
        )

    return AuthorizationResult(
        decision=AuthorizationDecision.APPROVED,
        authorized_command=authorized_command,
        reason="All policy rules passed and arguments validated for the requested tool.",
        rule_id=None,
    )
