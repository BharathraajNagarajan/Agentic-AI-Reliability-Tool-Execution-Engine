"""Policy layer.

Intended responsibility: the deterministic authorization layer that
decides whether a proposed action is allowed to run, independent of and
downstream from the LLM. This is the trust boundary between LLM proposal
and system-approved command.

The authorization data contracts (`AuthorizationResult`, `RuleOutcome`,
`AuthorizationContext`) live in `src.policy.schemas`, and the generic
rule-evaluation interface (`PolicyRule`, `authorize`) lives in
`src.policy.interface`. The first real rules (a refund-amount limit and a
status-transition table) live in `src.policy.rules`. Role/permission
checks and any further business rules do not exist yet.
"""

from src.policy.interface import PolicyRule, authorize
from src.policy.rules import (
    ALLOWED_STATUS_TRANSITIONS,
    refund_amount_rule,
    status_transition_rule,
)
from src.policy.schemas import AuthorizationContext, AuthorizationDecision, AuthorizationResult, RuleOutcome

__all__ = [
    "ALLOWED_STATUS_TRANSITIONS",
    "AuthorizationContext",
    "AuthorizationDecision",
    "AuthorizationResult",
    "PolicyRule",
    "RuleOutcome",
    "authorize",
    "refund_amount_rule",
    "status_transition_rule",
]
