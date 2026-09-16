"""Node and conditional-edge functions for the agent's LangGraph state machine.

Each node function takes the current `AgentState` and returns a partial
update dict (LangGraph merges it onto the state field-by-field) — this
matches the "nodes read state, return partial updates" convention already
documented on `AgentState`. Routing functions take the state and return a
string key used to look up the next node in a conditional-edge map.

`propose_node` takes no real LLM call: it wraps an injectable
`ProposeFn` callable (`Callable[[AgentState], ProposedAction]`) via
`make_propose_node()`, so callers (tests now, a real LLM integration
later) supply how a proposal is produced without this module knowing or
caring. `execute_node` similarly wraps an injected `ExecutionWrapper` via
`make_execute_node()`, so the same wrapper instance (and its idempotency
store) can be reused across a whole graph run, or a fresh one supplied
per test.

`authorize_node` wires in the two real rules that exist today
(`refund_amount_rule`, `status_transition_rule`) — this is graph wiring
only, not a place to add new business rules.

DENIED/COMPLETED/FAILED are set by dedicated terminal nodes
(`mark_denied_node`, `mark_completed_node`, `mark_failed_node`), not by
`authorize_node`/`execute_node` themselves, so each node has exactly one
responsibility: `authorize_node` decides and records the authorization
outcome, `execute_node` runs and records the execution outcome, and a
terminal node's only job is stamping the run's final `RunStatus`.
"""

from __future__ import annotations

from typing import Callable, Literal

from src.execution.executor import ExecutionWrapper
from src.graph.state import AgentState, ExecutionStatus, RunStatus
from src.policy.interface import authorize
from src.policy.rules import refund_amount_rule, status_transition_rule
from src.policy.schemas import AuthorizationContext, AuthorizationDecision
from src.tools.schemas import ProposedAction

ProposeFn = Callable[[AgentState], ProposedAction]
"""Stub signature for whatever produces a proposal: today a test fake, later a real LLM call.

Deliberately just `(AgentState) -> ProposedAction` — `propose_node` does
not know or care whether the implementation behind this is a hardcoded
fixture or a real model call.
"""


def make_propose_node(propose_fn: ProposeFn) -> Callable[[AgentState], dict]:
    """Build a propose node bound to `propose_fn`.

    The returned node calls `propose_fn(state)` to get an (untrusted)
    `ProposedAction`, records it, and advances `run_status` to
    `AUTHORIZING` to reflect that the next step is policy authorization.
    """

    def propose_node(state: AgentState) -> dict:
        proposed_action = propose_fn(state)
        return {"proposed_action": proposed_action, "run_status": RunStatus.AUTHORIZING}

    return propose_node


def authorize_node(state: AgentState) -> dict:
    """Run `state.proposed_action` through `authorize()` with the real policy rules wired in.

    Records the `AuthorizationResult`. When approved, also advances
    `run_status` to `EXECUTING` to reflect that the next step is running
    the authorized command; when denied, `run_status` is left for
    `mark_denied_node` (reached via `route_after_authorize`) to set.
    """
    result = authorize(
        proposed_action=state.proposed_action,
        ticket=state.ticket,
        context=AuthorizationContext(),
        rules=[refund_amount_rule, status_transition_rule],
    )
    updates: dict = {"authorization_result": result}
    if result.decision is AuthorizationDecision.APPROVED:
        updates["run_status"] = RunStatus.EXECUTING
    return updates


def route_after_authorize(state: AgentState) -> Literal["execute", "denied"]:
    """Route to `execute` when authorization approved the proposal, otherwise to `denied`."""
    if state.authorization_result.decision is AuthorizationDecision.APPROVED:
        return "execute"
    return "denied"


def make_execute_node(execution_wrapper: ExecutionWrapper) -> Callable[[AgentState], dict]:
    """Build an execute node bound to `execution_wrapper`.

    The returned node runs `state.authorization_result.authorized_command`
    (only reachable once authorization has approved, per
    `route_after_authorize`) against `state.ticket` via
    `execution_wrapper.execute()`, and records the resulting
    `ExecutionResult`. It does not itself set `run_status` — that is
    `mark_completed_node`/`mark_failed_node`'s job, reached via
    `route_after_execute`.
    """

    def execute_node(state: AgentState) -> dict:
        command = state.authorization_result.authorized_command
        result = execution_wrapper.execute(command, state.ticket)
        return {"execution_result": result}

    return execute_node


def route_after_execute(state: AgentState) -> Literal["completed", "failed"]:
    """Route to `completed` on `ExecutionStatus.SUCCESS`, otherwise to `failed`.

    `ExecutionStatus.FAILURE` (retries exhausted) and
    `ExecutionStatus.VERIFICATION_FAILED` both route to `failed`: a
    verification mismatch is a distinct, non-retryable outcome (decided in
    a prior phase) and is not looped back to `propose` or retried at the
    graph level — it simply ends the run as failed, same as exhausting
    `ExecutionWrapper`'s internal retries.
    """
    if state.execution_result.status is ExecutionStatus.SUCCESS:
        return "completed"
    return "failed"


def mark_denied_node(state: AgentState) -> dict:
    """Terminal node: stamp the run as denied."""
    return {"run_status": RunStatus.DENIED}


def mark_completed_node(state: AgentState) -> dict:
    """Terminal node: stamp the run as completed."""
    return {"run_status": RunStatus.COMPLETED}


def mark_failed_node(state: AgentState) -> dict:
    """Terminal node: stamp the run as failed."""
    return {"run_status": RunStatus.FAILED}
