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

Every node factory (`make_propose_node`, `make_authorize_node`,
`make_execute_node`, `make_mark_denied_node`, `make_mark_completed_node`,
`make_mark_failed_node`) also accepts an optional `Tracer`. When given
one, the node calls the matching `Tracer.record_*` method — using
`state.run_id` as the tracer's key, since `AgentState` already carries it
— right after producing the value being recorded, before returning its
state update. `tracer` defaults to `None`, in which case the node behaves
exactly as before this phase: no tracing, no observability side effect.
`ExecutionWrapper.execute()` itself is unchanged — `execute_node` only
traces the single final `ExecutionResult` it returns, not its internal
retry attempts (out of scope for this phase).
"""

from __future__ import annotations

from typing import Callable, Literal

from src.execution.executor import ExecutionWrapper
from src.graph.state import AgentState, ExecutionStatus, RunStatus
from src.observability.tracer import Tracer
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


def make_propose_node(propose_fn: ProposeFn, tracer: Tracer | None = None) -> Callable[[AgentState], dict]:
    """Build a propose node bound to `propose_fn` (and, optionally, a `Tracer`).

    The returned node calls `propose_fn(state)` to get an (untrusted)
    `ProposedAction`, records it, and advances `run_status` to
    `AUTHORIZING` to reflect that the next step is policy authorization.
    When `tracer` is given, also calls `tracer.record_proposal(state.run_id,
    proposed_action)` right after the proposal is produced.
    """

    def propose_node(state: AgentState) -> dict:
        proposed_action = propose_fn(state)
        if tracer is not None:
            tracer.record_proposal(state.run_id, proposed_action)
        return {"proposed_action": proposed_action, "run_status": RunStatus.AUTHORIZING}

    return propose_node


def make_authorize_node(tracer: Tracer | None = None) -> Callable[[AgentState], dict]:
    """Build an authorize node (optionally bound to a `Tracer`).

    The returned node runs `state.proposed_action` through `authorize()`
    with the real policy rules wired in, and records the resulting
    `AuthorizationResult`. When approved, also advances `run_status` to
    `EXECUTING` to reflect that the next step is running the authorized
    command; when denied, `run_status` is left for `mark_denied_node`
    (reached via `route_after_authorize`) to set. When `tracer` is given,
    also calls `tracer.record_authorization(state.run_id, result)` right
    after the decision is produced.
    """

    def authorize_node(state: AgentState) -> dict:
        result = authorize(
            proposed_action=state.proposed_action,
            ticket=state.ticket,
            context=AuthorizationContext(),
            rules=[refund_amount_rule, status_transition_rule],
        )
        if tracer is not None:
            tracer.record_authorization(state.run_id, result)
        updates: dict = {"authorization_result": result}
        if result.decision is AuthorizationDecision.APPROVED:
            updates["run_status"] = RunStatus.EXECUTING
        return updates

    return authorize_node


def route_after_authorize(state: AgentState) -> Literal["execute", "denied"]:
    """Route to `execute` when authorization approved the proposal, otherwise to `denied`."""
    if state.authorization_result.decision is AuthorizationDecision.APPROVED:
        return "execute"
    return "denied"


def make_execute_node(
    execution_wrapper: ExecutionWrapper, tracer: Tracer | None = None
) -> Callable[[AgentState], dict]:
    """Build an execute node bound to `execution_wrapper` (and, optionally, a `Tracer`).

    The returned node runs `state.authorization_result.authorized_command`
    (only reachable once authorization has approved, per
    `route_after_authorize`) against `state.ticket` via
    `execution_wrapper.execute()`, and records the resulting
    `ExecutionResult`. It does not itself set `run_status` — that is
    `mark_completed_node`/`mark_failed_node`'s job, reached via
    `route_after_execute`. When `tracer` is given, also calls
    `tracer.record_execution(state.run_id, result)` with the single final
    `ExecutionResult` `execution_wrapper.execute()` returns —
    `ExecutionWrapper`'s internal retry attempts are not individually
    traced, by design, in this phase.
    """

    def execute_node(state: AgentState) -> dict:
        command = state.authorization_result.authorized_command
        result = execution_wrapper.execute(command, state.ticket)
        if tracer is not None:
            tracer.record_execution(state.run_id, result)
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


def make_mark_denied_node(tracer: Tracer | None = None) -> Callable[[AgentState], dict]:
    """Build a terminal node that stamps the run as denied (and, optionally, traces it)."""

    def mark_denied_node(state: AgentState) -> dict:
        if tracer is not None:
            tracer.record_run_finished(state.run_id, RunStatus.DENIED)
        return {"run_status": RunStatus.DENIED}

    return mark_denied_node


def make_mark_completed_node(tracer: Tracer | None = None) -> Callable[[AgentState], dict]:
    """Build a terminal node that stamps the run as completed (and, optionally, traces it)."""

    def mark_completed_node(state: AgentState) -> dict:
        if tracer is not None:
            tracer.record_run_finished(state.run_id, RunStatus.COMPLETED)
        return {"run_status": RunStatus.COMPLETED}

    return mark_completed_node


def make_mark_failed_node(tracer: Tracer | None = None) -> Callable[[AgentState], dict]:
    """Build a terminal node that stamps the run as failed (and, optionally, traces it)."""

    def mark_failed_node(state: AgentState) -> dict:
        if tracer is not None:
            tracer.record_run_finished(state.run_id, RunStatus.FAILED)
        return {"run_status": RunStatus.FAILED}

    return mark_failed_node
