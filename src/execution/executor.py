"""Retry/idempotency execution wrapper around the pure tool functions.

`ExecutionWrapper.execute()` is the only thing in this module with any
state or control flow: it dispatches an `AuthorizedCommand` to the right
pure function in `src.tools.functions`, verifies the resulting ticket
state via `src.execution.verification.verify()`, retries on injected
transient failure, and de-duplicates re-execution of a command it has
already completed (successfully or with a failed verification). It
performs no real I/O and calls no external system — the tool functions it
wraps remain pure and in-memory.

Verification is not itself retried: a `should_fail`-simulated transient
failure means the tool function was never called and another attempt is
warranted, but once the tool function actually runs, whatever it produced
is final for that command — a failed verification means `ExecutionStatus.
VERIFICATION_FAILED`, not another attempt, since re-running the same
tool function again would risk double-applying its effect.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, MutableMapping

from src.execution.schemas import ExecutionResult, ExecutionStatus
from src.execution.verification import verify
from src.tools.functions import add_ticket_note, issue_refund, update_ticket_status
from src.tools.schemas import (
    AuthorizedAddTicketNoteCommand,
    AuthorizedCommand,
    AuthorizedIssueRefundCommand,
    AuthorizedUpdateTicketStatusCommand,
    Ticket,
)

FailurePredicate = Callable[[int], bool]
"""A callable `(attempt: int) -> bool` used to simulate transient failure.

`attempt` is the 1-indexed attempt number about to run. Returning `True`
makes `execute()` treat that attempt as a transient failure (the
underlying tool function is NOT called) and move on to the next attempt,
without this being a real I/O failure. Tests use this to force N failures
before a success, or force failure on every attempt.
"""

_TOOL_FUNCTIONS: dict[type, Callable[[AuthorizedCommand, Ticket], Ticket]] = {
    AuthorizedUpdateTicketStatusCommand: update_ticket_status,
    AuthorizedIssueRefundCommand: issue_refund,
    AuthorizedAddTicketNoteCommand: add_ticket_note,
}
"""Dispatch table from concrete AuthorizedCommand type to the pure tool function that applies it.

Module-level (rather than inlined in `execute()`) so tests can substitute
an entry (e.g. via `monkeypatch.setitem`) to observe/control calls to the
underlying tool function without touching real I/O.
"""


def _compute_idempotency_key(command: AuthorizedCommand, ticket: Ticket) -> str:
    """Deterministically derive an idempotency key from a command's content and the target ticket.

    Built from the command's tool name, its validated arguments (JSON-
    encoded so a `Decimal` amount or enum status participates in the hash
    by value, not by Python object identity), and the ticket id. Same
    inputs always produce the same key; any difference in tool name,
    argument values, or ticket id produces a different key.
    """
    payload = {
        "tool_name": command.tool_name.value,
        "args": command.args.model_dump(mode="json"),
        "ticket_id": ticket.id,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ExecutionWrapper:
    """Executes AuthorizedCommands against a Ticket with retry and idempotency.

    Holds an in-memory record, for the lifetime of this instance, of
    idempotency keys for commands the underlying tool function has already
    run for (whether that run's effect was subsequently verified or not).
    Re-executing a command (same tool, same validated arguments, same
    ticket id) that is already recorded returns the previously recorded
    `ExecutionResult` unchanged — the underlying tool function is not
    called again and no further mutation occurs. A `should_fail`-simulated
    transient failure is NOT recorded, since the tool function never ran.

    No real persistence: the record is a plain in-memory dict that is
    gone once this instance is garbage collected. A future phase can
    swap this for a real store without changing `execute()`'s behavior.
    """

    def __init__(self) -> None:
        self._idempotency_store: dict[str, ExecutionResult] = {}

    @property
    def idempotency_store(self) -> MutableMapping[str, ExecutionResult]:
        """Read-only-in-spirit view of recorded successful executions, keyed by idempotency key."""
        return dict(self._idempotency_store)

    def execute(
        self,
        command: AuthorizedCommand,
        ticket: Ticket,
        *,
        max_attempts: int = 3,
        should_fail: FailurePredicate | None = None,
    ) -> ExecutionResult:
        """Run `command` against `ticket`, retrying on simulated transient failure.

        1. Compute the command's idempotency key. If it is already
           recorded (the tool function already ran for this exact command,
           whether verified or not), return that recorded `ExecutionResult`
           immediately without calling any tool function.
        2. Otherwise, for attempt 1..max_attempts: if `should_fail(attempt)`
           is True, treat this attempt as a transient failure and continue
           to the next attempt without calling the tool function.
        3. Once the tool function actually runs, its result is final for
           this call: validate the resulting ticket dict back into a
           `Ticket` and verify it against `command`. A passing verification
           returns `ExecutionStatus.SUCCESS` with `verified=True`; a failing
           one returns `ExecutionStatus.VERIFICATION_FAILED` with
           `verified=False` and the verification reason as `error` —
           neither case retries further, and both are recorded in the
           idempotency store so re-executing the identical command later
           returns this same result rather than running the tool function
           (and risking double-applying its effect) again.
        4. If every attempt is exhausted via `should_fail` without the tool
           function ever running, return a FAILURE result (never raises)
           with `attempt` set to `max_attempts`. This case is NOT recorded
           in the idempotency store, since nothing was actually applied.

        A real exception raised by the underlying tool function (e.g. a
        ticket-id mismatch) is not treated as a transient failure and
        propagates — only `should_fail` simulates retryable failure.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        should_fail = should_fail or (lambda attempt: False)

        idempotency_key = _compute_idempotency_key(command, ticket)
        cached_result = self._idempotency_store.get(idempotency_key)
        if cached_result is not None:
            return cached_result

        tool_fn = _TOOL_FUNCTIONS[type(command)]
        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            if should_fail(attempt):
                last_error = f"Simulated transient failure on attempt {attempt}."
                continue

            updated_ticket = tool_fn(command, ticket)
            ticket_after = Ticket.model_validate(updated_ticket.model_dump(mode="json"))
            verification = verify(command, ticket_before=ticket, ticket_after=ticket_after)

            if verification.verified:
                result = ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    output={"ticket": updated_ticket.model_dump(mode="json")},
                    verified=True,
                    attempt=attempt,
                )
            else:
                result = ExecutionResult(
                    status=ExecutionStatus.VERIFICATION_FAILED,
                    output={"ticket": updated_ticket.model_dump(mode="json")},
                    error=verification.reason,
                    verified=False,
                    attempt=attempt,
                )

            self._idempotency_store[idempotency_key] = result
            return result

        return ExecutionResult(
            status=ExecutionStatus.FAILURE,
            error=last_error or f"Execution failed after {max_attempts} attempt(s).",
            attempt=max_attempts,
        )
