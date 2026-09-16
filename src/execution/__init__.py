"""Execution layer.

Intended responsibility: executes commands that have already been
authorized and validated by the policy layer, wrapped with retry and
idempotency handling.

`ExecutionWrapper` (in `src.execution.executor`) dispatches an
`AuthorizedCommand` to the matching pure tool function in
`src.tools.functions`, verifies the result via `src.execution.
verification.verify()`, retries on injected transient failure, and
de-duplicates re-execution of an already-run command (successful or with
a failed verification) via an in-memory idempotency-key record.

A passing verification yields `ExecutionStatus.SUCCESS` with
`verified=True`; a failing one yields `ExecutionStatus.VERIFICATION_FAILED`
with `verified=False` and the verification reason as `error` — neither
case is retried further, since the tool function already ran. `execute()`
is now wired into the graph layer's `execute_node`
(`src.graph.nodes.make_execute_node`); no policy exists yet for what
happens next in response to a `VERIFICATION_FAILED` result beyond ending
the run as `RunStatus.FAILED`. No real persistence or I/O exists yet.

`ExecutionResult`/`ExecutionStatus` (in `src.execution.schemas`) are this
layer's own data contracts — `src.graph.state` imports them, mirroring how
`AuthorizationResult` is owned by `src.policy.schemas` and imported by
`src.graph.state` too.
"""

from src.execution.executor import ExecutionWrapper, FailurePredicate
from src.execution.schemas import ExecutionResult, ExecutionStatus
from src.execution.verification import VerificationOutcome, verify

__all__ = [
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutionWrapper",
    "FailurePredicate",
    "VerificationOutcome",
    "verify",
]
