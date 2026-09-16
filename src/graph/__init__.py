"""Graph layer.

Intended responsibility: the LangGraph state machine that drives the agent
loop — node definitions, edges/transitions, and the shared state schema
that flows between nodes.

The shared state schema (`AgentState` and its component models) is defined
in `src.graph.state`. `AuthorizationResult`/`AuthorizationDecision` are
owned by the policy layer (`src.policy.schemas`) and `ExecutionResult`/
`ExecutionStatus` are owned by the execution layer (`src.execution.
schemas`); both are re-exported here only because `AgentState` embeds
them.

Node and routing functions (`propose_node`/`authorize_node`/`execute_node`
and the conditional-edge routers between them) are defined in
`src.graph.nodes`; `build_agent_graph()` in `src.graph.build` wires them
into a compiled, provably-terminating (strict DAG, no cycles) LangGraph
state machine. `propose_node` takes an injectable proposal callable rather
than a real LLM call — that integration is a future phase. Every node
factory also accepts an optional `src.observability.tracer.Tracer`;
`build_agent_graph()` shares one `Tracer` across all of them, keyed by
`AgentState.run_id`.

This package's `__init__.py` intentionally does NOT re-export
`src.graph.nodes`/`src.graph.build`, only `src.graph.state` (plus the
policy/execution result types `AgentState` embeds): `src.graph.nodes` now
imports `src.observability.tracer`, which itself imports `RunStatus` from
`src.graph.state` — if this package eagerly imported `nodes`/`build` here,
merely importing `src.graph.state` from anywhere would force-load
`nodes`/`build` too (to finish initializing this package first), which in
turn re-enters `src.observability.tracer` before it has finished
initializing, causing a circular import. Import `src.graph.build` and
`src.graph.nodes` directly by their submodule path instead — every
existing caller already does.
"""

from src.execution.schemas import ExecutionResult, ExecutionStatus
from src.graph.state import AgentState, ConversationMessage, RunStatus
from src.policy.schemas import AuthorizationDecision, AuthorizationResult

__all__ = [
    "AgentState",
    "AuthorizationDecision",
    "AuthorizationResult",
    "ConversationMessage",
    "ExecutionResult",
    "ExecutionStatus",
    "RunStatus",
]
