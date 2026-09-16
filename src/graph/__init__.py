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
than a real LLM call — that integration is a future phase.
"""

from src.graph.build import build_agent_graph
from src.graph.nodes import (
    ProposeFn,
    authorize_node,
    make_execute_node,
    make_propose_node,
    mark_completed_node,
    mark_denied_node,
    mark_failed_node,
    route_after_authorize,
    route_after_execute,
)
from src.graph.state import (
    AgentState,
    ConversationMessage,
    ExecutionResult,
    ExecutionStatus,
    RunStatus,
)
from src.policy.schemas import AuthorizationDecision, AuthorizationResult

__all__ = [
    "AgentState",
    "AuthorizationDecision",
    "AuthorizationResult",
    "ConversationMessage",
    "ExecutionResult",
    "ExecutionStatus",
    "ProposeFn",
    "RunStatus",
    "authorize_node",
    "build_agent_graph",
    "make_execute_node",
    "make_propose_node",
    "mark_completed_node",
    "mark_denied_node",
    "mark_failed_node",
    "route_after_authorize",
    "route_after_execute",
]
