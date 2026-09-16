"""Compiles the agent's LangGraph state machine from the node functions in src.graph.nodes.

Topology (a strict DAG — every edge points strictly forward, so no path
can revisit a node, which makes the graph provably terminating: it always
reaches one of the three terminal nodes within exactly 4 node
executions of `propose`, `authorize`, and either `execute` then a
terminal, or a terminal directly):

    START -> propose -> authorize --(execute)--> execute --(completed)--> mark_completed -> END
                              \\                          \\
                               (denied)                    (failed)
                                \\                           \\
                                 -> mark_denied -> END        -> mark_failed -> END

No node ever appears as its own ancestor: `propose` and `authorize` each
run at most once per invocation, `execute` runs at most once (its own
internal retries happen inside `ExecutionWrapper.execute()`, not by
looping back through this graph), and every terminal node has a single
outgoing edge straight to `END`. There is no conditional edge target that
points back to `propose`, `authorize`, or `execute`.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.execution.executor import ExecutionWrapper
from src.graph.nodes import (
    ProposeFn,
    make_authorize_node,
    make_execute_node,
    make_mark_completed_node,
    make_mark_denied_node,
    make_mark_failed_node,
    make_propose_node,
    route_after_authorize,
    route_after_execute,
)
from src.graph.state import AgentState
from src.observability.tracer import Tracer


def build_agent_graph(
    propose_fn: ProposeFn,
    execution_wrapper: ExecutionWrapper | None = None,
    tracer: Tracer | None = None,
) -> CompiledStateGraph:
    """Build and compile the propose -> authorize -> execute agent graph.

    `propose_fn` supplies proposals (a test fake today, a real LLM call in
    a later phase). `execution_wrapper` defaults to a fresh
    `ExecutionWrapper()` if not given; passing one in lets a caller (e.g.
    a test) inspect its idempotency store afterward or reuse it across
    multiple `invoke()` calls. `tracer` similarly defaults to a fresh
    `Tracer()` if not given, and every node in this graph shares that same
    instance, so a run's events all land under its `AgentState.run_id` in
    one place; pass one in to inspect `tracer.events_for_run(run_id)`
    after `invoke()`, or to share a `Tracer` across multiple graph runs.

    The compiled graph's `.invoke(AgentState(...))` returns a plain dict
    of the final state (LangGraph's convention for a pydantic state
    schema, not an `AgentState` instance) — reconstruct a typed object
    with `AgentState.model_validate(result)` if needed.
    """
    execution_wrapper = execution_wrapper or ExecutionWrapper()
    tracer = tracer or Tracer()

    graph = StateGraph(AgentState)
    graph.add_node("propose", make_propose_node(propose_fn, tracer))
    graph.add_node("authorize", make_authorize_node(tracer))
    graph.add_node("execute", make_execute_node(execution_wrapper, tracer))
    graph.add_node("mark_denied", make_mark_denied_node(tracer))
    graph.add_node("mark_completed", make_mark_completed_node(tracer))
    graph.add_node("mark_failed", make_mark_failed_node(tracer))

    graph.add_edge(START, "propose")
    graph.add_edge("propose", "authorize")
    graph.add_conditional_edges(
        "authorize", route_after_authorize, {"execute": "execute", "denied": "mark_denied"}
    )
    graph.add_conditional_edges(
        "execute", route_after_execute, {"completed": "mark_completed", "failed": "mark_failed"}
    )
    graph.add_edge("mark_denied", END)
    graph.add_edge("mark_completed", END)
    graph.add_edge("mark_failed", END)

    return graph.compile()
