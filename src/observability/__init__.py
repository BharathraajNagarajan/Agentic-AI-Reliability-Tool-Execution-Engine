"""Observability layer.

Intended responsibility: structured recording of traces and events for
agent runs (proposals, authorization decisions, executions, verification
outcomes).

`Tracer` (in `src.observability.tracer`) records an in-memory, ordered
`TraceEvent` sequence per run_id covering each proposal, authorization
decision, execution attempt, and the run's final status. It is not yet
wired into `src.graph.nodes`/`src.graph.build` — that integration, and any
real (persistent) storage backend, are future phases.
"""

from src.observability.tracer import EventType, TraceEvent, Tracer

__all__ = [
    "EventType",
    "TraceEvent",
    "Tracer",
]
