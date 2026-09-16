"""Tests for the Tracer wired into the compiled agent graph (src.graph.build.build_agent_graph).

Builds a graph with an explicit Tracer, invokes it, and confirms
tracer.events_for_run(run_id) contains the expected event sequence in the
expected order for an approved-and-completed run and a denied run.
"""

from datetime import datetime, timezone
from decimal import Decimal

from src.graph.build import build_agent_graph
from src.graph.state import AgentState, AuthorizationDecision, ExecutionStatus, RunStatus
from src.observability.tracer import EventType, Tracer
from src.tools.schemas import ProposedAction, Ticket, TicketStatus


def make_ticket(**overrides) -> Ticket:
    defaults = dict(
        id="T-1",
        status=TicketStatus.OPEN,
        customer_ref="cust-123",
        subject="Package arrived damaged",
        order_ref="ORD-456",
        refund_eligible_amount=Decimal("50.00"),
        currency="USD",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return Ticket(**defaults)


def make_initial_state(ticket: Ticket, run_id: str = "run-1") -> AgentState:
    return AgentState(run_id=run_id, ticket=ticket, task_description="Handle this ticket.")


class TestTracingApprovedAndCompletedRun:
    def test_events_recorded_in_expected_order(self):
        ticket = make_ticket(refund_eligible_amount=Decimal("50.00"))
        run_id = "run-completed"

        def propose_fn(state: AgentState) -> ProposedAction:
            return ProposedAction(
                tool_name="issue_refund",
                arguments={"ticket_id": state.ticket.id, "amount": "20.00", "currency": "USD", "reason": "damaged item"},
            )

        tracer = Tracer()
        graph = build_agent_graph(propose_fn, tracer=tracer)
        result = graph.invoke(make_initial_state(ticket, run_id=run_id))
        final_state = AgentState.model_validate(result)

        assert final_state.run_status is RunStatus.COMPLETED

        events = tracer.events_for_run(run_id)
        assert [e.event_type for e in events] == [
            EventType.PROPOSED,
            EventType.AUTHORIZED,
            EventType.EXECUTED,
            EventType.RUN_FINISHED,
        ]
        assert [e.sequence for e in events] == [0, 1, 2, 3]

    def test_event_payloads_match_the_run_outcome(self):
        ticket = make_ticket(refund_eligible_amount=Decimal("50.00"))
        run_id = "run-completed-payloads"

        def propose_fn(state: AgentState) -> ProposedAction:
            return ProposedAction(
                tool_name="issue_refund",
                arguments={"ticket_id": state.ticket.id, "amount": "20.00", "currency": "USD", "reason": "damaged item"},
            )

        tracer = Tracer()
        graph = build_agent_graph(propose_fn, tracer=tracer)
        graph.invoke(make_initial_state(ticket, run_id=run_id))

        events = {e.event_type: e for e in tracer.events_for_run(run_id)}
        assert events[EventType.PROPOSED].proposed_action.tool_name == "issue_refund"
        assert events[EventType.AUTHORIZED].authorization_result.decision is AuthorizationDecision.APPROVED
        assert events[EventType.AUTHORIZED].authorization_result.authorized_command is not None
        assert events[EventType.EXECUTED].execution_result.status is ExecutionStatus.SUCCESS
        assert events[EventType.EXECUTED].execution_result.verified is True
        assert events[EventType.RUN_FINISHED].run_status is RunStatus.COMPLETED


class TestTracingDeniedRun:
    def test_events_recorded_in_expected_order(self):
        """A refund over the limit is denied before execute_node ever runs, so no EXECUTED event exists."""
        ticket = make_ticket(refund_eligible_amount=Decimal("50.00"))
        run_id = "run-denied"

        def propose_fn(state: AgentState) -> ProposedAction:
            return ProposedAction(
                tool_name="issue_refund",
                arguments={
                    "ticket_id": state.ticket.id,
                    "amount": "999.00",
                    "currency": "USD",
                    "reason": "damaged item",
                },
            )

        tracer = Tracer()
        graph = build_agent_graph(propose_fn, tracer=tracer)
        result = graph.invoke(make_initial_state(ticket, run_id=run_id))
        final_state = AgentState.model_validate(result)

        assert final_state.run_status is RunStatus.DENIED

        events = tracer.events_for_run(run_id)
        assert [e.event_type for e in events] == [
            EventType.PROPOSED,
            EventType.AUTHORIZED,
            EventType.RUN_FINISHED,
        ]
        assert [e.sequence for e in events] == [0, 1, 2]

    def test_event_payloads_match_the_denial(self):
        ticket = make_ticket(refund_eligible_amount=Decimal("50.00"))
        run_id = "run-denied-payloads"

        def propose_fn(state: AgentState) -> ProposedAction:
            return ProposedAction(
                tool_name="issue_refund",
                arguments={
                    "ticket_id": state.ticket.id,
                    "amount": "999.00",
                    "currency": "USD",
                    "reason": "damaged item",
                },
            )

        tracer = Tracer()
        graph = build_agent_graph(propose_fn, tracer=tracer)
        graph.invoke(make_initial_state(ticket, run_id=run_id))

        events = {e.event_type: e for e in tracer.events_for_run(run_id)}
        assert events[EventType.AUTHORIZED].authorization_result.decision is AuthorizationDecision.DENIED
        assert events[EventType.AUTHORIZED].authorization_result.rule_id == "refund-amount-limit"
        assert events[EventType.AUTHORIZED].authorization_result.authorized_command is None
        assert events[EventType.RUN_FINISHED].run_status is RunStatus.DENIED


class TestTracerIsolationAcrossRuns:
    def test_two_runs_through_the_same_graph_have_independent_traces(self):
        """A single shared Tracer must keep separately-run_id'd events apart, not interleave them."""
        approved_ticket = make_ticket(id="T-approved", refund_eligible_amount=Decimal("50.00"))
        denied_ticket = make_ticket(id="T-denied", refund_eligible_amount=Decimal("50.00"))

        def make_propose_fn(amount: str):
            def propose_fn(state: AgentState) -> ProposedAction:
                return ProposedAction(
                    tool_name="issue_refund",
                    arguments={
                        "ticket_id": state.ticket.id,
                        "amount": amount,
                        "currency": "USD",
                        "reason": "damaged item",
                    },
                )

            return propose_fn

        tracer = Tracer()
        approved_graph = build_agent_graph(make_propose_fn("20.00"), tracer=tracer)
        denied_graph = build_agent_graph(make_propose_fn("999.00"), tracer=tracer)

        approved_graph.invoke(make_initial_state(approved_ticket, run_id="run-a"))
        denied_graph.invoke(make_initial_state(denied_ticket, run_id="run-b"))

        assert [e.event_type for e in tracer.events_for_run("run-a")] == [
            EventType.PROPOSED,
            EventType.AUTHORIZED,
            EventType.EXECUTED,
            EventType.RUN_FINISHED,
        ]
        assert [e.event_type for e in tracer.events_for_run("run-b")] == [
            EventType.PROPOSED,
            EventType.AUTHORIZED,
            EventType.RUN_FINISHED,
        ]
        assert set(tracer.run_ids()) == {"run-a", "run-b"}


class TestTracerDefaultsToFreshInstance:
    def test_no_tracer_argument_still_works_without_error(self):
        """Not passing `tracer` must not break the graph — build_agent_graph() defaults to a fresh internal Tracer."""
        ticket = make_ticket()

        def propose_fn(state: AgentState) -> ProposedAction:
            return ProposedAction(
                tool_name="add_ticket_note", arguments={"ticket_id": state.ticket.id, "note": "Handled."}
            )

        graph = build_agent_graph(propose_fn)
        result = graph.invoke(make_initial_state(ticket))
        final_state = AgentState.model_validate(result)

        assert final_state.run_status is RunStatus.COMPLETED
