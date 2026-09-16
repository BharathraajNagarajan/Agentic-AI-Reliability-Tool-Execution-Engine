"""End-to-end tests for the compiled agent graph (src.graph.build.build_agent_graph).

Each test builds a graph with a deterministic fake `propose_fn`, invokes
it once against a hand-constructed `AgentState`, and checks the final
`run_status` plus the relevant result field. `compiled.invoke(...)`
returns a plain dict (LangGraph's convention for a pydantic state schema),
so tests validate it back into an `AgentState` for typed access.
"""

from datetime import datetime, timezone
from decimal import Decimal

from src.execution import executor as executor_module
from src.execution.executor import ExecutionWrapper
from src.graph.build import build_agent_graph
from src.graph.state import AgentState, AuthorizationDecision, ExecutionStatus, RunStatus
from src.tools.schemas import AuthorizedUpdateTicketStatusCommand, ProposedAction, Ticket, TicketStatus


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


def make_initial_state(ticket: Ticket, task_description: str = "Handle this ticket.") -> AgentState:
    return AgentState(run_id="run-1", ticket=ticket, task_description=task_description)


class TestApprovedAndVerifiedHappyPath:
    def test_ends_in_completed(self):
        ticket = make_ticket(refund_eligible_amount=Decimal("50.00"))

        def propose_fn(state: AgentState) -> ProposedAction:
            return ProposedAction(
                tool_name="issue_refund",
                arguments={"ticket_id": state.ticket.id, "amount": "20.00", "currency": "USD", "reason": "damaged item"},
            )

        graph = build_agent_graph(propose_fn)
        result = graph.invoke(make_initial_state(ticket))
        final_state = AgentState.model_validate(result)

        assert final_state.run_status is RunStatus.COMPLETED
        assert final_state.authorization_result.decision is AuthorizationDecision.APPROVED
        assert final_state.execution_result.status is ExecutionStatus.SUCCESS
        assert final_state.execution_result.verified is True
        assert final_state.execution_result.output["ticket"]["refund_eligible_amount"] == "30.00"


class TestPolicyDeniedPath:
    def test_refund_over_limit_ends_in_denied(self):
        ticket = make_ticket(refund_eligible_amount=Decimal("50.00"))

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

        graph = build_agent_graph(propose_fn)
        result = graph.invoke(make_initial_state(ticket))
        final_state = AgentState.model_validate(result)

        assert final_state.run_status is RunStatus.DENIED
        assert final_state.authorization_result.decision is AuthorizationDecision.DENIED
        assert final_state.authorization_result.rule_id == "refund-amount-limit"
        assert final_state.execution_result is None  # execute_node never ran


class TestVerificationFailedPath:
    def test_verification_failure_ends_in_failed(self, monkeypatch):
        """Force a genuine verification mismatch via the buggy-tool-function-stub pattern.

        The stub simulates a tool function that runs without error but does
        NOT actually apply the authorized status change, matching the
        pattern used in tests/test_execution_executor.py.
        """
        ticket = make_ticket(status=TicketStatus.OPEN)

        def propose_fn(state: AgentState) -> ProposedAction:
            return ProposedAction(
                tool_name="update_ticket_status",
                arguments={"ticket_id": state.ticket.id, "new_status": "in_progress", "reason": "starting work"},
            )

        def buggy_update_ticket_status(cmd, tkt):
            return tkt.model_copy()  # status left untouched: does not reflect cmd.args.new_status

        monkeypatch.setitem(
            executor_module._TOOL_FUNCTIONS, AuthorizedUpdateTicketStatusCommand, buggy_update_ticket_status
        )

        graph = build_agent_graph(propose_fn, execution_wrapper=ExecutionWrapper())
        result = graph.invoke(make_initial_state(ticket))
        final_state = AgentState.model_validate(result)

        assert final_state.run_status is RunStatus.FAILED
        assert final_state.authorization_result.decision is AuthorizationDecision.APPROVED
        assert final_state.execution_result.status is ExecutionStatus.VERIFICATION_FAILED
        assert final_state.execution_result.verified is False
        assert final_state.execution_result.error is not None


class TestExecutionWrapperReuse:
    def test_provided_execution_wrapper_records_idempotency(self):
        """Passing an ExecutionWrapper in lets a caller inspect its idempotency store after invoke()."""
        ticket = make_ticket()

        def propose_fn(state: AgentState) -> ProposedAction:
            return ProposedAction(
                tool_name="add_ticket_note", arguments={"ticket_id": state.ticket.id, "note": "Handled via graph."}
            )

        wrapper = ExecutionWrapper()
        graph = build_agent_graph(propose_fn, execution_wrapper=wrapper)
        result = graph.invoke(make_initial_state(ticket))
        final_state = AgentState.model_validate(result)

        assert final_state.run_status is RunStatus.COMPLETED
        assert len(wrapper.idempotency_store) == 1
