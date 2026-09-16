"""Tests for the scripted-sequence mock ProposeFn (src.graph.providers.mock).

Pure Python, no external SDK, no network — these tests must pass with
zero API key and zero network access.
"""

from datetime import datetime, timezone

import pytest

from src.graph.providers.mock import ScriptedProposeProvider
from src.graph.state import AgentState
from src.tools.schemas import ProposedAction, Ticket, TicketStatus


def make_ticket() -> Ticket:
    return Ticket(
        id="T-1",
        status=TicketStatus.OPEN,
        customer_ref="cust-123",
        subject="Package arrived damaged",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def make_state() -> AgentState:
    return AgentState(run_id="run-1", ticket=make_ticket(), task_description="Handle this ticket.")


class TestScriptedProposeProvider:
    def test_returns_proposals_in_order(self):
        first = ProposedAction(tool_name="add_ticket_note", arguments={"ticket_id": "T-1", "note": "first"})
        second = ProposedAction(tool_name="add_ticket_note", arguments={"ticket_id": "T-1", "note": "second"})
        provider = ScriptedProposeProvider([first, second])

        state = make_state()
        assert provider(state) is first
        assert provider(state) is second

    def test_calls_made_tracks_consumption(self):
        proposal = ProposedAction(tool_name="add_ticket_note", arguments={})
        provider = ScriptedProposeProvider([proposal, proposal, proposal])

        state = make_state()
        assert provider.calls_made == 0
        provider(state)
        assert provider.calls_made == 1
        provider(state)
        assert provider.calls_made == 2

    def test_raises_index_error_when_script_exhausted(self):
        proposal = ProposedAction(tool_name="add_ticket_note", arguments={})
        provider = ScriptedProposeProvider([proposal])

        state = make_state()
        provider(state)
        with pytest.raises(IndexError):
            provider(state)

    def test_empty_script_raises_on_first_call(self):
        provider = ScriptedProposeProvider([])
        with pytest.raises(IndexError):
            provider(make_state())

    def test_conforms_to_proposefn_and_works_with_make_propose_node(self):
        """A ScriptedProposeProvider must be a drop-in ProposeFn for propose_node."""
        from src.graph.nodes import make_propose_node

        proposal = ProposedAction(tool_name="add_ticket_note", arguments={"ticket_id": "T-1", "note": "hi"})
        provider = ScriptedProposeProvider([proposal])
        node = make_propose_node(provider)

        updates = node(make_state())
        assert updates["proposed_action"] is proposal
