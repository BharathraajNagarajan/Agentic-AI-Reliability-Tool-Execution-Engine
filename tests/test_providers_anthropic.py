"""Tests for the Anthropic-backed ProposeFn (src.graph.providers.anthropic_provider).

All tests here use a plain duck-typed stand-in for Anthropic SDK objects
(SimpleNamespace, or a hand-rolled fake client) rather than the real
`anthropic` package — none of them make a network call, require an API
key, or even require `anthropic` to be installed: `_parse_tool_use_response`
is a pure function, and `AnthropicProposeProvider` is exercised only with
an injected fake `client`, so its lazy `import anthropic` branch is never
reached.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

from src.graph.nodes import make_propose_node
from src.graph.providers.anthropic_provider import (
    AnthropicProposeProvider,
    _build_prompt,
    _default_tool_definitions,
    _parse_tool_use_response,
)
from src.graph.state import AgentState
from src.tools.schemas import (
    AddTicketNoteArgs,
    IssueRefundArgs,
    Ticket,
    TicketStatus,
    ToolName,
    UpdateTicketStatusArgs,
)


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


class TestParseToolUseResponseValid:
    def test_valid_tool_use_block_parses_correctly(self):
        message = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="issue_refund",
                    input={"ticket_id": "T-1", "amount": "20.00", "currency": "USD", "reason": "damaged item"},
                )
            ]
        )
        proposal = _parse_tool_use_response(message)
        assert proposal.tool_name == "issue_refund"
        assert proposal.arguments == {
            "ticket_id": "T-1",
            "amount": "20.00",
            "currency": "USD",
            "reason": "damaged item",
        }

    def test_leading_text_block_before_tool_use_is_still_parsed(self):
        """Real responses often include reasoning text before the tool_use block."""
        message = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="I'll add a note to this ticket."),
                SimpleNamespace(type="tool_use", name="add_ticket_note", input={"ticket_id": "T-1", "note": "hi"}),
            ]
        )
        proposal = _parse_tool_use_response(message)
        assert proposal.tool_name == "add_ticket_note"
        assert proposal.arguments == {"ticket_id": "T-1", "note": "hi"}


class TestParseToolUseResponseMalformed:
    """None of these should raise — a malformed/unexpected response still yields a ProposedAction."""

    def test_message_with_no_content_attribute(self):
        proposal = _parse_tool_use_response(object())
        assert proposal.tool_name == ""

    def test_message_is_none(self):
        proposal = _parse_tool_use_response(None)
        assert proposal.tool_name == ""

    def test_empty_content_list(self):
        proposal = _parse_tool_use_response(SimpleNamespace(content=[]))
        assert proposal.tool_name == ""

    def test_only_text_blocks_no_tool_use(self):
        message = SimpleNamespace(content=[SimpleNamespace(type="text", text="I am not going to call a tool.")])
        proposal = _parse_tool_use_response(message)
        assert proposal.tool_name == ""
        assert "not going to call a tool" in proposal.raw_response

    def test_tool_use_block_with_non_dict_input(self):
        message = SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name="issue_refund", input="not-a-dict")]
        )
        proposal = _parse_tool_use_response(message)
        assert proposal.tool_name == ""

    def test_tool_use_block_with_non_string_name(self):
        message = SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=None, input={"a": 1})])
        proposal = _parse_tool_use_response(message)
        assert proposal.tool_name == ""

    def test_unexpected_block_type_is_ignored_not_fatal(self):
        message = SimpleNamespace(content=[SimpleNamespace(type="thinking")])
        proposal = _parse_tool_use_response(message)
        assert proposal.tool_name == ""

    def test_never_raises_on_completely_broken_input(self):
        for broken in [None, object(), 42, "a string", SimpleNamespace(content="not-a-list-of-blocks")]:
            proposal = _parse_tool_use_response(broken)
            assert proposal.tool_name == ""


class TestDefaultToolDefinitions:
    def test_covers_all_three_known_tools_with_matching_schemas(self):
        definitions = {d["name"]: d for d in _default_tool_definitions()}
        assert set(definitions) == {tn.value for tn in ToolName}
        assert definitions["update_ticket_status"]["input_schema"] == UpdateTicketStatusArgs.model_json_schema()
        assert definitions["issue_refund"]["input_schema"] == IssueRefundArgs.model_json_schema()
        assert definitions["add_ticket_note"]["input_schema"] == AddTicketNoteArgs.model_json_schema()


class TestBuildPrompt:
    def test_prompt_includes_ticket_and_task_details(self):
        prompt = _build_prompt(make_state())
        assert "T-1" in prompt
        assert "Handle this ticket." in prompt


class FakeMessages:
    def __init__(self, response) -> None:
        self._response = response
        self.create_calls: list[dict] = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self._response


class FakeAnthropicClient:
    def __init__(self, response) -> None:
        self.messages = FakeMessages(response)


class TestAnthropicProposeProviderWithInjectedClient:
    def test_call_returns_parsed_proposed_action(self):
        fake_response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="add_ticket_note",
                    input={"ticket_id": "T-1", "note": "Called customer."},
                )
            ]
        )
        client = FakeAnthropicClient(fake_response)
        provider = AnthropicProposeProvider(client=client)

        proposal = provider(make_state())

        assert proposal.tool_name == "add_ticket_note"
        assert proposal.arguments == {"ticket_id": "T-1", "note": "Called customer."}

    def test_call_passes_model_and_tools_to_client(self):
        fake_response = SimpleNamespace(content=[])
        client = FakeAnthropicClient(fake_response)
        provider = AnthropicProposeProvider(client=client)

        provider(make_state())

        assert len(client.messages.create_calls) == 1
        call_kwargs = client.messages.create_calls[0]
        assert call_kwargs["model"] == "claude-haiku-4-5"
        assert {t["name"] for t in call_kwargs["tools"]} == {tn.value for tn in ToolName}
        assert call_kwargs["messages"][0]["role"] == "user"

    def test_conforms_to_proposefn_and_works_with_make_propose_node(self):
        """An AnthropicProposeProvider (with an injected client) must be a drop-in ProposeFn."""
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name="add_ticket_note", input={"ticket_id": "T-1", "note": "x"})]
        )
        provider = AnthropicProposeProvider(client=FakeAnthropicClient(fake_response))
        node = make_propose_node(provider)

        updates = node(make_state())

        assert updates["proposed_action"].tool_name == "add_ticket_note"

    def test_never_touches_real_anthropic_sdk_when_client_injected(self):
        """Constructing/calling with an injected client must not import the real anthropic package."""
        import sys

        was_already_imported = "anthropic" in sys.modules
        fake_response = SimpleNamespace(content=[])
        provider = AnthropicProposeProvider(client=FakeAnthropicClient(fake_response))
        provider(make_state())

        if not was_already_imported:
            assert "anthropic" not in sys.modules
