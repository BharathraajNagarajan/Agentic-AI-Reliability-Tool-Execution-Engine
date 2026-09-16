"""A ProposeFn that calls the Anthropic API to produce a ProposedAction.

The `anthropic` SDK is an optional runtime dependency (see pyproject.toml)
and is imported ONLY inside `AnthropicProposeProvider._get_client()` —
never at module level — so importing this module, constructing a
provider, and unit-testing `_parse_tool_use_response()` all work with
`anthropic` uninstalled. Only actually invoking a provider that was not
given an injected `client` touches the SDK.

The API key is never read or stored by this module directly: it is left
to the `anthropic` SDK's own default behavior of reading the
`ANTHROPIC_API_KEY` environment variable when no key is passed to
`anthropic.Anthropic()`. Nothing here hardcodes or logs a key.
"""

from __future__ import annotations

from typing import Any

from src.graph.state import AgentState
from src.tools.schemas import AddTicketNoteArgs, IssueRefundArgs, ProposedAction, ToolName, UpdateTicketStatusArgs

DEFAULT_MODEL = "claude-haiku-4-5"
"""Cheapest current Claude model suitable for this tool-selection task."""


def _default_tool_definitions() -> list[dict]:
    """Anthropic tool-use definitions for the three known tools, derived from their argument schemas.

    Built from `UpdateTicketStatusArgs`/`IssueRefundArgs`/`AddTicketNoteArgs`'s
    own `model_json_schema()` rather than hand-duplicated JSON, so the tool
    definitions the LLM sees never drift from the argument shapes the
    policy layer will actually validate against.
    """
    return [
        {
            "name": ToolName.UPDATE_TICKET_STATUS.value,
            "description": "Change a support ticket's status.",
            "input_schema": UpdateTicketStatusArgs.model_json_schema(),
        },
        {
            "name": ToolName.ISSUE_REFUND.value,
            "description": "Issue a refund against a support ticket's order.",
            "input_schema": IssueRefundArgs.model_json_schema(),
        },
        {
            "name": ToolName.ADD_TICKET_NOTE.value,
            "description": "Attach an internal note to a support ticket.",
            "input_schema": AddTicketNoteArgs.model_json_schema(),
        },
    ]


def _build_prompt(state: AgentState) -> str:
    """Build a plain-text prompt describing the ticket, task, and conversation so far."""
    lines = [
        f"Ticket ID: {state.ticket.id}",
        f"Status: {state.ticket.status.value}",
        f"Priority: {state.ticket.priority.value}",
        f"Subject: {state.ticket.subject}",
    ]
    if state.ticket.refund_eligible_amount is not None:
        lines.append(f"Refund-eligible amount: {state.ticket.refund_eligible_amount} {state.ticket.currency}")
    if state.ticket.notes:
        lines.append("Existing notes:")
        lines.extend(f"  - {note}" for note in state.ticket.notes)
    lines.append(f"Task: {state.task_description}")
    if state.conversation_history:
        lines.append("Conversation so far:")
        lines.extend(f"  [{message.role}] {message.content}" for message in state.conversation_history)
    lines.append("Decide on exactly one tool call to make progress on this ticket.")
    return "\n".join(lines)


def _parse_tool_use_response(message: Any) -> ProposedAction:
    """Parse an Anthropic Message-like object into a ProposedAction.

    Duck-typed rather than SDK-typed: `message.content` is expected to be
    an iterable of blocks, each exposing `.type`, and a `tool_use` block
    additionally exposing `.name` (str) and `.input` (dict) — this matches
    the real `anthropic` SDK's response shape without importing its types.

    Never raises. Anything that doesn't look like a well-formed tool_use
    block — no content, an empty list, an unexpected block type, a
    tool_use block with a non-dict `.input`, or `message` itself being
    `None` or some other unrelated object — falls through to a
    `ProposedAction` with an empty `tool_name`, preserving
    `ProposedAction`'s "untrusted, tolerant" contract: parsing failures
    here are the policy layer's problem to reject downstream, not this
    function's to raise on.
    """
    raw_text_parts: list[str] = []
    for block in getattr(message, "content", None) or []:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            name = getattr(block, "name", None)
            tool_input = getattr(block, "input", None)
            if isinstance(name, str) and isinstance(tool_input, dict):
                return ProposedAction(tool_name=name, arguments=tool_input, raw_response=repr(message))
        elif block_type == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str):
                raw_text_parts.append(text)

    return ProposedAction(tool_name="", arguments={}, raw_response="\n".join(raw_text_parts) or repr(message))


class AnthropicProposeProvider:
    """A `ProposeFn` (`Callable[[AgentState], ProposedAction]`) backed by the Anthropic API.

    Pass `client` to inject a fake/stand-in object exposing
    `.messages.create(...)` for offline testing; leave it unset in
    production to lazily construct a real `anthropic.Anthropic()` client
    (which reads `ANTHROPIC_API_KEY` from the environment) the first time
    this provider is actually called.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._tools = tools if tools is not None else _default_tool_definitions()
        self._max_tokens = max_tokens
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import anthropic  # deferred: only touches the SDK when actually needed

        self._client = anthropic.Anthropic()
        return self._client

    def __call__(self, state: AgentState) -> ProposedAction:
        client = self._get_client()
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            tools=self._tools,
            messages=[{"role": "user", "content": _build_prompt(state)}],
        )
        return _parse_tool_use_response(message)
