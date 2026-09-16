"""A standalone MCP server exposing the three support-ticket tool functions.

Wraps `src.tools.functions.update_ticket_status`/`issue_refund`/
`add_ticket_note` — today only reachable from `execute_node` after
`src.policy.interface.authorize()` has approved a proposal — as MCP tools
an MCP client can call directly by name and arguments.

TRUST BOUNDARY, STATED EXPLICITLY: this server does NOT wire in
`src.policy` authorization. It validates that a tool call's arguments
match the shape the matching `*Args` model expects (structurally, via
each tool's `inputSchema`, and again via `model_validate` in the handler),
but it does not check refund limits, status-transition legality, or any
other business rule — it trusts that whatever called it has already
decided this action is authorized. That is what a tool-execution MCP
server does: it is the mechanism a calling agent's authorization decision
is carried out through, not the boundary that makes that decision. A
policy-authorizing caller (human, this project's own graph, or another
agent) is this server's caller's responsibility, not this module's.

Each tool's `inputSchema` is derived from the corresponding pydantic
`*Args` model via `model_json_schema()` — the same technique
`src.graph.providers.anthropic_provider._default_tool_definitions()`
already uses to build Anthropic tool-use schemas from the same `*Args`
models. That function's output is Anthropic-shaped (`input_schema`, a
list of plain dicts), not `mcp.types.Tool` (`inputSchema`), so reusing it
here would still require adapting its return shape; combined with it
being a private (underscore-prefixed) helper not intended for external
reuse, and this MCP server being meant to stand alone rather than import
from `src.graph` (a directory this phase does not touch), the three-line
`model_json_schema()` pattern is duplicated in `_tool_definitions()`
below rather than shared.

Tool calls need a `Ticket` to run against, which an MCP tool call has no
ambient way to supply — `src.mcp.store.TicketStore` (a new, standalone,
in-memory store scoped to this server only) fills that gap: each handler
looks up `Ticket` by the `ticket_id` its arguments carry, runs it through
the matching tool function, and writes the result back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import anyio
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from src.mcp.store import TicketStore, build_default_store
from src.tools.functions import add_ticket_note, issue_refund, update_ticket_status
from src.tools.schemas import (
    AddTicketNoteArgs,
    AuthorizedAddTicketNoteCommand,
    AuthorizedIssueRefundCommand,
    AuthorizedUpdateTicketStatusCommand,
    IssueRefundArgs,
    Ticket,
    ToolName,
    UpdateTicketStatusArgs,
)

SERVER_NAME = "support-ticket-ops-tools"


@dataclass(frozen=True)
class _ToolSpec:
    """Everything one MCP tool needs: its argument model, command model, and tool function."""

    description: str
    args_model: type
    command_model: type
    tool_function: Callable[[Any, Ticket], Ticket]


_TOOL_SPECS: dict[str, _ToolSpec] = {
    ToolName.UPDATE_TICKET_STATUS.value: _ToolSpec(
        description="Change a support ticket's status.",
        args_model=UpdateTicketStatusArgs,
        command_model=AuthorizedUpdateTicketStatusCommand,
        tool_function=update_ticket_status,
    ),
    ToolName.ISSUE_REFUND.value: _ToolSpec(
        description="Issue a refund against a support ticket's order.",
        args_model=IssueRefundArgs,
        command_model=AuthorizedIssueRefundCommand,
        tool_function=issue_refund,
    ),
    ToolName.ADD_TICKET_NOTE.value: _ToolSpec(
        description="Attach an internal note to a support ticket.",
        args_model=AddTicketNoteArgs,
        command_model=AuthorizedAddTicketNoteCommand,
        tool_function=add_ticket_note,
    ),
}


def _tool_definitions() -> list[types.Tool]:
    """Build the three MCP `Tool` definitions, one per `_TOOL_SPECS` entry.

    See this module's docstring for why this duplicates, rather than
    imports, `anthropic_provider.py`'s `model_json_schema()`-derived-schema
    pattern.
    """
    return [
        types.Tool(name=name, description=spec.description, inputSchema=spec.args_model.model_json_schema())
        for name, spec in _TOOL_SPECS.items()
    ]


def build_mcp_server(store: TicketStore | None = None) -> Server:
    """Build an MCP `Server` exposing `update_ticket_status`/`issue_refund`/`add_ticket_note`.

    `store` defaults to a fresh `build_default_store()` (seeded with `T-1`/
    `T-2`) if not given; passing one in lets a caller (e.g. a test) inspect
    ticket state after tool calls, or share a store across multiple
    servers/sessions.

    No authorization is performed here — see this module's docstring.
    """
    store = store if store is not None else build_default_store()
    server: Server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return _tool_definitions()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        spec = _TOOL_SPECS.get(name)
        if spec is None:
            raise ValueError(f"Unknown tool: {name!r}")

        args = spec.args_model.model_validate(arguments)
        ticket = store.get(args.ticket_id)
        if ticket is None:
            raise ValueError(f"No ticket found with id {args.ticket_id!r}")

        command = spec.command_model(args=args)
        updated_ticket = spec.tool_function(command, ticket)
        store.put(updated_ticket)

        return updated_ticket.model_dump(mode="json")

    return server


async def _run_stdio() -> None:
    """Run `build_mcp_server()` over stdio — the standard way an MCP client launches a local server."""
    server = build_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Entry point for running this server as a standalone process (`python -m src.mcp.server`)."""
    anyio.run(_run_stdio)


if __name__ == "__main__":
    main()
