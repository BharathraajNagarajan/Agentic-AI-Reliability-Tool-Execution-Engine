"""Tests for the standalone MCP server (src.mcp.server) via an in-memory MCP client session.

Uses `mcp.shared.memory.create_connected_server_and_client_session` — the
MCP SDK's own in-process client/server wiring over memory streams — so
every test drives the server through the real MCP protocol (list_tools/
call_tool, including the SDK's own inputSchema validation) without any
process, socket, or stdio transport involved.

No `pytest-asyncio`/`anyio` pytest plugin is installed in this project, so
each test's async body runs via a plain `anyio.run(...)` call from an
ordinary synchronous `def test_...():` function, rather than an
`async def` test decorated with a marker.
"""

from __future__ import annotations

from decimal import Decimal

import anyio
from mcp.shared.memory import create_connected_server_and_client_session

from src.mcp.server import build_mcp_server
from src.mcp.store import build_default_store
from src.tools.schemas import AddTicketNoteArgs, IssueRefundArgs, UpdateTicketStatusArgs


class TestListTools:
    def test_lists_three_tools_with_schemas_matching_args_models(self):
        async def scenario() -> None:
            server = build_mcp_server()
            async with create_connected_server_and_client_session(server) as client:
                result = await client.list_tools()
                tools_by_name = {tool.name: tool for tool in result.tools}

                assert set(tools_by_name) == {"update_ticket_status", "issue_refund", "add_ticket_note"}
                assert tools_by_name["update_ticket_status"].inputSchema == UpdateTicketStatusArgs.model_json_schema()
                assert tools_by_name["issue_refund"].inputSchema == IssueRefundArgs.model_json_schema()
                assert tools_by_name["add_ticket_note"].inputSchema == AddTicketNoteArgs.model_json_schema()

        anyio.run(scenario)


class TestCallToolIssueRefund:
    def test_updates_seeded_ticket_readable_back_from_store(self):
        async def scenario() -> None:
            store = build_default_store()
            server = build_mcp_server(store)
            async with create_connected_server_and_client_session(server) as client:
                result = await client.call_tool(
                    "issue_refund",
                    {"ticket_id": "T-1", "amount": "20.00", "currency": "USD", "reason": "damaged item"},
                )

                assert result.isError is not True
                updated_ticket = store.get("T-1")
                assert updated_ticket is not None
                assert updated_ticket.refund_eligible_amount == Decimal("30.00")

        anyio.run(scenario)


class TestCallToolUpdateTicketStatus:
    def test_updates_seeded_ticket_readable_back_from_store(self):
        async def scenario() -> None:
            store = build_default_store()
            server = build_mcp_server(store)
            async with create_connected_server_and_client_session(server) as client:
                result = await client.call_tool(
                    "update_ticket_status",
                    {"ticket_id": "T-1", "new_status": "in_progress", "reason": "starting work"},
                )

                assert result.isError is not True
                updated_ticket = store.get("T-1")
                assert updated_ticket is not None
                assert updated_ticket.status.value == "in_progress"

        anyio.run(scenario)


class TestCallToolAddTicketNote:
    def test_updates_seeded_ticket_readable_back_from_store(self):
        async def scenario() -> None:
            store = build_default_store()
            server = build_mcp_server(store)
            async with create_connected_server_and_client_session(server) as client:
                result = await client.call_tool(
                    "add_ticket_note", {"ticket_id": "T-2", "note": "Called customer back."}
                )

                assert result.isError is not True
                updated_ticket = store.get("T-2")
                assert updated_ticket is not None
                assert updated_ticket.notes == ["Called customer back."]

        anyio.run(scenario)


class TestCallToolMalformedArguments:
    def test_missing_required_field_is_rejected_not_crashed(self):
        async def scenario() -> None:
            server = build_mcp_server()
            async with create_connected_server_and_client_session(server) as client:
                result = await client.call_tool("issue_refund", {"ticket_id": "T-1"})
                assert result.isError is True

        anyio.run(scenario)

    def test_wrong_argument_type_is_rejected_not_crashed(self):
        async def scenario() -> None:
            server = build_mcp_server()
            async with create_connected_server_and_client_session(server) as client:
                result = await client.call_tool(
                    "issue_refund",
                    {"ticket_id": "T-1", "amount": "not-a-number", "currency": "USD", "reason": "x"},
                )
                assert result.isError is True

        anyio.run(scenario)

    def test_unknown_tool_name_is_rejected_not_crashed(self):
        async def scenario() -> None:
            server = build_mcp_server()
            async with create_connected_server_and_client_session(server) as client:
                result = await client.call_tool("delete_everything", {"ticket_id": "T-1"})
                assert result.isError is True

        anyio.run(scenario)

    def test_unknown_ticket_id_is_rejected_not_crashed(self):
        async def scenario() -> None:
            server = build_mcp_server()
            async with create_connected_server_and_client_session(server) as client:
                result = await client.call_tool(
                    "issue_refund",
                    {"ticket_id": "does-not-exist", "amount": "5.00", "currency": "USD", "reason": "x"},
                )
                assert result.isError is True

        anyio.run(scenario)
