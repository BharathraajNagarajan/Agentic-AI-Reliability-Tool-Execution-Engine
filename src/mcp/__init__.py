"""MCP layer.

A standalone MCP server (`src.mcp.server`) exposing the three pure tool
functions in `src.tools.functions` as MCP tools, backed by its own
in-memory `TicketStore` (`src.mcp.store`). This server does NOT enforce
`src.policy` authorization — it trusts its caller's arguments are already
authorized; see `src.mcp.server`'s module docstring for the full trust-
boundary explanation. No MCP client exists yet.
"""
