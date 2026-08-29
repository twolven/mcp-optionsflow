import pytest
from fastmcp import Client

from optionsflow.server import mcp


@pytest.mark.asyncio
async def test_only_legacy_public_tool_is_declared():
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert [tool.name for tool in tools] == ["analyze_basic_strategies"]
    assert {"symbol", "expiration_date", "width_pct"} <= set(tools[0].inputSchema["properties"])
