import pytest
from fastmcp import Client

from optionsflow.server import mcp


@pytest.mark.asyncio
async def test_only_legacy_public_tool_is_declared():
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert [tool.name for tool in tools] == ["analyze_basic_strategies"]
    schema = tools[0].inputSchema
    assert {"symbol", "strategy", "expiration_date", "delta_target", "width_pct"} == set(
        schema["properties"]
    )
    assert schema["required"] == ["symbol", "strategy", "expiration_date"]
    assert schema["properties"]["strategy"]["enum"] == ["ccs", "pcs", "csp", "cc"]
    assert schema["properties"]["symbol"]["pattern"] == r"^[A-Za-z0-9.^=-]+$"
    assert {"success", "timestamp", "data", "provider", "warnings"} <= set(
        tools[0].outputSchema["properties"]
    )
