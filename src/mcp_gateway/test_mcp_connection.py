import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    project_root = Path.cwd()

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_gateway.server"],
        cwd=str(project_root),
        env=None,
    )

    print("Starting MCP server and connecting client...", flush=True)

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("PASS: MCP connection initialized.", flush=True)

            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            print("Available tools:", tool_names, flush=True)

            result = await session.call_tool(
                "lookup_threat_knowledge",
                {"label": "DDoS"},
            )

            print("Tool call error:", result.is_error, flush=True)
            for item in result.content:
                if hasattr(item, "text"):
                    print(item.text)

            if result.is_error:
                raise RuntimeError("MCP tool returned an error.")
            if "lookup_threat_knowledge" not in tool_names:
                raise RuntimeError("Expected MCP tool was not registered.")

            print("PASS: End-to-end MCP client test completed.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

