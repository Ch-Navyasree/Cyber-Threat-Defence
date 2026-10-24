import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ThreatKnowledgeMCPClient:
    """Call the local threat knowledge MCP server."""

    async def lookup_async(self, label: str) -> dict:
        project_root = Path(__file__).resolve().parents[2]

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp_gateway.server"],
            cwd=str(project_root),
            env=None,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(
                    "lookup_threat_knowledge",
                    {"label": label},
                )

                if result.is_error:
                    raise RuntimeError("MCP threat lookup failed.")

                for item in result.content:
                    if hasattr(item, "text"):
                        import json
                        return json.loads(item.text)

                raise RuntimeError("MCP tool returned no text result.")

    def lookup(self, label: str) -> dict:
        """Synchronous entry point for the existing workflow."""
        return asyncio.run(self.lookup_async(label))
