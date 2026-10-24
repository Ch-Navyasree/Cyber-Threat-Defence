from mcp.server.mcpserver import MCPServer

from src.intelligence.knowledge_retriever import (
    retrieve_threat_knowledge,
)


server = MCPServer(
    name="cyber-defense-gateway",
    version="1.0.0",
    instructions=(
        "Provides read-only access to the local cybersecurity "
        "threat knowledge base. Does not execute response actions."
    ),
)


@server.tool()
def lookup_threat_knowledge(label: str) -> dict:
    """Retrieve threat knowledge for a CICIDS2017 attack label.

    Supported labels include BENIGN, BOT, BRUTE_FORCE, DDoS,
    DoS, PORT_SCAN, and WEB_ATTACK.
    This tool only reads the knowledge base.
    """
    if not isinstance(label, str) or not label.strip():
        return {
            "status": "error",
            "message": "A non-empty attack label is required.",
        }

    try:
        knowledge = retrieve_threat_knowledge(label)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "status": "error",
            "message": str(exc),
        }

    return {
        "status": (
            "found"
            if knowledge.get("retrieval_status") != "not_found"
            else "not_found"
        ),
        "knowledge": knowledge,
        "action_taken": False,
    }


if __name__ == "__main__":
    server.run(transport="stdio")