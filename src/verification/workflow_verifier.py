import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WorkflowVerifier:
    """Verify workflow results and record an audit trail."""

    def __init__(self, audit_path="reports/workflow_audit.jsonl"):
        self.audit_path = Path(audit_path)

    def verify(self, plan: dict, safety: dict) -> dict[str, Any]:
        errors = []

        allowed_decisions = {
            "MONITOR",
            "INVESTIGATE",
            "MANUAL_REVIEW",
        }

        decision = plan.get("decision")
        action_taken = plan.get("action_taken")
        approved = safety.get("approved")
        action_permitted = safety.get("action_permitted")

        if decision not in allowed_decisions:
            errors.append("Unknown or missing planner decision.")

        if action_taken is not False:
            errors.append("Unexpected response action recorded.")

        if approved is not True:
            errors.append("Safety policy did not approve the recommendation.")

        if action_permitted is not False:
            errors.append("Response execution must remain disabled.")

        return {
            "verified": len(errors) == 0,
            "errors": errors,
            "decision": decision,
            "response_execution_enabled": False,
        }

    def record(
        self,
        verification: dict,
        plan: dict,
        safety: dict,
        mcp_knowledge: dict | None = None,
    ) -> dict[str, Any]:
        """Append a workflow audit record, including MCP lookup status."""

        mcp_knowledge = mcp_knowledge or {}
        knowledge = mcp_knowledge.get("knowledge", {})

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "verification": verification,
            "plan": plan,
            "safety": safety,
            "mcp": {
                "status": mcp_knowledge.get("status", "not_provided"),
                "category": (
                    knowledge.get("category")
                    if isinstance(knowledge, dict)
                    else None
                ),
                "message": mcp_knowledge.get("message"),
            },
            "action_taken": False,
        }

        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

        with self.audit_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=str) + "\n")

        return {
            "recorded": True,
            "audit_file": str(self.audit_path),
            "verified": verification["verified"],
        }