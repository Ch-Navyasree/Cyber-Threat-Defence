from typing import Any


class DecisionPlanner:
    """Create read-only recommendations using detection and MCP knowledge."""

    def plan(self, analysis: dict[str, Any]) -> dict[str, Any]:
        detection = analysis.get("detection", {})
        investigation = analysis.get("investigation", {})
        mcp = analysis.get("mcp_knowledge", {})

        attack = str(detection.get("predicted_attack", "UNKNOWN"))
        confidence = float(detection.get("confidence", 0.0))
        status = str(
            investigation.get("investigation_status", "unknown")
        )
        mcp_status = str(mcp.get("status", "not_provided"))

        knowledge = mcp.get("knowledge", {})
        category = (
            knowledge.get("category")
            if isinstance(knowledge, dict)
            else None
        )

        if attack == "UNKNOWN":
            decision = "MANUAL_REVIEW"
            priority = "HIGH"
            reason = "The predicted attack category is unavailable."

        elif attack == "BENIGN" and status == "benign_prediction":
            decision = "MONITOR"
            priority = "LOW"
            reason = "The detector classified the traffic as benign."

        else:
            decision = "INVESTIGATE"
            priority = "HIGH"

            if mcp_status == "found":
                reason = (
                    f"Detected {attack}; threat knowledge was retrieved "
                    f"(category: {category or 'unspecified'}). "
                    "Investigate before considering any response."
                )
            elif mcp_status == "not_found":
                reason = (
                    f"Detected {attack}, but matching threat knowledge "
                    "was not found. Investigate the available evidence."
                )
            elif mcp_status == "error":
                reason = (
                    f"Detected {attack}, but threat-knowledge retrieval "
                    "failed. Investigate using available detection evidence "
                    "and treat threat intelligence as unavailable."
                )
            else:
                reason = (
                    f"Detected {attack}, but threat-knowledge retrieval "
                    "status is unavailable or unconfirmed. Investigate "
                    "before considering any response."
                )

        return {
            "predicted_attack": attack,
            "confidence": confidence,
            "investigation_status": status,
            "mcp_status": mcp_status,
            "threat_category": category,
            "decision": decision,
            "priority": priority,
            "reason": reason,
            "action_taken": False,
        }