from src.intelligence.detection_integration import DetectionIntegration
from src.mcp_gateway.client import ThreatKnowledgeMCPClient
from src.planner.decision_planner import DecisionPlanner
from src.safety.response_policy import ResponsePolicy
from src.verification.workflow_verifier import WorkflowVerifier


class CyberDefenseWorkflow:
    """Coordinate detection, MCP threat lookup, planning, safety, and verification."""

    def __init__(self):
        self.integration = DetectionIntegration()
        self.mcp_client = ThreatKnowledgeMCPClient()
        self.planner = DecisionPlanner()
        self.safety_policy = ResponsePolicy()
        self.verifier = WorkflowVerifier()

    def analyze(self, network_flow):
        # 1. Detect and investigate the traffic.
        analysis = self.integration.analyze(network_flow)

        # 2. Retrieve threat knowledge through MCP.
        detection = analysis.get("detection", {})
        predicted_attack = detection.get("predicted_attack", "UNKNOWN")

        try:
            mcp_knowledge = self.mcp_client.lookup(predicted_attack)
        except Exception as exc:
            # Keep the workflow safe if the MCP lookup fails.
            mcp_knowledge = {
                "status": "error",
                "message": str(exc),
                "action_taken": False,
            }

        analysis["mcp_knowledge"] = mcp_knowledge

        # 3. Generate a recommendation using the existing planner.
        plan = self.planner.plan(analysis)

        # 4. Apply the existing safety checks.
        safety = self.safety_policy.evaluate(plan)

        # 5. Verify the decision and safety result.
        verification = self.verifier.verify(plan, safety)

        # 6. Record the workflow result.
        audit = self.verifier.record(
    verification=verification,
    plan=plan,
    safety=safety,
    mcp_knowledge=mcp_knowledge,
)

        return {
            "analysis": analysis,
            "plan": plan,
            "safety": safety,
            "verification": verification,
            "audit": audit,
            "action_taken": False,
        }