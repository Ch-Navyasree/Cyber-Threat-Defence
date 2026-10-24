from src.agents.cyber_defense_workflow import CyberDefenseWorkflow
from src.planner.decision_planner import DecisionPlanner


def test_benign_workflow():
    workflow = CyberDefenseWorkflow()
    result = workflow.analyze(_load_sample("BENIGN"))

    assert result["analysis"]["detection"]["predicted_attack"] == "BENIGN"
    assert result["analysis"]["mcp_knowledge"]["status"] == "found"
    assert result["plan"]["decision"] == "MONITOR"
    assert result["plan"]["mcp_status"] == "found"
    assert result["safety"]["approved"] is True
    assert result["verification"]["verified"] is True
    assert result["action_taken"] is False


def test_ddos_workflow():
    workflow = CyberDefenseWorkflow()
    result = workflow.analyze(_load_sample("DDoS"))

    assert result["analysis"]["detection"]["predicted_attack"] == "DDoS"
    assert result["analysis"]["mcp_knowledge"]["status"] == "found"
    assert result["plan"]["decision"] == "INVESTIGATE"
    assert result["plan"]["mcp_status"] == "found"
    assert result["safety"]["approved"] is True
    assert result["verification"]["verified"] is True
    assert result["action_taken"] is False


def test_planner_with_mcp_outcomes():
    planner = DecisionPlanner()

    base_analysis = {
        "detection": {
            "predicted_attack": "DDoS",
            "confidence": 0.99,
        },
        "investigation": {
            "investigation_status": "requires_investigation",
        },
    }

    cases = [
        {
            "status": "found",
            "knowledge": {
                "category": "distributed_denial_of_service"
            },
            "expected_text": "threat knowledge was retrieved",
        },
        {
            "status": "not_found",
            "expected_text": "knowledge was not found",
        },
        {
            "status": "error",
            "message": "Simulated connection failure",
            "expected_text": "retrieval failed",
        },
    ]

    for case in cases:
        analysis = {
            **base_analysis,
            "mcp_knowledge": case,
        }
        plan = planner.plan(analysis)

        assert plan["decision"] == "INVESTIGATE"
        assert plan["mcp_status"] == case["status"]
        assert case["expected_text"] in plan["reason"]
        assert plan["action_taken"] is False


def _load_sample(label):
    import pandas as pd

    path = "data/processed/transformed/selected_v2/validation.parquet"
    df = pd.read_parquet(path)
    row = df[df["Attack_Type"] == label].iloc[0]
    return row.drop(labels=["Attack_Type"]).to_dict()


if __name__ == "__main__":
    test_benign_workflow()
    print("PASS: Benign workflow")

    test_ddos_workflow()
    print("PASS: DDoS workflow")

    test_planner_with_mcp_outcomes()
    print("PASS: Planner handles all MCP outcomes")

    print("PASS: All end-to-end and planner tests passed.")