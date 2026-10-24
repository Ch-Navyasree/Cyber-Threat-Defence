from src.planner.decision_planner import DecisionPlanner
from src.safety.response_policy import ResponsePolicy


def test_benign_traffic():
    analysis = {
        "detection": {
            "predicted_attack": "BENIGN",
            "confidence": 0.99,
        },
        "investigation": {
            "investigation_status": "benign_prediction",
        },
    }

    plan = DecisionPlanner().plan(analysis)
    safety = ResponsePolicy().evaluate(plan)

    assert plan["decision"] == "MONITOR"
    assert safety["approved"] is True
    assert safety["action_permitted"] is False


def test_suspicious_traffic():
    analysis = {
        "detection": {
            "predicted_attack": "DDoS",
            "confidence": 0.99,
        },
        "investigation": {
            "investigation_status": "requires_investigation",
        },
    }

    plan = DecisionPlanner().plan(analysis)
    safety = ResponsePolicy().evaluate(plan)

    assert plan["decision"] == "INVESTIGATE"
    assert safety["approved"] is True
    assert safety["action_permitted"] is False


if __name__ == "__main__":
    test_benign_traffic()
    test_suspicious_traffic()
    print("PASS: Both planner and safety-policy tests passed.")