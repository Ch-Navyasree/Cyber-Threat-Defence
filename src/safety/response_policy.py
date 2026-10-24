class ResponsePolicy:
    """Apply safety rules to planned cybersecurity decisions."""

    ALLOWED_DECISIONS = {
        "MONITOR",
        "INVESTIGATE",
        "MANUAL_REVIEW",
    }

    def evaluate(self, plan: dict) -> dict:
        decision = plan.get("decision", "MANUAL_REVIEW")
        action_taken = plan.get("action_taken", False)

        if decision not in self.ALLOWED_DECISIONS:
            return {
                "approved": False,
                "reason": "Unknown decision type.",
                "action_permitted": False,
            }

        if action_taken is not False:
            return {
                "approved": False,
                "reason": "The planner must not execute response actions.",
                "action_permitted": False,
            }

        return {
            "approved": True,
            "reason": "Read-only decision passed safety checks.",
            "action_permitted": False,
        }