
from copy import deepcopy
from typing import Any


SUPPORTED_LABELS = {
    "BENIGN",
    "BOT",
    "BRUTE_FORCE",
    "DDoS",
    "DoS",
    "PORT_SCAN",
    "WEB_ATTACK",
}

ALLOWED_ACTIONS = {
    "MONITOR",
    "SIMULATE_TRAFFIC_FILTER",
    "SIMULATE_LOGIN_BLOCK",
    "SIMULATE_RATE_LIMIT",
    "SIMULATE_SOURCE_BLOCK",
    "SIMULATE_MANUAL_REVIEW",
}


# Illustrative rules applied only to the simulator's internal state.
DEFENSE_RULES = {
    "DDoS": {
        "action": "SIMULATE_RATE_LIMIT",
        "rule_name": "DDoS_RATE_LIMIT",
        "description": "Simulated rate limit for suspected DDoS traffic.",
    },
    "DoS": {
        "action": "SIMULATE_TRAFFIC_FILTER",
        "rule_name": "DOS_TRAFFIC_FILTER",
        "description": "Simulated filtering of suspected DoS events.",
    },
    "BRUTE_FORCE": {
        "action": "SIMULATE_LOGIN_BLOCK",
        "rule_name": "BRUTE_FORCE_LOGIN_GUARD",
        "description": "Simulated blocking of repeated login attempts.",
    },
    "PORT_SCAN": {
        "action": "SIMULATE_SOURCE_BLOCK",
        "rule_name": "PORT_SCAN_SOURCE_GUARD",
        "description": "Simulated blocking of a scanning source.",
    },
    "BOT": {
        "action": "SIMULATE_TRAFFIC_FILTER",
        "rule_name": "BOT_TRAFFIC_FILTER",
        "description": "Simulated filtering of suspected bot traffic.",
    },
    "WEB_ATTACK": {
        "action": "SIMULATE_MANUAL_REVIEW",
        "rule_name": "WEB_ATTACK_REVIEW",
        "description": "Flagged for analyst review; no automatic blocking.",
    },
}


class AttackSimulator:
    """Simulate defense responses without touching real systems."""

    def run(
        self,
        attack_label: str,
        predicted_label: str,
        event_count: int = 100,
        workflow_decision: str = "INVESTIGATE",
        action_permitted: bool = False,
        simulation_only: bool = True,
    ) -> dict[str, Any]:
        # Validate inputs.
        if attack_label not in SUPPORTED_LABELS:
            raise ValueError("Unsupported attack scenario.")

        if predicted_label not in SUPPORTED_LABELS | {"UNKNOWN"}:
            raise ValueError("Unsupported predicted label.")

        if not isinstance(event_count, int) or isinstance(event_count, bool):
            raise ValueError("event_count must be an integer.")

        if not 1 <= event_count <= 1000:
            raise ValueError("event_count must be between 1 and 1000.")

        if workflow_decision not in {
            "INVESTIGATE",
            "MONITOR",
            "SIMULATE_RESPONSE",
        }:
            raise ValueError("Unsupported workflow decision.")

        if not isinstance(action_permitted, bool):
            raise ValueError("action_permitted must be a boolean.")

        if not isinstance(simulation_only, bool):
            raise ValueError("simulation_only must be a boolean.")

        # All changes are restricted to internal simulation counters.
        state = {
            "events_total": event_count,
            "events_allowed": event_count,
            "events_filtered": 0,
            "events_rate_limited": 0,
            "events_login_blocked": 0,
            "events_source_blocked": 0,
            "events_manual_review": 0,
        }
        before = deepcopy(state)

        rule = DEFENSE_RULES.get(predicted_label)

        # Simulation authorization is separate from real-action permission.
        # This never authorizes real containment or changes the safety policy.
        simulation_authorized = (
            simulation_only
            and rule is not None
            and predicted_label not in {"BENIGN", "UNKNOWN"}
        )

        if simulation_authorized:
            action = rule["action"]
            rule_name = rule["rule_name"]
            description = rule["description"]
            outcome = "SIMULATED_ONLY"

        elif predicted_label in {"BENIGN", "UNKNOWN"}:
            action = "MONITOR"
            rule_name = "MONITOR_ONLY"
            description = (
                "No automated containment for benign or unknown traffic."
            )
            outcome = "MONITORING"

        else:
            action = "SIMULATE_MANUAL_REVIEW"
            rule_name = "INVESTIGATION_REQUIRED"
            description = (
                "Simulation-only response is disabled; investigation "
                "is required before considering a response."
            )
            outcome = "REVIEW_REQUIRED"

        if action not in ALLOWED_ACTIONS:
            raise ValueError("Action is not permitted by the simulator.")

        # Update simulated counters only.
        if action == "SIMULATE_TRAFFIC_FILTER":
            state["events_allowed"] = 0
            state["events_filtered"] = event_count

        elif action == "SIMULATE_LOGIN_BLOCK":
            state["events_allowed"] = 0
            state["events_login_blocked"] = event_count

        elif action == "SIMULATE_RATE_LIMIT":
            limited = event_count // 2
            state["events_allowed"] = event_count - limited
            state["events_rate_limited"] = limited

        elif action == "SIMULATE_SOURCE_BLOCK":
            state["events_allowed"] = 0
            state["events_source_blocked"] = event_count

        elif action == "SIMULATE_MANUAL_REVIEW":
            # Reviewed events remain allowed in this counter model.
            state["events_manual_review"] = event_count

        # Verify the simulated outcome and preserve the real-action safety rule.
        checks = {
            "event_counts_consistent": (
                state["events_allowed"]
                + state["events_filtered"]
                + state["events_rate_limited"]
                + state["events_login_blocked"]
                + state["events_source_blocked"]
                == state["events_total"]
                and state["events_manual_review"] <= state["events_total"]
            ),
            "response_matches_prediction": (
                (
                    action == rule["action"]
                    and rule_name == rule["rule_name"]
                )
                if simulation_authorized
                else action in {"MONITOR", "SIMULATE_MANUAL_REVIEW"}
            ),
            "simulation_action_valid": (
                simulation_authorized
                or action in {"MONITOR", "SIMULATE_MANUAL_REVIEW"}
            ),
            "real_response_remains_disabled": action_permitted is False,
            "monitor_preserves_events": (
                action != "MONITOR" or state == before
            ),
            "no_real_action_executed": True,
        }

        verified = all(checks.values())

        return {
            "simulation_status": "COMPLETED" if verified else "FAILED",
            "actual_dataset_label": attack_label,
            "predicted_label": predicted_label,
            "detection_match": attack_label == predicted_label,
            "workflow_decision": workflow_decision,
            "action_permitted": action_permitted,
            "simulation_only": simulation_only,
            "simulation_authorized": simulation_authorized,
            "action": action,
            "response": {
                "rule_name": rule_name,
                "description": description,
                "outcome": outcome,
                "simulated_rule_active": action != "MONITOR",
            },
            "before": before,
            "after": state,
            "verification": {
                "verified": verified,
                "checks": checks,
            },
            "real_action_executed": False,
            "notice": (
                "Simulation only. Rules and event counts exist only "
                "inside the simulator. No real network traffic, firewall "
                "rules, login service, or system settings were changed."
            ),
        }
