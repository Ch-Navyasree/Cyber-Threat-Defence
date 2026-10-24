"""Integrate NF-UNSW binary detection with local threat intelligence."""

from typing import Any

from src.agents.nf_unsw_investigation_tool import (
    NFUNSWInvestigationTool,
)
from src.intelligence.threat_intelligence_agent import investigate_threat


class NFUNSWDetectionIntegration:
    """Connect binary detection to evidence-based investigation."""

    def __init__(self):
        self.investigation_tool = NFUNSWInvestigationTool()

    def analyze(self, raw_features: dict) -> dict[str, Any]:
        """Analyze one flow and return detection and investigation results."""

        # Step 1: Classify the network flow.
        detection = self.investigation_tool.investigate(raw_features)

        predicted_label = detection["finding"]
        confidence = (
            detection["attack_probability"]
            if predicted_label == "Attack"
            else detection["benign_probability"]
        )

        # Step 2: Do not map a binary Attack prediction to a
        # specific attack category without supporting evidence.
        if predicted_label == "Benign":
            investigation = investigate_threat(
                predicted_label="BENIGN",
                confidence=confidence,
            )
        else:
            investigation = {
                "predicted_label": "Attack",
                "confidence": confidence,
                "investigation_status": "requires_investigation",
                "attack_suspected": True,
                "category": "unknown",
                "description": (
                    "The binary detector flagged this flow as an attack. "
                    "The specific attack category has not been established."
                ),
                "mitre_tactic_ids": [],
                "mitre_technique_ids": [],
                "mapping_status": "unmapped_requires_investigation",
                "evidence_required": [
                    "Inspect supporting network-flow and security telemetry.",
                    "Correlate the event with other relevant events.",
                    "Validate any proposed MITRE ATT&CK mapping against "
                    "observed behavior.",
                ],
                "references": [],
                "recommended_next_step": (
                    "Collect supporting evidence and perform further "
                    "threat-intelligence investigation. Do not automatically "
                    "execute a defensive action."
                ),
            }

        # Step 3: Return both outputs without executing a response.
        return {
            "dataset": "NF-UNSW-NB15-v2",
            "detection": detection,
            "investigation": investigation,
            "action_taken": False,
        }