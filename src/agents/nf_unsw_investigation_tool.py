from pathlib import Path
import sys

# Locate the project and detector modules.
ROOT = Path(__file__).resolve().parents[2]
DETECTION_DIR = ROOT / "src" / "detection"

if str(DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_DIR))

from nf_unsw_predictor import NFUNSWPredictor


class NFUNSWInvestigationTool:
    """Read-only threat investigation tool."""

    def __init__(self):
        self.predictor = NFUNSWPredictor()

    def investigate(self, raw_features: dict) -> dict:
        """
        Analyze one network flow and return a structured result.
        This tool does not execute network actions.
        """

        prediction = self.predictor.predict_raw(raw_features)

        return {
            "tool": "nf_unsw_investigation",
            "status": "completed",
            "finding": prediction["predicted_class"],
            "attack_probability": prediction["attack_probability"],
            "benign_probability": prediction["benign_probability"],
            "threshold": prediction["threshold"],
            "dataset": prediction["dataset"],
            "model": prediction["model"],
            "checkpoint_epoch": prediction["checkpoint_epoch"],
            "recommended_next_step": (
                "Review evidence and threat intelligence"
                if prediction["predicted_class"] == "Attack"
                else "Continue monitoring"
            ),
            "action_taken": False,
        }