
from src.detection.predictor import ThreatPredictor
from src.intelligence.threat_intelligence_agent import investigate_threat


class DetectionIntegration:
    """Connect threat detection with threat-intelligence investigation."""

    def __init__(self):
        self.predictor = ThreatPredictor()

    def analyze(self, network_flow):
        # Step 1: Predict the attack category.
        prediction = self.predictor.predict(network_flow)

        # Step 2: Extract the first prediction.
        predicted_attack = str(
            prediction.iloc[0]["predicted_attack"]
        )
        confidence = float(
            prediction.iloc[0]["confidence"]
        )

        # Step 3: Investigate the predicted category.
        investigation = investigate_threat(
            predicted_label=predicted_attack,
            confidence=confidence,
        )

        # Step 4: Return both results.
        return {
            "detection": {
                "predicted_attack": predicted_attack,
                "confidence": confidence,
            },
            "investigation": investigation,
        }
