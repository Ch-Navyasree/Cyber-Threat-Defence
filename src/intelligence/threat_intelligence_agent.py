"""Threat Intelligence Agent for the Agentic Cyber Defense project."""

from typing import Any

from src.intelligence.knowledge_retriever import retrieve_threat_knowledge


def investigate_threat(
    predicted_label: str,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Combine a detector prediction with retrieved threat knowledge.

    Args:
        predicted_label: Class predicted by the Transformer detector.
        confidence: Optional prediction confidence between 0 and 1.

    Returns:
        A structured threat-investigation result for downstream agents.
    """
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0.")

    knowledge = retrieve_threat_knowledge(predicted_label)

    if knowledge.get("retrieval_status") == "not_found":
        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "investigation_status": "knowledge_not_found",
            "attack_suspected": None,
            "category": "unknown",
            "description": knowledge["description"],
            "mitre_tactic_ids": [],
            "mitre_technique_ids": [],
            "evidence_required": knowledge["evidence_required"],
            "references": [],
            "recommended_next_step": (
                "Request further investigation; do not automatically execute "
                "a defensive action."
            ),
        }

    is_benign = predicted_label.strip().upper() == "BENIGN"

    if is_benign:
        status = "benign_prediction"
        next_step = "Record the prediction and continue monitoring."
    else:
        status = "requires_investigation"
        next_step = (
            "Validate the prediction against supporting telemetry before "
            "selecting a response."
        )

    return {
        "predicted_label": knowledge["label"],
        "confidence": confidence,
        "investigation_status": status,
        "attack_suspected": knowledge["attack_suspected"],
        "category": knowledge["category"],
        "description": knowledge["description"],
        "mitre_tactic_ids": knowledge["mitre_tactic_ids"],
        "mitre_technique_ids": knowledge["mitre_technique_ids"],
        "mapping_status": (
            "not_applicable"
            if is_benign
            else (
                "candidate_requires_behavior_validation"
                if knowledge["mitre_technique_ids"]
                else "unmapped_requires_investigation"
            )
        ),
        "evidence_required": knowledge["evidence_required"],
        "references": knowledge["references"],
        "recommended_next_step": next_step,
    }