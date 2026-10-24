"""Retrieve threat investigation knowledge from the local JSON knowledge base."""

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT / "data" / "threat_intelligence" / "threat_knowledge_base.json"
)


def load_knowledge_base() -> dict[str, Any]:
    """Load and validate the basic structure of the threat knowledge base."""
    if not KNOWLEDGE_BASE_PATH.is_file():
        raise FileNotFoundError(
            f"Threat knowledge base not found: {KNOWLEDGE_BASE_PATH}"
        )

    with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as file:
        knowledge_base = json.load(file)

    if not isinstance(knowledge_base.get("attack_categories"), list):
        raise ValueError(
            "Knowledge base must contain an 'attack_categories' list."
        )

    return knowledge_base


def retrieve_threat_knowledge(label: str) -> dict[str, Any]:
    """Retrieve the knowledge entry matching a CICIDS2017 attack label."""
    normalized_label = label.strip().upper()
    normalized_label = {
        "DDOS": "DDoS",
        "DOS": "DoS",
    }.get(normalized_label, normalized_label)

    knowledge_base = load_knowledge_base()

    for entry in knowledge_base["attack_categories"]:
        if entry.get("label") == normalized_label:
            return dict(entry)

    return {
        "label": label,
        "category": "unknown",
        "description": "No matching threat knowledge was found.",
        "attack_suspected": None,
        "mitre_tactic_ids": [],
        "mitre_technique_ids": [],
        "evidence_required": [
            "Verify the predicted label and inspect supporting telemetry."
        ],
        "references": [],
        "retrieval_status": "not_found",
    }