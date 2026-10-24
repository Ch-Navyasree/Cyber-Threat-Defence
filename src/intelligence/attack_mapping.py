
"""CICIDS2017 to candidate MITRE ATT&CK mappings.

A dataset label alone does not prove that an ATT&CK technique occurred.
Candidate mappings must be validated against observed behavior.
"""

from typing import Any


ATTACK_MAPPING: dict[str, dict[str, Any]] = {
    "BENIGN": {
        "investigation_category": "normal_traffic",
        "description": "Traffic labelled benign by CICIDS2017.",
        "attack_suspected": False,
        "mitre_tactic_ids": [],
        "mitre_technique_ids": [],
        "mapping_status": "not_applicable",
        "evidence_required": [],
        "references": [],
    },
    "BOT": {
        "investigation_category": "bot_related_activity",
        "description": "Investigate automated or bot-related network behavior.",
        "attack_suspected": True,
        "mitre_tactic_ids": [],
        "mitre_technique_ids": [],
        "mapping_status": "unmapped_requires_investigation",
        "evidence_required": [
            "Traffic patterns and destination behavior",
            "Evidence of command-and-control or automated activity",
        ],
        "references": [],
    },
    "BRUTE_FORCE": {
        "investigation_category": "credential_guessing",
        "description": "Investigate repeated authentication or credential-guessing behavior.",
        "attack_suspected": True,
        "mitre_tactic_ids": ["TA0006"],
        "mitre_technique_ids": ["T1110"],
        "mapping_status": "candidate_requires_behavior_validation",
        "evidence_required": [
            "Repeated authentication attempts",
            "Evidence that attempts involve credential guessing",
        ],
        "references": [
            "https://attack.mitre.org/techniques/T1110/",
        ],
    },
    "DDoS": {
        "investigation_category": "distributed_denial_of_service",
        "description": "Investigate traffic consistent with distributed denial of service.",
        "attack_suspected": True,
        "mitre_tactic_ids": ["TA0040"],
        "mitre_technique_ids": ["T1498"],
        "mapping_status": "candidate_requires_behavior_validation",
        "evidence_required": [
            "Evidence of traffic from multiple sources or distributed attack behavior",
            "Evidence of attempted network-resource exhaustion",
        ],
        "references": [
            "https://attack.mitre.org/techniques/T1498/",
        ],
    },
    "DoS": {
        "investigation_category": "denial_of_service",
        "description": "Investigate traffic consistent with denial of service.",
        "attack_suspected": True,
        "mitre_tactic_ids": ["TA0040"],
        "mitre_technique_ids": ["T1498", "T1499"],
        "mapping_status": "candidate_requires_behavior_validation",
        "evidence_required": [
            "Determine whether the behavior targets network capacity or endpoint resources",
            "Validate the specific DoS subtype before selecting a technique",
        ],
        "references": [
            "https://attack.mitre.org/techniques/T1498/",
            "https://attack.mitre.org/techniques/T1499/",
        ],
    },
    "PORT_SCAN": {
        "investigation_category": "network_service_discovery",
        "description": "Investigate port scanning and service-probing behavior.",
        "attack_suspected": True,
        "mitre_tactic_ids": ["TA0007"],
        "mitre_technique_ids": ["T1046"],
        "mapping_status": "candidate_requires_behavior_validation",
        "evidence_required": [
            "Connections probing multiple ports or services",
            "Evidence consistent with network service discovery",
        ],
        "references": [
            "https://attack.mitre.org/techniques/T1046/",
        ],
    },
    "WEB_ATTACK": {
        "investigation_category": "web_application_attack",
        "description": "Investigate the specific web-application attack subtype.",
        "attack_suspected": True,
        "mitre_tactic_ids": [],
        "mitre_technique_ids": [],
        "mapping_status": "unmapped_requires_subtype",
        "evidence_required": [
            "Identify the web-attack subtype",
            "Inspect relevant HTTP requests and available application evidence",
        ],
        "references": [],
    },
}


def get_attack_mapping(label: str) -> dict[str, Any]:
    """Return a copy of the mapping for a supported CICIDS2017 label."""
    normalized = label.strip().upper()
    normalized = {
        "DDOS": "DDoS",
        "DOS": "DoS",
    }.get(normalized, normalized)

    if normalized not in ATTACK_MAPPING:
        supported = ", ".join(ATTACK_MAPPING)
        raise ValueError(
            f"Unknown attack label: {label!r}. "
            f"Supported labels: {supported}"
        )

    return dict(ATTACK_MAPPING[normalized])


def get_supported_labels() -> list[str]:
    """Return the supported CICIDS2017 labels."""
    return list(ATTACK_MAPPING.keys())
