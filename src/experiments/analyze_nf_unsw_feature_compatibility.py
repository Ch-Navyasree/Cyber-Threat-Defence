from pathlib import Path
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = ROOT / "models" / "transformer_detector_2000_best_with_features.pt"
REFERENCE = ROOT / "data" / "raw" / "NF-UNSW-NB15-v2" / "NetFlow_v2_Features.csv"
OUTPUT = ROOT / "reports" / "nf_unsw_feature_compatibility.csv"

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

model_features = checkpoint["feature_names"]
reference = pd.read_csv(REFERENCE)

# These are candidates for investigation, not confirmed equivalences.
candidate_map = {
    "Destination Port": ("L4_DST_PORT", "Candidate: destination port"),
    "Flow Duration": ("FLOW_DURATION_MILLISECONDS", "Candidate: check units"),
    "Total Fwd Packets": ("IN_PKTS", "Candidate: verify direction"),
    "Total Backward Packets": ("OUT_PKTS", "Candidate: verify direction"),
    "Total Length of Fwd Packets": ("IN_BYTES", "Candidate: verify direction"),
    "Total Length of Bwd Packets": ("OUT_BYTES", "Candidate: verify direction"),
    "Fwd Packet Length Max": ("LONGEST_FLOW_PKT", "Partial candidate: direction may differ"),
    "Bwd Packet Length Min": ("SHORTEST_FLOW_PKT", "Partial candidate: direction may differ"),
    "Min Packet Length": ("MIN_IP_PKT_LEN", "Partial candidate: definition may differ"),
    "Max Packet Length": ("MAX_IP_PKT_LEN", "Partial candidate: definition may differ"),
    "Flow Bytes/s": ("SRC_TO_DST_SECOND_BYTES; DST_TO_SRC_SECOND_BYTES", "Requires formula and unit validation"),
    "Flow Packets/s": ("IN_PKTS; OUT_PKTS; FLOW_DURATION_MILLISECONDS", "Potentially derived; validate definitions"),
    "FIN Flag Count": ("TCP_FLAGS; CLIENT_TCP_FLAGS; SERVER_TCP_FLAGS", "Not equivalent without flag semantics"),
}

reference_features = set(reference["Feature"].astype(str).str.strip())
rows = []

for feature in model_features:
    if feature in candidate_map:
        external, assessment = candidate_map[feature]
        rows.append({
            "CICIDS2017_model_feature": feature,
            "External_candidate": external,
            "Assessment": assessment,
        })
    else:
        rows.append({
            "CICIDS2017_model_feature": feature,
            "External_candidate": "",
            "Assessment": "No direct candidate identified; inspect definitions",
        })

report = pd.DataFrame(rows)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(OUTPUT, index=False)

print(f"Model features checked: {len(model_features)}")
print(f"External reference entries: {len(reference_features)}")
print(f"Report saved: {OUTPUT}")
print("\nCompatibility report:")
print(report.to_string(index=False))
print("\nReminder: candidate matches are not proof of feature equivalence.")
