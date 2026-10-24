from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detection"))
from nf_unsw_binary_detector import NFUNSWBinaryTransformer


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "nf_unsw" / "test.parquet"
CHECKPOINT_PATH = ROOT / "models" / "nf_unsw_binary_transformer.pt"
OUTPUT_PATH = ROOT / "reports" / "nf_unsw_test_results.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    # Load the saved checkpoint without changing it.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    feature_names = checkpoint["feature_names"]

    # Load only the held-out test split.
    df = pd.read_parquet(DATA_PATH)
    expected_columns = feature_names + ["Label"]

    if list(df.columns) != expected_columns:
        raise ValueError("Test feature columns/order do not match checkpoint.")

    X = df[feature_names].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    y_true = df["Label"].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    if not np.isfinite(X).all():
        raise ValueError("Test data contains non-finite feature values.")

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Unexpected labels found in test data.")

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y_true),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = NFUNSWBinaryTransformer(
        num_features=checkpoint["num_features"],
        num_classes=checkpoint["num_classes"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probabilities = []

    with torch.no_grad():
        for features, _ in loader:
            features = features.to(DEVICE)
            logits = model(features)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            all_probabilities.append(probabilities.cpu().numpy())

    attack_probs = np.concatenate(all_probabilities)
    predictions = (attack_probs >= 0.5).astype(np.int64)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            labels=[0, 1],
            zero_division=0,
        )
    )

    results = {
        "dataset": "NF-UNSW-NB15-v2",
        "evaluation_split": "test",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "threshold": 0.5,
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "benign_precision": float(precision[0]),
        "benign_recall": float(recall[0]),
        "benign_f1": float(f1[0]),
        "attack_precision": float(precision[1]),
        "attack_recall": float(recall[1]),
        "attack_f1": float(f1[1]),
        "macro_f1": float(np.mean(f1)),
        "pr_auc": float(average_precision_score(y_true, attack_probs)),
        "roc_auc": float(roc_auc_score(y_true, attack_probs)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "class_support": {
            "benign": int(support[0]),
            "attack": int(support[1]),
        },
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Attack"],
            output_dict=True,
            zero_division=0,
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nNF-UNSW-NB15-v2 TEST RESULTS")
    print("=" * 45)

    for key in [
        "checkpoint_epoch",
        "test_samples",
        "accuracy",
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "macro_f1",
        "pr_auc",
        "roc_auc",
    ]:
        print(f"{key}: {results[key]}")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(results["confusion_matrix"])
    print("\nFull report saved to:", OUTPUT_PATH)
    print("Checkpoint and test data were not modified.")


if __name__ == "__main__":
    main()