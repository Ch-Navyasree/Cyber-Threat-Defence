from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.detection.baseline_detector import ThreatDetectionTransformer

ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "data" / "processed" / "transformed" / "selected_v2" / "test.parquet"

CHECKPOINTS = {
    "Production": ROOT / "models" / "transformer_detector_2000_best_with_features.pt",
    "Experiment_V1": ROOT / "models" / "transformer_detector_experiment_v1.pt",
}

CLASSES = [
    "BENIGN", "BOT", "BRUTE_FORCE", "DDoS", "DoS", "PORT_SCAN", "WEB_ATTACK"
]
THRESHOLDS = {"BOT": 0.80, "WEB_ATTACK": 0.80}
BATCH_SIZE = 2048
DEVICE = torch.device("cpu")


def load_checkpoint(path):
    safe_types = [
        np._core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(safe_types):
        return torch.load(path, map_location=DEVICE, weights_only=True)


def apply_threshold_policy(probabilities, class_names):
    predictions = probabilities.argmax(axis=1).copy()
    target_indices = [class_names.index(name) for name in THRESHOLDS]

    for target, threshold in THRESHOLDS.items():
        target_idx = class_names.index(target)
        rows = np.where(
            (predictions == target_idx)
            & (probabilities[:, target_idx] < threshold)
        )[0]

        if len(rows):
            alternatives = probabilities[rows].copy()
            alternatives[:, target_indices] = -1.0
            predictions[rows] = alternatives.argmax(axis=1)

    return predictions


def evaluate(name, checkpoint, X, y):
    model = ThreatDetectionTransformer(
        num_features=checkpoint["num_features"],
        num_classes=checkpoint["num_classes"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probabilities = []
    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.tensor(
                X[start:start + BATCH_SIZE],
                dtype=torch.float32,
                device=DEVICE,
            )
            probabilities.append(
                F.softmax(model(batch), dim=1).cpu().numpy()
            )

    probabilities = np.concatenate(probabilities, axis=0)
    predictions = apply_threshold_policy(probabilities, checkpoint["class_names"])

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, labels=np.arange(len(CLASSES)),
        average="macro", zero_division=0,
    )

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print(f"Accuracy:        {accuracy_score(y, predictions):.6f}")
    print(f"Macro precision: {precision:.6f}")
    print(f"Macro recall:    {recall:.6f}")
    print(f"Macro F1:        {f1:.6f}")
    print("\nPer-class metrics:")
    print(classification_report(
        y, predictions, labels=np.arange(len(CLASSES)),
        target_names=CLASSES, digits=4, zero_division=0,
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(CLASSES)
    print(confusion_matrix(y, predictions, labels=np.arange(len(CLASSES))))

    return {
        "model": name,
        "accuracy": accuracy_score(y, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


if __name__ == "__main__":
    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    for name, path in CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} checkpoint missing: {path}")

    print("Reading held-out test dataset...")
    test = pd.read_parquet(TEST_PATH)

    production = load_checkpoint(CHECKPOINTS["Production"])
    feature_names = production["feature_names"]

    if production["class_names"] != CLASSES:
        raise ValueError("Unexpected production class order.")

    for name, path in CHECKPOINTS.items():
        checkpoint = load_checkpoint(path)
        if checkpoint["feature_names"] != feature_names:
            raise ValueError(f"{name}: feature order mismatch.")
        if checkpoint["class_names"] != CLASSES:
            raise ValueError(f"{name}: class order mismatch.")

    expected_columns = feature_names + ["Attack_Type"]
    if set(test.columns) != set(expected_columns):
        raise ValueError(
            f"Test columns mismatch. Missing: "
            f"{set(expected_columns) - set(test.columns)}; "
            f"extra: {set(test.columns) - set(expected_columns)}"
        )

    X = test.loc[:, feature_names].to_numpy(dtype=np.float32)
    label_map = {label: i for i, label in enumerate(CLASSES)}
    mapped = test["Attack_Type"].map(label_map)

    if mapped.isna().any():
        raise ValueError("Unknown target label found in test data.")
    if not np.isfinite(X).all():
        raise ValueError("Test features contain NaN or infinity.")

    y = mapped.to_numpy(dtype=np.int64)

    print(f"Test records: {len(test):,}")
    print(f"Features: {len(feature_names)}")
    print("Thresholds: BOT=0.80, WEB_ATTACK=0.80")
    print("No training or checkpoint writes will be performed.")

    results = [
        evaluate(name, load_checkpoint(path), X, y)
        for name, path in CHECKPOINTS.items()
    ]

    print("\n" + "=" * 72)
    print("HELD-OUT TEST SUMMARY")
    print("=" * 72)
    print(f"{'Model':<18}{'Accuracy':>12}{'Macro F1':>12}")
    for result in results:
        print(
            f"{result['model']:<18}"
            f"{result['accuracy']:>12.6f}"
            f"{result['macro_f1']:>12.6f}"
        )
