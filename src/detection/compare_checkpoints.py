from pathlib import Path
import copy
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
VALIDATION_PATH = (
    ROOT / "data" / "processed" / "transformed"
    / "selected_v2" / "validation.parquet"
)

CHECKPOINTS = {
    "Production": ROOT / "models"
    / "transformer_detector_2000_best_with_features.pt",
    "Experiment_V1": ROOT / "models"
    / "transformer_detector_experiment_v1.pt",
}

DEVICE = torch.device("cpu")
BATCH_SIZE = 2048
THRESHOLDS = {"BOT": 0.80, "WEB_ATTACK": 0.80}


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Allow only the NumPy types needed by the experiment metadata.
    safe_types = [
        np._core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
    ]

    with torch.serialization.safe_globals(safe_types):
        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=True,
        )

    return checkpoint


def apply_threshold_policy(probabilities, class_names):
    predictions = probabilities.argmax(axis=1).copy()

    target_indices = [
        class_names.index(name) for name in THRESHOLDS
    ]

    for target, threshold in THRESHOLDS.items():
        target_idx = class_names.index(target)

        low_confidence = (
            (predictions == target_idx)
            & (probabilities[:, target_idx] < threshold)
        )

        rows = np.where(low_confidence)[0]

        if len(rows):
            alternative_scores = probabilities[rows].copy()
            alternative_scores[:, target_indices] = -1.0
            predictions[rows] = alternative_scores.argmax(axis=1)

    return predictions


def evaluate_checkpoint(name, path, X, y, expected_features):
    checkpoint = load_checkpoint(path)

    feature_names = checkpoint["feature_names"]
    class_names = checkpoint["class_names"]

    if feature_names != expected_features:
        raise ValueError(
            f"{name}: feature order does not match production."
        )

    if checkpoint["num_features"] != len(feature_names):
        raise ValueError(f"{name}: feature count mismatch.")

    if checkpoint["num_classes"] != len(class_names):
        raise ValueError(f"{name}: class count mismatch.")

    model = ThreatDetectionTransformer(
        num_features=checkpoint["num_features"],
        num_classes=checkpoint["num_classes"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probabilities = []

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.tensor(
                X[start:start + BATCH_SIZE],
                dtype=torch.float32,
                device=DEVICE,
            )

            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            all_probabilities.append(probs.cpu().numpy())

    probabilities = np.concatenate(all_probabilities, axis=0)
    predictions = apply_threshold_policy(probabilities, class_names)

    # Both models must use the same label order.
    if class_names != EXPECTED_CLASSES:
        raise ValueError(f"{name}: unexpected class order: {class_names}")

    accuracy = accuracy_score(y, predictions)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y,
        predictions,
        labels=np.arange(len(class_names)),
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    print(f"Accuracy:       {accuracy:.6f} ({accuracy:.2%})")
    print(f"Macro precision:{macro_p:.6f}")
    print(f"Macro recall:   {macro_r:.6f}")
    print(f"Macro F1:       {macro_f1:.6f}")
    print("\nPer-class report:")
    print(
        classification_report(
            y,
            predictions,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(class_names)
    print(confusion_matrix(
        y,
        predictions,
        labels=np.arange(len(class_names)),
    ))

    return {
        "name": name,
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }


if __name__ == "__main__":
    EXPECTED_CLASSES = [
        "BENIGN",
        "BOT",
        "BRUTE_FORCE",
        "DDoS",
        "DoS",
        "PORT_SCAN",
        "WEB_ATTACK",
    ]

    for name, path in CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}")

    print("Loading validation data...")
    validation = pd.read_parquet(VALIDATION_PATH)

    production_checkpoint = load_checkpoint(CHECKPOINTS["Production"])
    feature_names = production_checkpoint["feature_names"]

    expected_columns = feature_names + ["Attack_Type"]
    if set(validation.columns) != set(expected_columns):
        missing = set(expected_columns) - set(validation.columns)
        extra = set(validation.columns) - set(expected_columns)
        raise ValueError(
            f"Validation columns mismatch. Missing={missing}, Extra={extra}"
        )

    X = validation.loc[:, feature_names].to_numpy(dtype=np.float32)
    label_to_index = {
        label: index for index, label in enumerate(EXPECTED_CLASSES)
    }

    y_series = validation["Attack_Type"].map(label_to_index)
    if y_series.isna().any():
        raise ValueError(
            "Validation contains labels outside the expected seven classes."
        )

    y = y_series.to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        raise ValueError("Validation features contain NaN or infinity.")

    print(f"Validation records: {len(validation):,}")
    print(f"Features: {len(feature_names)}")
    print("Policy: BOT=0.80, WEB_ATTACK=0.80")
    print("Test set will not be loaded or evaluated.")

    results = []
    for name, path in CHECKPOINTS.items():
        results.append(
            evaluate_checkpoint(
                name, path, X, y, feature_names
            )
        )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(
        f"{'Model':<18} {'Accuracy':>12} "
        f"{'Macro F1':>12} {'Macro F1 (%)':>15}"
    )
    for result in results:
        print(
            f"{result['name']:<18} "
            f"{result['accuracy']:>12.6f} "
            f"{result['macro_f1']:>12.6f} "
            f"{result['macro_f1'] * 100:>14.2f}%"
        )
