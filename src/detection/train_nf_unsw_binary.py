
from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)

from nf_unsw_binary_detector import NFUNSWBinaryTransformer


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "nf_unsw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "Label"
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 1e-3
SEED = 42

CHECKPOINT_PATH = MODEL_DIR / "nf_unsw_binary_transformer.pt"
REPORT_PATH = REPORT_DIR / "nf_unsw_validation_results.json"
PREPROCESSOR_PATH = MODEL_DIR / "nf_unsw_binary_preprocessor.joblib"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(name, expected_features):
    path = DATA_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)

    expected_columns = expected_features + [TARGET]
    if list(df.columns) != expected_columns:
        raise ValueError(f"{name}: feature columns/order mismatch")

    X = df[expected_features].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.int64)

    if not np.isfinite(X).all():
        raise ValueError(f"{name}: non-finite input values found")

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{name}: unexpected target values")

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    return dataset, y


def evaluate(model, loader, criterion):
    model.eval()
    losses = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)[:, 1]

            losses.append(loss.item() * len(y))
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    attack_probs = np.concatenate(all_probs)
    predictions = (attack_probs >= 0.5).astype(np.int64)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        "loss": float(sum(losses) / len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "benign_precision": float(precision[0]),
        "benign_recall": float(recall[0]),
        "attack_precision": float(precision[1]),
        "attack_recall": float(recall[1]),
        "attack_f1": float(f1[1]),
        "macro_f1": float(np.mean(f1)),
        "pr_auc": float(average_precision_score(y_true, attack_probs)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(PREPROCESSOR_PATH)

    preprocessing = joblib.load(PREPROCESSOR_PATH)
    feature_names = preprocessing["feature_columns"]

    print("Device:", DEVICE)
    print("Features:", len(feature_names))
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    train_dataset, y_train = load_split("train", feature_names)
    val_dataset, y_val = load_split("validation", feature_names)

    print(f"Training rows: {len(train_dataset):,}")
    print(f"Validation rows: {len(val_dataset):,}")
    print("Training class counts:", np.bincount(y_train, minlength=2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = NFUNSWBinaryTransformer(
        num_features=len(feature_names),
        num_classes=2,
    ).to(DEVICE)

    # Class-weighted loss gives the minority attack class more influence.
    counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2.0 * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_score = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for batch_index, (X, y) in enumerate(train_loader, start=1):
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_seen += len(y)

            if batch_index % 500 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"batch {batch_index}/{len(train_loader)} | "
                    f"loss {loss.item():.5f}"
                )

        train_loss = total_loss / total_seen
        metrics = evaluate(model, val_loader, criterion)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(train_loss)
        history.append(metrics)

        print(f"\nEpoch {epoch}: train_loss={train_loss:.5f}")
        print(json.dumps(metrics, indent=2))

        # Select by attack-class F1; PR-AUC is also reported.
        score = metrics["attack_f1"]
        if score > best_score:
            best_score = score

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_features": len(feature_names),
                    "num_classes": 2,
                    "feature_names": feature_names,
                    "class_names": ["Benign", "Attack"],
                    "label_mapping": {"Benign": 0, "Attack": 1},
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "seed": SEED,
                },
                CHECKPOINT_PATH,
            )
            print("Saved best checkpoint:", CHECKPOINT_PATH)

    report = {
        "dataset": "NF-UNSW-NB15-v2",
        "selection_metric": "validation attack F1 at threshold 0.5",
        "device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "class_weights": class_weights.tolist(),
        "history": history,
        "best_checkpoint": str(CHECKPOINT_PATH),
        "best_validation_attack_f1": best_score,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nTraining completed.")
    print("Validation report:", REPORT_PATH)
    print("Test split was not loaded or evaluated.")


if __name__ == "__main__":
    main()
