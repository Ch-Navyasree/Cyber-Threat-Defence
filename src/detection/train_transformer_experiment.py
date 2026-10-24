from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.detection.baseline_detector import (
    CICIDS2017Dataset,
    ThreatDetectionTransformer,
    create_moderated_sampler,
    evaluate_model,
    print_evaluation_results,
    train_one_epoch,
    CLASS_NAMES,
    LABEL_MAPPING,
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data/processed/transformed/selected_v2/train.parquet"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data/processed/transformed/selected_v2/validation.parquet"
)

REFERENCE_CHECKPOINT = (
    PROJECT_ROOT
    / "models/transformer_detector_2000_best_with_features.pt"
)

EXPERIMENT_CHECKPOINT = (
    PROJECT_ROOT
    / "models/transformer_detector_experiment_v1.pt"
)

NUM_EPOCHS = 3
MAX_TRAIN_BATCHES = 2000
SEED = 42


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("TRANSFORMER EXPERIMENT V1")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Maximum batches per epoch: {MAX_TRAIN_BATCHES}")

    # Read reference metadata; do not modify the checkpoint.
    reference = torch.load(
        REFERENCE_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    feature_names = reference["feature_names"]

    if reference["class_names"] != CLASS_NAMES:
        raise ValueError("Class names differ from the reference checkpoint.")

    train_df = pd.read_parquet(TRAIN_PATH)
    validation_df = pd.read_parquet(VALIDATION_PATH)

    expected_columns = feature_names + ["Attack_Type"]

    if train_df.columns.tolist() != expected_columns:
        raise ValueError(
            "Training feature columns/order do not match "
            "the reference checkpoint."
        )

    if validation_df.columns.tolist() != expected_columns:
        raise ValueError(
            "Validation feature columns/order do not match "
            "the reference checkpoint."
        )

    # Dataset wrappers read the same verified files.
    train_dataset = CICIDS2017Dataset(TRAIN_PATH)
    validation_dataset = CICIDS2017Dataset(VALIDATION_PATH)

    sampler = create_moderated_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ThreatDetectionTransformer(
        num_features=len(feature_names),
        num_classes=len(CLASS_NAMES),
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_macro_f1 = -1.0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            max_batches=MAX_TRAIN_BATCHES,
        )

        print(f"Training loss: {train_loss:.6f}")

        results = evaluate_model(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=DEVICE,
        )

        print_evaluation_results(results)

        if results["macro_f1"] > best_macro_f1:
            best_macro_f1 = results["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced.")

    torch.save(
        {
            "model_state_dict": best_state,
            "num_features": len(feature_names),
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "label_mapping": LABEL_MAPPING,
            "feature_names": feature_names,
            "macro_f1": best_macro_f1,
            "experiment": "v1",
            "seed": SEED,
            "epochs": NUM_EPOCHS,
            "max_train_batches": MAX_TRAIN_BATCHES,
        },
        EXPERIMENT_CHECKPOINT,
    )

    print("\nExperiment completed.")
    print(f"Best validation macro F1: {best_macro_f1:.6f}")
    print(f"Experimental checkpoint: {EXPERIMENT_CHECKPOINT}")


if __name__ == "__main__":
    main()