from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/transformed/selected_v2/train.parquet"
)

VALIDATION_DATA_PATH = Path(
    "data/processed/transformed/selected_v2/validation.parquet"
)

TARGET_COLUMN = "Attack_Type"


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_MAPPING = {
    "BENIGN": 0,
    "BOT": 1,
    "BRUTE_FORCE": 2,
    "DDoS": 3,
    "DoS": 4,
    "PORT_SCAN": 5,
    "WEB_ATTACK": 6,
}

CLASS_NAMES = [
    "BENIGN",
    "BOT",
    "BRUTE_FORCE",
    "DDoS",
    "DoS",
    "PORT_SCAN",
    "WEB_ATTACK",
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1

# Controlled experiment only.
MAX_TRAIN_BATCHES = 2000


# ============================================================
# MODEL CHECKPOINT
# ============================================================

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = (
    MODEL_DIR / "transformer_detector_2000_best.pt"
)


# ============================================================
# DATASET
# ============================================================

class CICIDS2017Dataset(Dataset):
    """
    Dataset wrapper for the processed CICIDS2017 data.
    """

    def __init__(self, parquet_path):

        self.parquet_path = Path(parquet_path)

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.parquet_path}"
            )

        df = pd.read_parquet(self.parquet_path)

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' not found."
            )

        self.features = (
            df.drop(columns=[TARGET_COLUMN])
            .values
            .astype("float32")
        )

        self.labels = (
            df[TARGET_COLUMN]
            .map(LABEL_MAPPING)
            .values
            .astype("int64")
        )

        if pd.isna(self.labels).any():
            raise ValueError(
                "Unknown labels found in dataset."
            )

        self.features = torch.tensor(
            self.features,
            dtype=torch.float32,
        )

        self.labels = torch.tensor(
            self.labels,
            dtype=torch.long,
        )

        print(
            f"Loaded {self.parquet_path}: "
            f"{len(self.features):,} samples, "
            f"{self.features.shape[1]} features"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.features[index],
            self.labels[index],
        )


# ============================================================
# MODERATED CLASS SAMPLER
# ============================================================

def create_moderated_sampler(dataset):
    """
    Create a WeightedRandomSampler using:

        weight = 1 / sqrt(class_frequency)

    This provides minority-class oversampling while
    reducing the extreme correction produced by
    full inverse-frequency sampling.

    The original dataset is NOT modified.
    """

    class_counts = torch.bincount(
        dataset.labels,
        minlength=len(CLASS_NAMES),
    ).float()

    print("\nTraining class counts:")

    for index, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"  {class_name:<15}: "
            f"{int(class_counts[index]):,}"
        )

    # --------------------------------------------------------
    # Moderated inverse-frequency weighting
    # --------------------------------------------------------

    class_sampling_weights = (
        1.0 / torch.sqrt(class_counts)
    )

    print(
        "\nModerated class sampling weights:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"  {class_name:<15}: "
            f"{class_sampling_weights[index]:.8f}"
        )

    # Assign each sample the weight of its class.
    sample_weights = class_sampling_weights[
        dataset.labels
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )

    return sampler


# ============================================================
# TRANSFORMER DETECTOR
# ============================================================

class ThreatDetectionTransformer(nn.Module):
    """
    Transformer Encoder for CICIDS2017 network-flow
    classification.

    Each numerical feature is represented as one token.

    Feature-value embedding:
        [B, 58, 1] -> [B, 58, 64]

    Feature identity embedding:
        Identifies which network feature each token represents.

    Transformer Encoder:
        [B, 58, 64]

    Mean pooling:
        [B, 64]

    Classifier:
        [B, 7]
    """

    def __init__(
        self,
        num_features,
        num_classes,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()

        self.num_features = num_features

        # ----------------------------------------------------
        # Feature value embedding
        # ----------------------------------------------------

        self.feature_value_embedding = nn.Linear(
            1,
            d_model,
        )

        # ----------------------------------------------------
        # Feature identity embedding
        # ----------------------------------------------------

        self.feature_identity_embedding = nn.Embedding(
            num_features,
            d_model,
        )

        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            d_model,
            num_classes,
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------
        #
        # [B, 58]
        #

        batch_size = x.size(0)

        # ----------------------------------------------------
        # Convert each scalar feature into a token
        # ----------------------------------------------------

        x = x.unsqueeze(-1)

        # [B, 58, 1]

        x = self.feature_value_embedding(x)

        # [B, 58, 64]

        # ----------------------------------------------------
        # Add feature identity
        # ----------------------------------------------------

        feature_indices = torch.arange(
            self.num_features,
            device=x.device,
        )

        feature_identity = (
            self.feature_identity_embedding(
                feature_indices
            )
        )

        # [58, 64]

        feature_identity = (
            feature_identity
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        # [B, 58, 64]

        x = x + feature_identity

        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------

        x = self.transformer_encoder(x)

        # [B, 58, 64]

        # ----------------------------------------------------
        # Mean pooling
        # ----------------------------------------------------

        x = x.mean(dim=1)

        # [B, 64]

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(x)

        # [B, 7]

        return logits


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    max_batches=None,
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_index, (features, labels) in enumerate(
        dataloader
    ):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        if (batch_index + 1) % 100 == 0:
            print(
                f"  Batch {batch_index + 1}: "
                f"loss={loss.item():.6f}"
            )

    if total_samples == 0:
        return 0.0

    return total_loss / total_samples


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for features, labels in dataloader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(
                outputs,
                labels,
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = (
        total_loss / total_samples
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=list(range(len(CLASS_NAMES))),
            zero_division=0,
        )
    )

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_evaluation_results(results):

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Loss:             {results['loss']:.6f}"
    )

    print(
        f"Accuracy:         {results['accuracy']:.6f}"
    )

    print(
        f"Macro Precision:  "
        f"{results['macro_precision']:.6f}"
    )

    print(
        f"Macro Recall:     "
        f"{results['macro_recall']:.6f}"
    )

    print(
        f"Macro F1:         "
        f"{results['macro_f1']:.6f}"
    )

    print("\nPer-Class Results:")
    print("-" * 70)

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:<15} "
            f"Precision={results['precision'][index]:.6f} "
            f"Recall={results['recall'][index]:.6f} "
            f"F1={results['f1'][index]:.6f}"
        )

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRANSFORMER THREAT DETECTION")
    print("MODERATED CLASS-BALANCED SAMPLING EXPERIMENT")
    print("=" * 70)

    print(f"Device:            {DEVICE}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Learning rate:     {LEARNING_RATE}")
    print(f"Epochs:            {NUM_EPOCHS}")
    print(
        f"Max train batches: "
        f"{MAX_TRAIN_BATCHES}"
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_dataset = CICIDS2017Dataset(
        DATA_PATH
    )

    validation_dataset = CICIDS2017Dataset(
        VALIDATION_DATA_PATH
    )

    # --------------------------------------------------------
    # Create moderated sampler
    # --------------------------------------------------------

    train_sampler = create_moderated_sampler(
        train_dataset
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    num_features = train_dataset.features.shape[1]
    num_classes = len(CLASS_NAMES)

    model = ThreatDetectionTransformer(
        num_features=num_features,
        num_classes=num_classes,
    ).to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Normal CrossEntropyLoss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_macro_f1 = -1.0

    for epoch in range(NUM_EPOCHS):

        print("\n" + "=" * 70)
        print(
            f"EPOCH {epoch + 1}/{NUM_EPOCHS}"
        )
        print("=" * 70)

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            max_batches=MAX_TRAIN_BATCHES,
        )

        print(
            f"\nTraining loss: "
            f"{train_loss:.6f}"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        results = evaluate_model(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=DEVICE,
        )

        print_evaluation_results(
            results
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if (
            results["macro_f1"]
            > best_macro_f1
        ):

            best_macro_f1 = (
                results["macro_f1"]
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "num_features":
                        num_features,
                    "num_classes":
                        num_classes,
                    "class_names":
                        CLASS_NAMES,
                    "label_mapping":
                        LABEL_MAPPING,
                    "macro_f1":
                        best_macro_f1,
                },
                BEST_MODEL_PATH,
            )

            print(
                f"\nBest model saved to: "
                f"{BEST_MODEL_PATH}"
            )


if __name__ == "__main__":
    main()