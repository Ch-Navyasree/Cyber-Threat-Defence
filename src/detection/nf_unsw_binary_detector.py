
import torch
import torch.nn as nn


class NFUNSWBinaryTransformer(nn.Module):
    """
    Binary Transformer detector for NF-UNSW-NB15-v2.

    Input:
        [batch_size, 41]

    Output:
        [batch_size, 2]

    Output classes:
        0 = Benign
        1 = Attack
    """

    def __init__(
        self,
        num_features=41,
        num_classes=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()

        self.num_features = num_features

        self.feature_value_embedding = nn.Linear(
            1, d_model
        )

        self.feature_identity_embedding = nn.Embedding(
            num_features, d_model
        )

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

        self.classifier = nn.Linear(
            d_model, num_classes
        )

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(
                "Expected input shape [batch_size, num_features]"
            )

        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, "
                f"received {x.shape[1]}"
            )

        batch_size = x.shape[0]

        # Represent each numerical feature as a token.
        x = x.unsqueeze(-1)
        x = self.feature_value_embedding(x)

        # Add the identity of each feature.
        indices = torch.arange(
            self.num_features,
            device=x.device,
        )

        identity = self.feature_identity_embedding(indices)
        identity = identity.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        x = x + identity

        # Learn relationships between feature tokens.
        x = self.transformer_encoder(x)

        # Aggregate token representations.
        x = x.mean(dim=1)

        # Produce benign and attack logits.
        return self.classifier(x)


if __name__ == "__main__":
    model = NFUNSWBinaryTransformer(num_features=41)
    sample = torch.randn(4, 41)
    output = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Model parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    assert output.shape == (4, 2)
    print("Model shape test: PASS")
