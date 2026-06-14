"""
models.py - Model architectures for Deepfake Audio Detection

Two architectures:
1. ResNet18 - Baseline CNN (adapted for 1-channel spectrograms)
2. CNN-BiLSTM - Stronger model with temporal context (recommended)

Both support dynamic input sizing based on feature type (logmel vs lfcc).
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ─── ResNet18 Baseline ─────────────────────────────────────────────────────────

class ResNet18Classifier(nn.Module):
    """
    ResNet18 adapted for 1-channel spectrogram input.
    Architecture:
      - Conv1 replaced to accept (1, H, W) input
      - Global Average Pooling -> FC(2) for binary classification
    
    Args:
        input_height: Number of feature bins (128 for LogMel, 40 for LFCC)
        dropout: Dropout rate before final FC layer
        pretrained: Use ImageNet pretrained weights (adapts first conv)
    """

    def __init__(self, input_height: int = 128, dropout: float = 0.3, pretrained: bool = False):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Replace first conv: 3-channel -> 1-channel
        backbone.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove final FC; replace with dropout + new FC
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2)
        )
        self.backbone = backbone

    def forward(self, x):
        """x: (B, 1, H, W)"""
        return self.backbone(x)


# ─── CNN-BiLSTM Stronger Model ─────────────────────────────────────────────────

class CNNBlock(nn.Module):
    """2D CNN block: Conv -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, pool_size: tuple = (2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size)
        )

    def forward(self, x):
        return self.block(x)


class CNNBiLSTMClassifier(nn.Module):
    """
    CNN-BiLSTM model for deepfake audio detection.
    Architecture:
      4x CNN blocks (feature extraction from spectrogram)
      -> Reshape to sequence (time dimension as steps)
      -> BiLSTM (captures temporal context)
      -> Attention pooling (weighted average over time steps)
      -> FC(2) binary classifier

    Dynamically handles different input heights (128 for LogMel, 40 for LFCC).
    
    Args:
        input_height: Number of feature bins (H dimension)
        input_width:  Number of time frames (W dimension)
        lstm_hidden:  BiLSTM hidden units (x2 for bidirectional)
        lstm_layers:  Number of LSTM layers
        dropout:      Dropout probability
    """

    def __init__(self, input_height: int = 128, input_width: int = 94,
                 lstm_hidden: int = 128, lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        # 4 CNN blocks — each halves spatial dims via MaxPool(2,2)
        # After 4 blocks: H // 16, W // 16
        self.cnn = nn.Sequential(
            CNNBlock(1,   32, pool_size=(2, 2)),
            CNNBlock(32,  64, pool_size=(2, 2)),
            CNNBlock(64,  128, pool_size=(2, 2)),
            CNNBlock(128, 256, pool_size=(2, 2)),
        )

        # Calculate CNN output height (freq dim) after 4x pool of 2
        cnn_h = input_height // (2 ** 4)
        cnn_h = max(cnn_h, 1)  # at least 1

        # Each time step's feature vector = 256 channels * cnn_h
        cnn_feat_dim = 256 * cnn_h

        # BiLSTM (bidirectional doubles the hidden size)
        self.bilstm = nn.LSTM(
            input_size=cnn_feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Attention pooling over time steps
        self.attention = nn.Linear(lstm_hidden * 2, 1)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        """
        x: (B, 1, H, W)
        Returns: logits (B, 2)
        """
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (B, 256, H', W')
        B, C, H, W = cnn_out.shape

        # Reshape: treat W (time) as sequence steps, flatten C*H as features
        # (B, C, H, W) -> (B, W, C*H)
        cnn_out = cnn_out.permute(0, 3, 1, 2)  # (B, W, C, H)
        cnn_out = cnn_out.contiguous().view(B, W, C * H)  # (B, W, C*H)

        # BiLSTM
        lstm_out, _ = self.bilstm(cnn_out)  # (B, W, 2*lstm_hidden)

        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, W, 1)
        context = (attn_weights * lstm_out).sum(dim=1)  # (B, 2*lstm_hidden)

        # Classify
        logits = self.classifier(context)  # (B, 2)
        return logits


# ─── Model factory ─────────────────────────────────────────────────────────────

def get_model(arch: str, feature_type: str = 'logmel', **kwargs) -> nn.Module:
    """
    Return the requested model, configured for the given feature type.

    Args:
        arch: 'resnet18' or 'cnn_bilstm'
        feature_type: 'logmel' (H=128) or 'lfcc' (H=40)
    Returns:
        nn.Module
    """
    # Determine input height based on feature type
    if feature_type == 'logmel':
        input_height = 128
        input_width = 94
    elif feature_type == 'lfcc':
        input_height = 40
        input_width = 94
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    if arch == 'resnet18':
        model = ResNet18Classifier(input_height=input_height, **kwargs)
    elif arch == 'cnn_bilstm':
        model = CNNBiLSTMClassifier(input_height=input_height, input_width=input_width, **kwargs)
    else:
        raise ValueError(f"Unknown arch: {arch}. Choose 'resnet18' or 'cnn_bilstm'.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model '{arch}' with feature='{feature_type}': {n_params:,} trainable parameters")
    return model


if __name__ == "__main__":
    # Sanity forward pass
    print("Testing models...")

    for feat, H, W in [('logmel', 128, 94), ('lfcc', 40, 94)]:
        dummy = torch.randn(4, 1, H, W)

        m1 = get_model('resnet18', feature_type=feat)
        out1 = m1(dummy)
        print(f"  ResNet18   [{feat}] input {dummy.shape} -> output {out1.shape}")
        assert out1.shape == (4, 2), f"Expected (4,2), got {out1.shape}"

        m2 = get_model('cnn_bilstm', feature_type=feat)
        out2 = m2(dummy)
        print(f"  CNN-BiLSTM [{feat}] input {dummy.shape} -> output {out2.shape}")
        assert out2.shape == (4, 2), f"Expected (4,2), got {out2.shape}"

    print("All model checks passed!")
