"""
predict.py - Inference script for Deepfake Audio Detection

Usage (CLI):
  python predict.py --audio path/to/file.wav [--checkpoint models/best_model.pt] [--arch cnn_bilstm] [--feature logmel]

Also exposes a reusable predict() function for the Streamlit app.
"""

# Import pandas and numpy FIRST to prevent DLL conflict on Windows
import pandas as pd
import numpy as np

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import preprocess_audio, extract_features
from models import get_model

# Label mapping
LABEL_MAP = {0: 'Genuine (Human)', 1: 'Deepfake (AI-Generated)'}
LABEL_COLORS = {0: 'green', 1: 'red'}

# ─── Default checkpoint config ─────────────────────────────────────────────────
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pt')
DEFAULT_ARCH = 'cnn_bilstm'
DEFAULT_FEATURE = 'logmel'


# ─── Core predict function ─────────────────────────────────────────────────────

def predict(audio_path: str,
            checkpoint_path: str = DEFAULT_CHECKPOINT,
            arch: str = DEFAULT_ARCH,
            feature_type: str = DEFAULT_FEATURE,
            device: str = None,
            model = None) -> dict:
    """
    Run inference on a single audio file.

    Args:
        audio_path:      Path to .wav / .mp3 / .flac file
        checkpoint_path: Path to the saved model checkpoint (.pt)
        arch:            Model architecture ('resnet18' or 'cnn_bilstm')
        feature_type:    Feature type ('logmel' or 'lfcc')
        device:          'cpu', 'cuda', or None (auto-detect)
        model:           Preloaded PyTorch model instance (optional)

    Returns:
        dict with keys:
          - 'label': str ('Genuine (Human)' or 'Deepfake (AI-Generated)')
          - 'label_id': int (0 or 1)
          - 'confidence': float (0.0 to 100.0)
          - 'genuine_prob': float (0.0 to 1.0)
          - 'deepfake_prob': float (0.0 to 1.0)
    """
    # Device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load model
    if model is None:
        model = _load_model(checkpoint_path, arch, feature_type, device)

    # Preprocess audio
    y = preprocess_audio(audio_path, training=False)
    features = extract_features(y, feature_type=feature_type)

    # Inference
    X = torch.tensor(features[np.newaxis], dtype=torch.float32).to(device)  # (1, 1, H, W)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    label_id   = int(np.argmax(probs))
    confidence = float(probs[label_id]) * 100.0

    return {
        'label':        LABEL_MAP[label_id],
        'label_id':     label_id,
        'confidence':   round(float(confidence), 2),
        'genuine_prob': float(probs[0]),
        'deepfake_prob': float(probs[1]),
    }


def _load_model(checkpoint_path: str, arch: str, feature_type: str, device):
    """Load model from checkpoint, inferring arch/feature from checkpoint if possible."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            "Please train the model first: python src/train.py --data_dir <path>"
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Use checkpoint metadata if available
    if 'arch' in ckpt:
        arch = ckpt['arch']
    if 'feature' in ckpt:
        feature_type = ckpt['feature']

    model = get_model(arch, feature_type=feature_type)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


# ─── Cached loader for Streamlit ───────────────────────────────────────────────

_cached_model = None
_cached_config = None

def load_model_cached(checkpoint_path: str = DEFAULT_CHECKPOINT,
                      arch: str = DEFAULT_ARCH,
                      feature_type: str = DEFAULT_FEATURE):
    """
    Load and cache the model in memory.
    Designed to be wrapped with @st.cache_resource in the Streamlit app.
    """
    global _cached_model, _cached_config
    config = (checkpoint_path, arch, feature_type)
    if _cached_model is None or _cached_config != config:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _cached_model  = _load_model(checkpoint_path, arch, feature_type, device)
        _cached_config = config
    return _cached_model


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deepfake Audio Detection — Single File Inference",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--audio",      required=True, help="Path to the audio file (.wav/.mp3/.flac)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--arch",       default=DEFAULT_ARCH,    choices=["resnet18", "cnn_bilstm"])
    parser.add_argument("--feature",    default=DEFAULT_FEATURE, choices=["logmel", "lfcc"])
    args = parser.parse_args()

    print(f"\nAnalyzing: {args.audio}")
    print("─" * 50)

    result = predict(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        arch=args.arch,
        feature_type=args.feature
    )

    print(f"  Result:     {result['label']}")
    print(f"  Confidence: {result['confidence']:.1f}%")
    print(f"  Genuine:    {result['genuine_prob'] * 100:.1f}%")
    print(f"  Deepfake:   {result['deepfake_prob'] * 100:.1f}%")
    print("─" * 50)
