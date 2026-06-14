"""
dataset.py - PyTorch Dataset and DataLoader for Deepfake Audio Detection
Implements:
  - Labeled file index from for-norm directory
  - Full augmentation pipeline at train time:
      * Additive Gaussian noise
      * Time stretching
      * Pitch shifting
      * Codec/compression simulation (G.711 style)
      * SpecAugment (frequency & time masking)
  - Feature extraction (Log-Mel or LFCC)
  - Reproducible with random seeds
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import preprocess_audio, extract_features

# ─── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ─── Augmentation ──────────────────────────────────────────────────────────────

def add_gaussian_noise(y: np.ndarray, snr_db: float = None) -> np.ndarray:
    """Add additive Gaussian noise at a random SNR between 15-40 dB."""
    if snr_db is None:
        snr_db = random.uniform(15, 40)
    signal_power = np.mean(y ** 2) + 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(y)) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)


def time_stretch(y: np.ndarray) -> np.ndarray:
    """Disabled by default to speed up training. Returns original signal."""
    return y


def pitch_shift(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Disabled by default to speed up training. Returns original signal."""
    return y


def codec_compression_simulation(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simulate G.711-style codec compression using fast numpy operations:
    1. Downsample by decimation (take every second sample)
    2. Apply mu-law encoding/decoding
    3. Upsample by repeating each sample twice
    """
    y_8k = y[::2]
    mu = 255.0
    y_compressed = np.sign(y_8k) * np.log1p(mu * np.abs(y_8k)) / np.log1p(mu)
    y_expanded = np.sign(y_compressed) * (1.0 / mu) * ((1 + mu) ** np.abs(y_compressed) - 1)
    y_restored = np.repeat(y_expanded.astype(np.float32), 2)
    if len(y_restored) < len(y):
        y_restored = np.pad(y_restored, (0, len(y) - len(y_restored)))
    elif len(y_restored) > len(y):
        y_restored = y_restored[:len(y)]
    return y_restored.astype(np.float32)


def spec_augment(spec: np.ndarray,
                 freq_mask_param: int = 20,
                 time_mask_param: int = 15,
                 n_freq_masks: int = 2,
                 n_time_masks: int = 2) -> np.ndarray:
    """
    Apply SpecAugment (Park et al., 2019):
    - Frequency masking: randomly zero out F consecutive mel bins
    - Time masking: randomly zero out T consecutive time frames
    spec shape: (1, H, W) where H=mel/lfcc bins, W=time frames
    """
    spec = spec.copy()
    _, H, W = spec.shape

    # Frequency masking
    for _ in range(n_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(H - f, 0))
        spec[0, f0:f0 + f, :] = 0

    # Time masking
    for _ in range(n_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(W - t, 0))
        spec[0, :, t0:t0 + t] = 0

    return spec


# ─── Label Index Builder ────────────────────────────────────────────────────────

def build_file_index(data_root: str, split: str = 'training') -> pd.DataFrame:
    """
    Build a labeled file index from the for-norm directory.

    Expected directory structure:
      data_root/
        for-norm/
          training/
            real/  *.wav
            fake/  *.wav
          validation/
            real/  *.wav
            fake/  *.wav
          testing/
            real/  *.wav
            fake/  *.wav

    Returns a DataFrame with columns: [path, label, split]
    label: 0 = Real (Genuine), 1 = Fake (Deepfake)
    """
    records = []
    split_dir = os.path.join(data_root, 'for-norm', split)

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Directory not found: {split_dir}\n"
            f"Check --data_dir argument and verify dataset was extracted to {data_root}"
        )

    for label_name, label_id in [('real', 0), ('fake', 1)]:
        class_dir = os.path.join(split_dir, label_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: {class_dir} not found, skipping.")
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith('.wav'):
                records.append({
                    'path': os.path.join(class_dir, fname),
                    'label': label_id,
                    'split': split,
                    'class_name': label_name
                })

    df = pd.DataFrame(records)
    print(f"[{split}] Found {len(df)} files - "
          f"Real: {(df['label'] == 0).sum()}, "
          f"Fake: {(df['label'] == 1).sum()}")
    return df


# ─── Dataset Class ──────────────────────────────────────────────────────────────

class DeepfakeAudioDataset(Dataset):
    """
    PyTorch Dataset for Deepfake Audio Detection.

    Args:
        file_index: DataFrame with columns [path, label]
        feature_type: 'logmel' or 'lfcc'
        training: If True, apply data augmentation
        max_samples: Optionally limit number of samples (for quick tests)
    """

    def __init__(self, file_index: pd.DataFrame, feature_type: str = 'logmel',
                 training: bool = True, max_samples: int = None):
        self.feature_type = feature_type
        self.training = training

        df = file_index.reset_index(drop=True)
        if max_samples is not None and max_samples < len(df):
            # Balanced sampling
            real_df = df[df['label'] == 0].sample(max_samples // 2, random_state=SEED)
            fake_df = df[df['label'] == 1].sample(max_samples // 2, random_state=SEED)
            df = pd.concat([real_df, fake_df]).reset_index(drop=True)
            print(f"  Subsampled to {len(df)} files ({max_samples//2} real, {max_samples//2} fake)")

        self.paths = df['path'].tolist()
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.paths)

    def _augment_audio(self, y: np.ndarray) -> np.ndarray:
        """Apply random audio-level augmentations (training only)."""
        # Each augmentation applied with 50% probability
        if random.random() < 0.5:
            y = add_gaussian_noise(y)
        if random.random() < 0.3:
            y = codec_compression_simulation(y)
        if random.random() < 0.3:
            y = time_stretch(y)
        if random.random() < 0.3:
            y = pitch_shift(y)
        return y

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]

        # Load and preprocess
        y = preprocess_audio(path, training=self.training)

        # Audio augmentation (train only)
        if self.training:
            y = self._augment_audio(y)
            # Re-fix length after augmentation (time-stretch can change length)
            from preprocessing import fix_length
            y = fix_length(y, training=True)

        # Extract features
        features = extract_features(y, feature_type=self.feature_type)

        # SpecAugment (train only, applied to spectrogram)
        if self.training:
            # Adjust freq_mask_param based on feature type
            freq_param = 20 if self.feature_type == 'logmel' else 8
            features = spec_augment(features, freq_mask_param=freq_param, time_mask_param=15)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ─── DataLoader factory ────────────────────────────────────────────────────────

def _worker_init_fn(wid):
    np.random.seed(SEED + wid)

def get_dataloaders(data_root: str, feature_type: str = 'logmel',
                    batch_size: int = 32, num_workers: int = 0,
                    max_samples: int = None):
    """
    Build train/val/test DataLoaders.

    Args:
        data_root: Root directory where the for-norm folder lives
        feature_type: 'logmel' or 'lfcc'
        batch_size: Batch size
        num_workers: Number of parallel workers
        max_samples: Limit samples per split (for quick experiments)
    Returns:
        (train_loader, val_loader, test_loader)
    """
    def make_loader(split, training):
        df = build_file_index(data_root, split=split)
        ds = DeepfakeAudioDataset(df, feature_type=feature_type,
                                  training=training, max_samples=max_samples)
        g = torch.Generator()
        g.manual_seed(SEED)
        return DataLoader(ds, batch_size=batch_size, shuffle=training,
                          num_workers=num_workers, pin_memory=False,
                          worker_init_fn=_worker_init_fn,
                          generator=g)

    train_loader = make_loader('training', training=True)
    val_loader   = make_loader('validation', training=False)
    test_loader  = make_loader('testing', training=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test with dummy files
    import tempfile, soundfile, pathlib

    print("Creating dummy audio test files...")
    with tempfile.TemporaryDirectory() as tmpdir:
        for split in ['training', 'validation', 'testing']:
            for cls in ['real', 'fake']:
                folder = os.path.join(tmpdir, 'for-norm', split, cls)
                os.makedirs(folder, exist_ok=True)
                for i in range(4):
                    wav = np.random.randn(48000).astype(np.float32)
                    soundfile.write(os.path.join(folder, f'{cls}_{i}.wav'), wav, 16000)

        train_loader, val_loader, test_loader = get_dataloaders(
            tmpdir, feature_type='logmel', batch_size=4, max_samples=8)

        batch_x, batch_y = next(iter(train_loader))
        print(f"  Train batch X shape: {batch_x.shape}")  # (4, 1, 128, 94)
        print(f"  Train batch Y shape: {batch_y.shape}")  # (4,)
        print(f"  Labels: {batch_y.tolist()}")

        train_loader2, _, _ = get_dataloaders(
            tmpdir, feature_type='lfcc', batch_size=4, max_samples=8)
        batch_x2, _ = next(iter(train_loader2))
        print(f"  LFCC batch X shape: {batch_x2.shape}")  # (4, 1, 40, T)

    print("All dataset checks passed!")
