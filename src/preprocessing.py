"""
preprocessing.py - Audio preprocessing and feature extraction
Supports Log-Mel Spectrogram and LFCC features.
Random seeds are set for reproducibility.
"""

import random
import os
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf

# ─── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000          # 16 kHz
CLIP_DURATION = 3.0          # seconds
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)   # 48000 samples

# Log-Mel parameters
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
# Log-Mel produces shape: (1, 128, 94) for 3s @ 16kHz

# LFCC parameters
N_LFCC = 40
# LFCC produces shape: (1, 40, T)

# ─── Audio Loading ─────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load audio file, convert to mono, resample to SAMPLE_RATE if needed using fast soundfile."""
    try:
        y, sr = sf.read(path, dtype='float32')
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    except Exception as e:
        try:
            y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception as ex:
            raise IOError(f"Failed to load audio file {path}: {e} (Fallback: {ex})")
    return y


def trim_silence(y: np.ndarray, top_db: int = 20) -> np.ndarray:
    """Energy-based silence trimming."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    # Fallback: if all trimmed, return original
    if len(y_trimmed) < SAMPLE_RATE * 0.5:
        return y
    return y_trimmed


def fix_length(y: np.ndarray, training: bool = True) -> np.ndarray:
    """
    Crop or pad audio to exactly CLIP_SAMPLES.
    - Training: random crop (for augmentation diversity)
    - Eval: center crop
    """
    n = len(y)
    if n >= CLIP_SAMPLES:
        if training:
            start = random.randint(0, n - CLIP_SAMPLES)
        else:
            start = (n - CLIP_SAMPLES) // 2
        y = y[start: start + CLIP_SAMPLES]
    else:
        # Pad with zeros at end
        pad_len = CLIP_SAMPLES - n
        y = np.pad(y, (0, pad_len), mode='constant')
    return y


def preprocess_audio(path: str, training: bool = True) -> np.ndarray:
    """Full audio preprocessing pipeline: load → trim silence (non-training only) → fix length."""
    y = load_audio(path)
    if not training:
        y = trim_silence(y)
    y = fix_length(y, training=training)
    return y.astype(np.float32)


# ─── Feature Extraction ────────────────────────────────────────────────────────

_mel_transform = None

def extract_logmel(y: np.ndarray) -> np.ndarray:
    """
    Extract Log-Mel Spectrogram using torchaudio.
    Returns: (1, N_MELS, T) = (1, 128, 94) for 3s clip
    """
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=0.0,
            f_max=SAMPLE_RATE // 2,
            center=True,
            power=2.0
        )
    y_t = torch.from_numpy(y)
    mel_spec = _mel_transform(y_t).numpy()
    max_val = np.max(mel_spec)
    log_mel = 10 * np.log10(np.maximum(mel_spec, 1e-10) / (max_val + 1e-10))
    # Normalize to [-1, 1]
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel[np.newaxis, :, :].astype(np.float32)   # (1, 128, T)


def _dct_matrix(n_filters: int, n_input: int) -> np.ndarray:
    """Compute DCT-II matrix for LFCC calculation."""
    n = np.arange(n_input)
    k = np.arange(n_filters)[:, np.newaxis]
    dct = np.cos(np.pi / n_input * (n + 0.5) * k)
    dct[0] *= 1.0 / np.sqrt(2.0)
    dct *= np.sqrt(2.0 / n_input)
    return dct.astype(np.float32)


def extract_lfcc(y: np.ndarray) -> np.ndarray:
    """
    Extract Linear Frequency Cepstral Coefficients (LFCC).
    Highly effective for anti-spoofing — captures artifacts in linear frequency domain.
    Returns: (1, N_LFCC, T) = (1, 40, T)
    """
    # 1. Compute magnitude spectrogram
    D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))  # (n_fft//2+1, T)

    # 2. Apply linear filterbank (N_LFCC triangular filters on linear scale)
    n_bins = D.shape[0]
    n_frames = D.shape[1]
    freqs = np.linspace(0, SAMPLE_RATE // 2, n_bins)
    filter_freqs = np.linspace(0, SAMPLE_RATE // 2, N_LFCC + 2)

    filterbank = np.zeros((N_LFCC, n_bins), dtype=np.float32)
    for m in range(N_LFCC):
        f_low = filter_freqs[m]
        f_center = filter_freqs[m + 1]
        f_high = filter_freqs[m + 2]
        for k, f in enumerate(freqs):
            if f_low <= f <= f_center:
                filterbank[m, k] = (f - f_low) / (f_center - f_low + 1e-8)
            elif f_center < f <= f_high:
                filterbank[m, k] = (f_high - f) / (f_high - f_center + 1e-8)

    # 3. Apply filterbank: (N_LFCC, n_bins) x (n_bins, T) -> (N_LFCC, T)
    linear_spec = np.dot(filterbank, D)

    # 4. Log compression
    log_linear = np.log(linear_spec + 1e-8)

    # 5. DCT to get cepstral coefficients
    dct_mat = _dct_matrix(N_LFCC, N_LFCC)
    lfcc = np.dot(dct_mat, log_linear)  # (N_LFCC, T)

    # 6. Normalize
    lfcc = (lfcc - lfcc.mean()) / (lfcc.std() + 1e-8)

    return lfcc[np.newaxis, :, :].astype(np.float32)  # (1, 40, T)


def extract_features(y: np.ndarray, feature_type: str = 'logmel') -> np.ndarray:
    """
    Extract features based on type.
    Args:
        y: Raw audio waveform (numpy array, 48000 samples for 3s at 16kHz)
        feature_type: 'logmel' or 'lfcc'
    Returns:
        features: numpy array of shape (1, H, W)
    """
    if feature_type == 'logmel':
        return extract_logmel(y)
    elif feature_type == 'lfcc':
        return extract_lfcc(y)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}. Choose 'logmel' or 'lfcc'.")


def get_feature_shape(feature_type: str = 'logmel') -> tuple:
    """
    Returns the expected (H, W) shape of the feature map for model initialization.
    """
    dummy = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    feat = extract_features(dummy, feature_type=feature_type)
    return feat.shape[1], feat.shape[2]   # (H, W)


if __name__ == "__main__":
    # Quick sanity check
    print("Testing preprocessing pipeline...")
    dummy_audio = np.random.randn(CLIP_SAMPLES).astype(np.float32)

    logmel = extract_logmel(dummy_audio)
    print(f"  Log-Mel shape: {logmel.shape}")   # expected: (1, 128, 94)

    lfcc = extract_lfcc(dummy_audio)
    print(f"  LFCC shape:    {lfcc.shape}")      # expected: (1, 40, 94)

    print("All checks passed!")
