"""
metrics.py - Pure computation metrics with no matplotlib/seaborn dependencies.
Safe to import in training loops before DataLoader/CUDA initialization.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """
    Compute Equal Error Rate (EER).
    Implements the exact interpolation method from the project brief.

    Args:
        y_true:   Ground truth binary labels (0=real, 1=fake)
        y_scores: Predicted probability for class 1 (fake)

    Returns:
        (eer: float, eer_threshold: float)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr  # False Negative Rate = False Rejection Rate

    # Find the index where |FPR - FNR| is minimized
    idx = np.nanargmin(np.absolute(fpr - fnr))

    # Interpolate between two surrounding points for precision
    if idx > 0:
        fpr_low, fpr_high = fpr[idx - 1], fpr[idx]
        fnr_low, fnr_high = fnr[idx - 1], fnr[idx]
        if (fpr_high - fpr_low + fnr_low - fnr_high) != 0:
            t = (fnr_low - fpr_low) / (fpr_high - fpr_low + fnr_low - fnr_high + 1e-8)
            eer = fpr_low + t * (fpr_high - fpr_low)
        else:
            eer = (fpr[idx] + fnr[idx]) / 2.0
    else:
        eer = (fpr[idx] + fnr[idx]) / 2.0

    eer_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
    return float(eer), float(eer_threshold)


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute per-class accuracy for both Real (0) and Fake (1)."""
    results = {}
    for cls, name in [(0, 'Genuine (Real)'), (1, 'Deepfake (Fake)')]:
        mask = (y_true == cls)
        correct = np.sum(y_pred[mask] == cls)
        total = np.sum(mask)
        results[name] = correct / total if total > 0 else 0.0
    return results
