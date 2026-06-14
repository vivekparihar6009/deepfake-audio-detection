"""
evaluate.py - Evaluation metrics and plot generation for Deepfake Audio Detection

Computes:
  - Overall Accuracy
  - Equal Error Rate (EER) with interpolation via sklearn ROC curve
  - F1 Score (binary)
  - Per-class Accuracy (Genuine and Deepfake separately)
  - Confusion Matrix (saved as PNG)
  - ROC Curve (saved as PNG)
  - DET Curve (saved as PNG)
  - Generates reports/performance_report.md with all metrics + embedded plots
"""

# Import pandas and numpy FIRST to prevent DLL conflict on Windows
import pandas as pd
import numpy as np

import os
import sys
import io
import random
import argparse
import torch

# Force UTF-8 encoding for stdout/stderr to avoid Windows charmap errors when printing emojis
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import matplotlib
matplotlib.use('Agg')   # must be set before any other matplotlib imports
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_curve, auc, classification_report
)

# Import pure metric helpers from metrics.py (no matplotlib dependency)
# These are re-exported here for backward compatibility
from metrics import compute_eer, per_class_accuracy  # noqa: F401

# ─── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ─── Plot generators ───────────────────────────────────────────────────────────


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Genuine', 'Deepfake'],
                yticklabels=['Genuine', 'Deepfake'],
                ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix -> {save_path}")


def plot_roc_curve(y_true, y_scores, save_path: str):
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve -> {save_path}")


def plot_det_curve(y_true, y_scores, save_path: str):
    """Detection Error Tradeoff (DET) curve: FRR vs FAR on normal deviate scale."""
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # Convert to normal deviate scale (clamp for numerical stability)
    eps = 1e-6
    fpr_nd = norm.ppf(np.clip(fpr, eps, 1 - eps))
    fnr_nd = norm.ppf(np.clip(fnr, eps, 1 - eps))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_nd, fnr_nd, color='steelblue', lw=2)

    # Mark EER point
    eer, _ = compute_eer(y_true, y_scores)
    eer_nd = norm.ppf(np.clip(eer, eps, 1 - eps))
    ax.plot(eer_nd, eer_nd, 'ro', markersize=8, label=f'EER = {eer * 100:.2f}%')

    # X/Y axis ticks in percentage
    ticks_pct = [1, 2, 5, 10, 20, 40]
    ticks_nd = [norm.ppf(t / 100) for t in ticks_pct]
    ax.set_xticks(ticks_nd)
    ax.set_xticklabels([f'{t}%' for t in ticks_pct])
    ax.set_yticks(ticks_nd)
    ax.set_yticklabels([f'{t}%' for t in ticks_pct])
    ax.set_xlabel('False Acceptance Rate (FAR)', fontsize=12)
    ax.set_ylabel('False Rejection Rate (FRR)', fontsize=12)
    ax.set_title('DET Curve', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved DET curve -> {save_path}")


# ─── Main evaluate function ────────────────────────────────────────────────────

def evaluate_model(model, data_loader, device, report_dir: str = 'reports',
                   split_name: str = 'test') -> dict:
    """
    Run full evaluation and generate all plots + performance_report.md.
    
    Args:
        model: Trained nn.Module
        data_loader: DataLoader for evaluation
        device: torch device
        report_dir: Directory to save plots and report
        split_name: e.g., 'test', 'validation', 'asvspoof'
    Returns:
        dict of all metric values
    """
    os.makedirs(report_dir, exist_ok=True)

    model.eval()
    all_preds, all_scores, all_labels = [], [], []

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_scores.extend(probs[:, 1].cpu().numpy().tolist())  # prob of fake
            all_labels.extend(y.cpu().numpy().tolist())

    y_true   = np.array(all_labels)
    y_scores = np.array(all_scores)

    # ── Core Metrics ──────────────────────────────────────────────────────────
    eer, eer_threshold = compute_eer(y_true, y_scores)

    # Calibrate predictions based on the EER threshold
    y_pred = (y_scores >= eer_threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    per_class = per_class_accuracy(y_true, y_pred)
    class_report = classification_report(y_true, y_pred,
                                         target_names=['Genuine', 'Deepfake'])

    # ── Threshold Checks ─────────────────────────────────────────────────────
    thresholds_met = {
        'Accuracy >= 80%': accuracy >= 0.80,
        'EER <= 12%':       eer <= 0.12,
        'F1 >= 80%':        f1 >= 0.80,
        'Genuine Acc >= 75%':  per_class['Genuine (Real)'] >= 0.75,
        'Deepfake Acc >= 75%': per_class['Deepfake (Fake)'] >= 0.75,
    }

    # ── Plots ─────────────────────────────────────────────────────────────────
    cm_path  = os.path.join(report_dir, f'confusion_matrix_{split_name}.png')
    roc_path = os.path.join(report_dir, f'roc_curve_{split_name}.png')
    det_path = os.path.join(report_dir, f'det_curve_{split_name}.png')

    plot_confusion_matrix(y_true, y_pred, cm_path)
    plot_roc_curve(y_true, y_scores, roc_path)
    plot_det_curve(y_true, y_scores, det_path)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS — {split_name.upper()}")
    print("=" * 60)
    print(f"  Accuracy:          {accuracy * 100:.2f}%")
    print(f"  EER:               {eer * 100:.2f}%  (threshold={eer_threshold:.4f})")
    print(f"  F1 Score:          {f1 * 100:.2f}%")
    for cls_name, acc in per_class.items():
        print(f"  {cls_name} Accuracy: {acc * 100:.2f}%")
    print("\nThreshold checks:")
    for check, passed in thresholds_met.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {check}")
    print("\nClassification Report:")
    print(class_report)

    # ── Write markdown report ─────────────────────────────────────────────────
    report_path = os.path.join(report_dir, 'performance_report.md')
    _write_markdown_report(
        report_path, split_name, accuracy, eer, eer_threshold, f1, per_class,
        thresholds_met, class_report, cm_path, roc_path, det_path
    )

    return {
        'accuracy': accuracy,
        'eer': eer,
        'f1': f1,
        'per_class': per_class,
        'thresholds_met': thresholds_met,
        'all_passed': all(thresholds_met.values())
    }


def _write_markdown_report(report_path, split_name, accuracy, eer, eer_threshold,
                           f1, per_class, thresholds_met, class_report,
                           cm_path, roc_path, det_path):
    """Write the complete performance_report.md."""
    lines = [
        "# Deepfake Audio Detection — Performance Report",
        "",
        f"**Evaluation Split:** {split_name}",
        "",
        "---",
        "",
        "## Metric Summary",
        "",
        "| Metric | Value | Threshold | Status |",
        "|--------|-------|-----------|--------|",
        f"| Overall Accuracy | **{accuracy * 100:.2f}%** | ≥ 80% | {'✅' if accuracy >= 0.80 else '❌'} |",
        f"| Equal Error Rate (EER) | **{eer * 100:.2f}%** | ≤ 12% | {'✅' if eer <= 0.12 else '❌'} |",
        f"| F1 Score | **{f1 * 100:.2f}%** | ≥ 80% | {'✅' if f1 >= 0.80 else '❌'} |",
        f"| Genuine (Real) Accuracy | **{per_class['Genuine (Real)'] * 100:.2f}%** | ≥ 75% | {'✅' if per_class['Genuine (Real)'] >= 0.75 else '❌'} |",
        f"| Deepfake Accuracy | **{per_class['Deepfake (Fake)'] * 100:.2f}%** | ≥ 75% | {'✅' if per_class['Deepfake (Fake)'] >= 0.75 else '❌'} |",
        "",
        f"> **EER Threshold (decision boundary):** {eer_threshold:.4f}",
        "",
        "---",
        "",
        "## Confusion Matrix",
        "",
        f"![Confusion Matrix]({os.path.basename(cm_path)})",
        "",
        "---",
        "",
        "## ROC Curve",
        "",
        f"![ROC Curve]({os.path.basename(roc_path)})",
        "",
        "---",
        "",
        "## DET Curve",
        "",
        f"![DET Curve]({os.path.basename(det_path)})",
        "",
        "---",
        "",
        "## Detailed Classification Report",
        "",
        "```",
        class_report,
        "```",
        "",
        "---",
        "",
        "## Threshold Verification",
        "",
    ]
    for check, passed in thresholds_met.items():
        icon = "✅" if passed else "❌"
        lines.append(f"- {icon} {check}")

    all_passed = all(thresholds_met.values())
    lines += [
        "",
        f"**All thresholds met: {'YES ✅' if all_passed else 'NO ❌ — Iteration required'}**",
        "",
        "---",
        "*Generated automatically by evaluate.py*"
    ]

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved performance report -> {report_path}")


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from models import get_model
    from dataset import get_dataloaders

    parser = argparse.ArgumentParser(description="Evaluate deepfake audio detection model.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--data_dir",   required=True, help="Root directory of the dataset")
    parser.add_argument("--arch",       default="cnn_bilstm", choices=["resnet18", "cnn_bilstm"])
    parser.add_argument("--feature",    default="logmel", choices=["logmel", "lfcc"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split",      default="testing", choices=["validation", "testing"])
    parser.add_argument("--report_dir", default="reports")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_loader, test_loader = get_dataloaders(
        args.data_dir, feature_type=args.feature, batch_size=args.batch_size)

    loader = test_loader if args.split == 'testing' else val_loader

    model = get_model(args.arch, feature_type=args.feature)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    results = evaluate_model(model, loader, device,
                             report_dir=args.report_dir,
                             split_name=args.split)
    print("\nDone! Report saved to", args.report_dir)
