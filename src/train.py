"""
train.py - Training script for Deepfake Audio Detection
Config-driven via argparse.

Features:
  - Random seeds for full reproducibility
  - AdamW optimizer + Cosine Annealing LR scheduler
  - Validation every epoch, best checkpoint saved by EER
  - Early stopping on validation EER
  - Class-weighted loss to handle imbalance
  - Logs per-epoch: loss, accuracy, EER
"""

# Import pandas and numpy FIRST to prevent DLL conflict on Windows
import pandas as pd
import numpy as np

import os
import sys
import time
import random
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score

# ─── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from models import get_model
from dataset import get_dataloaders
from metrics import compute_eer


# ─── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_scores, all_labels = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_scores.extend(probs[:, 1].cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    y_true   = np.array(all_labels)
    y_pred   = np.array(all_preds)
    y_scores = np.array(all_scores)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    eer, _ = compute_eer(y_true, y_scores)
    return avg_loss, acc, f1, eer


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train deepfake audio detection model.")
    parser.add_argument("--data_dir",    required=True,  help="Root directory of dataset (contains for-norm/)")
    parser.add_argument("--arch",        default="cnn_bilstm", choices=["resnet18", "cnn_bilstm"])
    parser.add_argument("--feature",     default="logmel", choices=["logmel", "lfcc"])
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--patience",    type=int, default=7, help="Early stopping patience (epochs)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples per split (None=use all)")
    parser.add_argument("--checkpoint_dir", default="models", help="Directory to save checkpoints")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data
    print(f"\nLoading data from: {args.data_dir}")
    print(f"Feature type: {args.feature}")
    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir,
        feature_type=args.feature,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples
    )

    # Model
    print(f"\nArchitecture: {args.arch}")
    model = get_model(args.arch, feature_type=args.feature, dropout=args.dropout)
    model = model.to(device)

    # Class weights for imbalance handling
    # Count real vs fake in train set
    labels = train_loader.dataset.labels
    n_real = labels.count(0)
    n_fake = labels.count(1)
    total  = n_real + n_fake
    w_real = total / (2.0 * n_real + 1e-8)
    w_fake = total / (2.0 * n_fake + 1e-8)
    class_weights = torch.tensor([w_real, w_fake], dtype=torch.float32).to(device)
    print(f"Class weights: Real={w_real:.3f}, Fake={w_fake:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training state
    best_eer     = float('inf')
    best_epoch   = 0
    patience_ctr = 0
    history      = []

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"best_model_{args.arch}_{args.feature}.pt"
    )

    print(f"\n{'='*70}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>8} | {'Val F1':>7} | {'Val EER':>8} | {'LR':>8}")
    print('-' * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_eer = validate(model, val_loader, criterion, device)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc*100:>8.2f}% | {val_loss:>8.4f} | {val_acc*100:>7.2f}% | {val_f1*100:>6.2f}% | {val_eer*100:>7.2f}% | {lr_now:>8.2e}  ({elapsed:.1f}s)")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1, 'val_eer': val_eer
        })

        # Save best model by validation EER
        if val_eer < best_eer:
            best_eer   = val_eer
            best_epoch = epoch
            patience_ctr = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_eer': val_eer,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'arch': args.arch,
                'feature': args.feature,
            }, checkpoint_path)
            print(f"  [BEST] New best model saved (EER={val_eer*100:.2f}%)")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n>> Early stopping triggered at epoch {epoch} (no EER improvement for {args.patience} epochs)")
                break

    print(f"\n{'='*70}")
    print(f"Training complete! Best Val EER: {best_eer*100:.2f}% at epoch {best_epoch}")
    print(f"Best checkpoint saved: {checkpoint_path}")

    # Also copy to canonical name for easy loading
    canonical = os.path.join(args.checkpoint_dir, "best_model.pt")
    import shutil
    shutil.copy(checkpoint_path, canonical)
    print(f"Copied to: {canonical}")

    return checkpoint_path


if __name__ == "__main__":
    main()
