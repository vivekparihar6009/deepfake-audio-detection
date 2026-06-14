"""debug_train.py - Step-by-step instrumentation of train.py main to find exact crash line."""
# Import pandas and numpy FIRST to prevent DLL conflict on Windows
import pandas as pd
import numpy as np

import sys
import os
import time
import random
import argparse
import traceback
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score

print("[DEBUG] Starting script...")
sys.stdout.flush()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from models import get_model
from dataset import get_dataloaders
from metrics import compute_eer

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    print("[DEBUG] Inside train_one_epoch")
    sys.stdout.flush()
    for batch_idx, (X, y) in enumerate(loader):
        if batch_idx % 10 == 0:
            print(f"[DEBUG] train batch {batch_idx}")
            sys.stdout.flush()
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
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
    print("[DEBUG] Inside validate")
    sys.stdout.flush()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            if batch_idx % 10 == 0:
                print(f"[DEBUG] val batch {batch_idx}")
                sys.stdout.flush()
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

def main():
    print("[DEBUG] main() entered")
    sys.stdout.flush()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--arch", default="cnn_bilstm")
    parser.add_argument("--feature", default="logmel")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--checkpoint_dir", default="models")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    print("[DEBUG] args parsed:", args)
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Device: {device}")
    sys.stdout.flush()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print("[DEBUG] Checkpoint dir created")
    sys.stdout.flush()

    try:
        print("[DEBUG] Getting dataloaders...")
        sys.stdout.flush()
        train_loader, val_loader, _ = get_dataloaders(
            args.data_dir,
            feature_type=args.feature,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples
        )
        print("[DEBUG] Dataloaders built")
        sys.stdout.flush()
    except BaseException as e:
        print("[DEBUG] EXCEPTION DETECTED IN GET_DATALOADERS:")
        print(type(e), e)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)

    print("[DEBUG] Getting model...")
    sys.stdout.flush()
    model = get_model(args.arch, feature_type=args.feature, dropout=args.dropout)
    model = model.to(device)
    print("[DEBUG] Model moved to device")
    sys.stdout.flush()

    print("[DEBUG] Counting class labels...")
    sys.stdout.flush()
    labels = train_loader.dataset.labels
    n_real = labels.count(0)
    n_fake = labels.count(1)
    total  = n_real + n_fake
    w_real = total / (2.0 * n_real + 1e-8)
    w_fake = total / (2.0 * n_fake + 1e-8)
    class_weights = torch.tensor([w_real, w_fake], dtype=torch.float32).to(device)
    print(f"[DEBUG] Weights calculated: Real={w_real:.3f}, Fake={w_fake:.3f}")
    sys.stdout.flush()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_eer = float('inf')
    best_epoch = 0
    patience_ctr = 0

    checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model_{args.arch}_{args.feature}.pt")
    print(f"[DEBUG] Checkpoint path: {checkpoint_path}")
    sys.stdout.flush()

    print("[DEBUG] Starting training loop...")
    sys.stdout.flush()
    for epoch in range(1, args.epochs + 1):
        print(f"[DEBUG] Epoch {epoch} starting")
        sys.stdout.flush()
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"[DEBUG] Epoch {epoch} train done. val starting...")
        sys.stdout.flush()
        val_loss, val_acc, val_f1, val_eer = validate(model, val_loader, criterion, device)
        print(f"[DEBUG] Epoch {epoch} val done.")
        sys.stdout.flush()
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(f"[DEBUG] Epoch {epoch} summary: loss={train_loss:.4f}, acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, eer={val_eer:.4f}")
        sys.stdout.flush()

        if val_eer < best_eer:
            best_eer = val_eer
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
            print(f"[DEBUG] Saved new best model at epoch {epoch}")
            sys.stdout.flush()
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"[DEBUG] Early stopping triggered at epoch {epoch}")
                sys.stdout.flush()
                break

    print("[DEBUG] main() complete!")
    sys.stdout.flush()

if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        print("[DEBUG] CRASH DETECTED!")
        print(f"[DEBUG] Exception type: {type(e)}")
        print(f"[DEBUG] Exception message: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
