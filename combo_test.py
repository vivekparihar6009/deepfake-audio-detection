"""combo_test.py - Test evaluate import followed by get_dataloaders (exact crash sequence)"""
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("STEP 1: importing evaluate (triggers matplotlib.use Agg)", flush=True)
from evaluate import compute_eer
print("  -> evaluate imported OK", flush=True)

print("STEP 2: importing models", flush=True)
from models import get_model
print("  -> models imported OK", flush=True)

print("STEP 3: importing torch, numpy", flush=True)
import torch
import numpy as np
print(f"  -> torch OK, CUDA: {torch.cuda.is_available()}", flush=True)

print("STEP 4: calling get_dataloaders...", flush=True)
from dataset import get_dataloaders
try:
    train_loader, val_loader, test_loader = get_dataloaders(
        "D:\\kaggle-data\\for-norm",
        feature_type='logmel',
        batch_size=32,
        num_workers=0,
        max_samples=None
    )
    print(f"  -> train: {len(train_loader.dataset)} samples", flush=True)
    print(f"  -> val:   {len(val_loader.dataset)} samples", flush=True)
    print(f"  -> test:  {len(test_loader.dataset)} samples", flush=True)
except Exception as e:
    print(f"  -> FAILED in get_dataloaders: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("STEP 5: getting model...", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model("cnn_bilstm", feature_type="logmel", dropout=0.3)
model = model.to(device)
print(f"  -> model on {device}", flush=True)

print("STEP 6: fetching one train batch...", flush=True)
try:
    X, y = next(iter(train_loader))
    X, y = X.to(device), y.to(device)
    print(f"  -> X={X.shape}, y={y.shape}", flush=True)
except Exception as e:
    print(f"  -> FAILED on first batch: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("STEP 7: forward pass...", flush=True)
try:
    with torch.no_grad():
        out = model(X)
    print(f"  -> output shape: {out.shape}", flush=True)
except Exception as e:
    print(f"  -> FAILED forward pass: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("ALL STEPS PASSED!", flush=True)
