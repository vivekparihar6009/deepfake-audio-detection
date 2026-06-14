"""granular_test.py - Inline get_dataloaders to find exact crash line after matplotlib"""
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Reproduce exact environment from debug_train.py
print("Loading evaluate (matplotlib.use Agg)...", flush=True)
from evaluate import compute_eer
print("  -> OK", flush=True)

print("Loading models...", flush=True)
from models import get_model
print("  -> OK", flush=True)

import torch
import numpy as np
import pandas as pd
print(f"Torch CUDA: {torch.cuda.is_available()}", flush=True)

# Now inline get_dataloaders step by step
print("\n--- Inlining get_dataloaders ---", flush=True)

print("A) importing build_file_index + DeepfakeAudioDataset...", flush=True)
from dataset import build_file_index, DeepfakeAudioDataset
from torch.utils.data import DataLoader
print("  -> imports OK", flush=True)

SEED = 42
DATA_ROOT = "D:\\kaggle-data\\for-norm"

print("B) build_file_index training...", flush=True)
sys.stdout.flush()
df_train = build_file_index(DATA_ROOT, split='training')
print(f"  -> {len(df_train)} files", flush=True)

print("C) build_file_index validation...", flush=True)
sys.stdout.flush()
df_val = build_file_index(DATA_ROOT, split='validation')
print(f"  -> {len(df_val)} files", flush=True)

print("D) build_file_index testing...", flush=True)
sys.stdout.flush()
df_test = build_file_index(DATA_ROOT, split='testing')
print(f"  -> {len(df_test)} files", flush=True)

print("E) DeepfakeAudioDataset (train)...", flush=True)
sys.stdout.flush()
ds_train = DeepfakeAudioDataset(df_train, feature_type='logmel', training=True)
print(f"  -> {len(ds_train)} samples", flush=True)

print("F) torch.Generator()...", flush=True)
sys.stdout.flush()
g = torch.Generator()
g.manual_seed(SEED)
print("  -> OK", flush=True)

print("G) DataLoader(train, num_workers=0)...", flush=True)
sys.stdout.flush()
train_loader = DataLoader(ds_train, batch_size=32, shuffle=True,
                          num_workers=0, pin_memory=False,
                          worker_init_fn=lambda wid: np.random.seed(SEED + wid),
                          generator=g)
print(f"  -> {len(train_loader)} batches", flush=True)

print("H) DeepfakeAudioDataset (val)...", flush=True)
sys.stdout.flush()
ds_val = DeepfakeAudioDataset(df_val, feature_type='logmel', training=False)
print(f"  -> {len(ds_val)} samples", flush=True)

print("I) DataLoader(val, num_workers=0)...", flush=True)
sys.stdout.flush()
g2 = torch.Generator()
g2.manual_seed(SEED)
val_loader = DataLoader(ds_val, batch_size=32, shuffle=False,
                        num_workers=0, pin_memory=False,
                        worker_init_fn=lambda wid: np.random.seed(SEED + wid),
                        generator=g2)
print(f"  -> {len(val_loader)} batches", flush=True)

print("J) DeepfakeAudioDataset (test)...", flush=True)
sys.stdout.flush()
ds_test = DeepfakeAudioDataset(df_test, feature_type='logmel', training=False)
g3 = torch.Generator()
g3.manual_seed(SEED)
test_loader = DataLoader(ds_test, batch_size=32, shuffle=False,
                         num_workers=0, pin_memory=False,
                         worker_init_fn=lambda wid: np.random.seed(SEED + wid),
                         generator=g3)
print(f"  -> {len(test_loader)} batches", flush=True)

print("K) fetch first train batch...", flush=True)
sys.stdout.flush()
X, y = next(iter(train_loader))
print(f"  -> X={X.shape}, y={y.shape}", flush=True)

print("\nALL STEPS PASSED!", flush=True)
