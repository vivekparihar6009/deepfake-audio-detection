"""isolate_dataloader.py - find the exact crash line inside get_dataloaders"""
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("STEP 1: importing numpy, torch", flush=True)
import numpy as np
import torch

print("STEP 2: importing pandas", flush=True)
import pandas as pd

print("STEP 3: importing librosa", flush=True)
import librosa

print("STEP 4: importing soundfile", flush=True)
import soundfile as sf

print("STEP 5: importing preprocessing", flush=True)
from preprocessing import preprocess_audio, extract_features

print("STEP 6: importing dataset module pieces", flush=True)
from dataset import build_file_index, DeepfakeAudioDataset

print("STEP 7: building file index for training split", flush=True)
try:
    df_train = build_file_index("D:\\kaggle-data\\for-norm", split='training')
    print(f"  -> training index: {len(df_train)} files", flush=True)
except Exception as e:
    print(f"  -> FAILED: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("STEP 8: building file index for validation split", flush=True)
try:
    df_val = build_file_index("D:\\kaggle-data\\for-norm", split='validation')
    print(f"  -> validation index: {len(df_val)} files", flush=True)
except Exception as e:
    print(f"  -> FAILED: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("STEP 9: building file index for testing split", flush=True)
try:
    df_test = build_file_index("D:\\kaggle-data\\for-norm", split='testing')
    print(f"  -> testing index: {len(df_test)} files", flush=True)
except Exception as e:
    print(f"  -> FAILED: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("STEP 10: creating DeepfakeAudioDataset (train)", flush=True)
try:
    ds_train = DeepfakeAudioDataset(df_train, feature_type='logmel', training=True)
    print(f"  -> dataset length: {len(ds_train)}", flush=True)
except Exception as e:
    print(f"  -> FAILED: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("STEP 11: creating DataLoader (train, num_workers=0)", flush=True)
try:
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(ds_train, batch_size=32, shuffle=True,
                              num_workers=0, pin_memory=False, generator=g)
    print(f"  -> DataLoader created, {len(train_loader)} batches", flush=True)
except Exception as e:
    print(f"  -> FAILED: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("STEP 12: fetching ONE batch from train_loader", flush=True)
try:
    it = iter(train_loader)
    X, y = next(it)
    print(f"  -> batch shape: X={X.shape}, y={y.shape}", flush=True)
except Exception as e:
    print(f"  -> FAILED: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("ALL STEPS PASSED - dataloader is working!", flush=True)
