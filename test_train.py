"""Minimal training test to debug errors."""
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Step 1: Importing modules...")
    from dataset import get_dataloaders
    from models import get_model
    from evaluate import compute_eer
    import torch
    print("  Imports OK")

    print("Step 2: Building dataloaders (max_samples=16)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        r'D:\kaggle-data\for-norm',
        feature_type='logmel',
        batch_size=32,
        num_workers=0,
        max_samples=200
    )
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")

    print("Step 3: Loading a batch...")
    X, y = next(iter(train_loader))
    print(f"  Batch X: {X.shape}, y: {y.shape}")

    print("Step 4: Creating model...")
    model = get_model('cnn_bilstm', feature_type='logmel')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  Model on {device}")

    print("Step 5: Forward pass...")
    X = X.to(device)
    logits = model(X)
    print(f"  Output: {logits.shape}")

    print("Step 6: Loss computation...")
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y.to(device))
    print(f"  Loss: {loss.item():.4f}")

    print("Step 7: Backward pass...")
    loss.backward()
    print("  Backward OK")

    print("\n[OK] ALL STEPS PASSED!")

except Exception as e:
    print(f"\n[ERROR] ERROR at current step:")
    traceback.print_exc()
    sys.exit(1)
