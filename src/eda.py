"""
eda.py - Exploratory Data Analysis (EDA) for Deepfake Audio Detection.
Analyzes class balance, file count, duration distribution, and generates waveform + spectrogram plots.
"""

# Import pandas and numpy FIRST to prevent DLL conflicts on Windows
import pandas as pd
import numpy as np

import os
import sys
import time
import random
import argparse
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt

# Add src to path if needed
sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_file_index
from preprocessing import SAMPLE_RATE

def analyze_dataset(data_dir, reports_dir="reports"):
    print("=== Deepfake Audio Detection EDA ===")
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Build file indexes
    print("\nBuilding file indices...")
    df_train = build_file_index(data_dir, split='training')
    df_val = build_file_index(data_dir, split='validation')
    df_test = build_file_index(data_dir, split='testing')
    
    total_files = len(df_train) + len(df_val) + len(df_test)
    print(f"\nTotal Files: {total_files}")
    print(f"  - Training: {len(df_train)} files")
    print(f"  - Validation: {len(df_val)} files")
    print(f"  - Testing: {len(df_test)} files")
    
    # 2. Analyze class balance in training set
    train_real = (df_train['label'] == 0).sum()
    train_fake = (df_train['label'] == 1).sum()
    print(f"\nClass Balance (Training Set):")
    print(f"  - Real (Genuine): {train_real} ({train_real/len(df_train)*100:.2f}%)")
    print(f"  - Fake (Deepfake): {train_fake} ({train_fake/len(df_train)*100:.2f}%)")
    
    # 3. Analyze duration distribution of a sample subset
    # Reading all 50k+ files' duration might take several minutes, so we subsample 500 files
    print("\nEstimating audio duration distribution (subsampling 500 files)...")
    sample_paths = df_train.sample(500, random_state=42)['path'].tolist()
    durations = []
    
    for p in sample_paths:
        try:
            info = sf.info(p)
            durations.append(info.duration)
        except Exception as e:
            print(f"Warning: could not read info for {p}: {e}")
            
    durations = np.array(durations)
    print(f"Duration Statistics:")
    print(f"  - Min: {durations.min():.2f}s")
    print(f"  - Max: {durations.max():.2f}s")
    print(f"  - Mean: {durations.mean():.2f}s")
    print(f"  - Median: {np.median(durations):.2f}s")
    
    # 4. Generate waveform and spectrogram plots for a Real and a Fake file
    print("\nGenerating waveform & spectrogram plots...")
    real_sample_path = df_train[df_train['label'] == 0].iloc[0]['path']
    fake_sample_path = df_train[df_train['label'] == 1].iloc[0]['path']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Load samples
    y_real, sr_real = librosa.load(real_sample_path, sr=SAMPLE_RATE)
    y_fake, sr_fake = librosa.load(fake_sample_path, sr=SAMPLE_RATE)
    
    # Waveform plots
    times_real = np.arange(len(y_real)) / SAMPLE_RATE
    axes[0, 0].plot(times_real, y_real, color='royalblue', alpha=0.8)
    axes[0, 0].set_title("Waveform: Real (Genuine) Audio")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    times_fake = np.arange(len(y_fake)) / SAMPLE_RATE
    axes[0, 1].plot(times_fake, y_fake, color='crimson', alpha=0.8)
    axes[0, 1].set_title("Waveform: Fake (Deepfake) Audio")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # Spectrogram plots
    S_real = librosa.feature.melspectrogram(y=y_real, sr=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512)
    S_real_db = librosa.power_to_db(S_real, ref=np.max)
    img_real = librosa.display.specshow(S_real_db, sr=SAMPLE_RATE, hop_length=512, x_axis='time', y_axis='mel', ax=axes[1, 0], fmax=SAMPLE_RATE//2)
    fig.colorbar(img_real, ax=axes[1, 0], format='%+2.0f dB')
    axes[1, 0].set_title("Mel-Spectrogram: Real (Genuine)")
    
    S_fake = librosa.feature.melspectrogram(y=y_fake, sr=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512)
    S_fake_db = librosa.power_to_db(S_fake, ref=np.max)
    img_fake = librosa.display.specshow(S_fake_db, sr=SAMPLE_RATE, hop_length=512, x_axis='time', y_axis='mel', ax=axes[1, 1], fmax=SAMPLE_RATE//2)
    fig.colorbar(img_fake, ax=axes[1, 1], format='%+2.0f dB')
    axes[1, 1].set_title("Mel-Spectrogram: Fake (Deepfake)")
    
    plt.tight_layout()
    plot_path = os.path.join(reports_dir, "eda_waveform_spectrogram.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plots saved to {plot_path}")
    print("\nEDA Completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA on Deepfake Audio Dataset.")
    parser.add_argument("--data_dir", default="D:\\kaggle-data\\for-norm", help="Path to data directory containing for-norm")
    args = parser.parse_args()
    
    analyze_dataset(args.data_dir)
