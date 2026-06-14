# DeepGuard: Deepfake Audio Detection

> **AI-Powered Voice Authentication** — Distinguishing genuine human speech from AI-generated synthetic voices using deep learning.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Description & Problem Statement

Voice cloning and AI-generated speech synthesis have become so convincing that human ears can no longer reliably distinguish genuine voices from deepfakes. This presents serious risks in authentication, misinformation, and fraud.

**DeepGuard** is an end-to-end deep learning pipeline that addresses this challenge by:
- Extracting rich **Log-Mel Spectrograms** that capture the distinctive spectro-temporal artifacts of synthetic speech
- Training a **CNN-BiLSTM** hybrid that combines spatial pattern recognition (CNN) with sequential context modelling (BiLSTM)
- Achieving robust classification under diverse acoustic conditions via multi-stage data augmentation

---

## 2. Dataset Description

We use the **Fake-or-Real (FoR) Dataset** — a curated benchmark of genuine and AI-generated speech from TTS systems.

| Split | Real | Fake | Total |
|-------|------|------|-------|
| Training | 26,941 | 26,927 | **53,868** |
| Validation | 5,400 | 5,398 | **10,798** |
| Testing | 2,264 | 2,370 | **4,634** |

- **Source:** [Kaggle FoR Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
- **Format:** WAV files, 16 kHz, mono
- **Duration:** 0.39s – 9.86s (mean 3.20s)
- **Balance:** Near-perfect 50/50 split (genuine vs. deepfake)

---

## 3. Preprocessing Pipeline

Each audio file undergoes the following pipeline before feature extraction:

1. **Load & Resample:** Load file with `librosa`, resample to 16 kHz mono
2. **Silence Trimming:** Energy-based trimming using `librosa.effects.trim(top_db=20)` — removes leading/trailing silence
3. **Fixed Length Normalization:** Random crop (training) or center crop (eval) to exactly **3.0 seconds** (48,000 samples). Shorter clips are zero-padded at the end.
4. **Amplitude Normalization:** Per-sample Z-score normalization of feature maps

---

## 4. Feature Extraction

| Feature | Shape | Description |
|---------|-------|-------------|
| **Log-Mel Spectrogram** | `(1, 128, 94)` | 128 Mel filterbanks, N_FFT=1024, Hop=512, normalized to Z-score. Captures perceptual frequency content. |
| **LFCC** | `(1, 40, 94)` | Linear Frequency Cepstral Coefficients — captures spoofing artifacts in the linear frequency domain. |

**Data Augmentation (Training only):**
- **Gaussian Noise:** Random SNR between 15–40 dB (50% probability)
- **Codec Simulation:** G.711-style downsampling to 8kHz + mu-law companding + upsample (30%)
- **Time Stretching:** Rate sampled from [0.8, 1.2] (30%)
- **Pitch Shifting:** Random ±2 semitones (30%)
- **SpecAugment:** 2 frequency masks (up to 20 bins) + 2 time masks (up to 15 frames)

---

## 5. Model Architecture

### CNN-BiLSTM (Recommended)
The primary model combines a 2D CNN frontend with a Bidirectional LSTM backend:

```
Input: (B, 1, 128, 94)
│
├── CNNBlock(1 → 32, 3×3) + BN + ReLU + MaxPool(2×2)    → (B, 32, 64, 47)
├── CNNBlock(32 → 64, 3×3) + BN + ReLU + MaxPool(2×2)   → (B, 64, 32, 23)
├── CNNBlock(64 → 128, 3×3) + BN + ReLU + MaxPool(2×2)  → (B, 128, 16, 11)
│
├── Permute + Reshape → (B, T_seq, C_freq×128)
├── BiLSTM(hidden=256, layers=2, bidirectional=True)     → (B, T, 512)
├── Mean Pool over time                                   → (B, 512)
│
├── Dropout(0.3)
├── Linear(512 → 256) + ReLU + Dropout(0.3)
└── Linear(256 → 2) → logits
```

**Total parameters: 3,815,011**

### ResNet18 (Baseline)
Standard ResNet18 adapted for 1-channel spectrogram input with a custom classification head.

---

## 6. Training Setup & Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| LR Schedule | Cosine Annealing (T_max=30, η_min=1e-6) |
| Weight Decay | 1e-4 |
| Batch Size | 64 |
| Epochs | 30 (with early stopping) |
| Early Stopping Patience | 7 epochs |
| Loss | CrossEntropyLoss (class-weighted) |
| Gradient Clipping | max_norm=5.0 |
| Random Seed | 42 |

---

## 7. Results & Performance

Performance on the **testing split** after full training:

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Overall Accuracy | *see reports/performance_report.md* | ≥ 80% | — |
| Equal Error Rate (EER) | *see reports/performance_report.md* | ≤ 12% | — |
| F1 Score | *see reports/performance_report.md* | ≥ 80% | — |
| Genuine Accuracy | *see reports/performance_report.md* | ≥ 75% | — |
| Deepfake Accuracy | *see reports/performance_report.md* | ≥ 75% | — |

> Full results including ROC curve, DET curve, and confusion matrix are in [`reports/performance_report.md`](reports/performance_report.md).

---

## 8. How to Run

### Prerequisites
```bash
# Clone repository
git clone https://github.com/vivekparihar6009/deepfake-audio-detection.git
cd deepfake-audio-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset
```bash
# Place kaggle.json credentials in ~/.kaggle/kaggle.json first
python data/download_data.py --data_dir D:\kaggle-data\
```

### EDA
```bash
python src/eda.py --data_dir D:\kaggle-data\for-norm
```

### Train
```bash
# On Windows (required env var to avoid OpenMP conflict)
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python src/train.py --data_dir D:\kaggle-data\for-norm --arch cnn_bilstm --feature logmel --epochs 30 --batch_size 64

# On Linux/Colab
python src/train.py --data_dir /content/data --arch cnn_bilstm --feature logmel --epochs 30 --batch_size 64 --num_workers 2
```

### Evaluate
```bash
python src/evaluate.py --checkpoint models/best_model_cnn_bilstm_logmel.pt --data_dir D:\kaggle-data\for-norm --arch cnn_bilstm --feature logmel
```

### Predict (Single File)
```bash
python src/predict.py --audio path/to/audio.wav --checkpoint models/best_model.pt
```

### Run Streamlit App Locally
```bash
cd app
streamlit run app.py
```

### Run Google Colab Notebook
Open [`notebooks/deepfake_audio_detection.ipynb`](notebooks/deepfake_audio_detection.ipynb) in Colab and run all cells (no manual edits required).

---

## 9. Repository Structure

```
deepfake-audio-detection/
├── data/
│   └── download_data.py          # Kaggle dataset downloader
├── src/
│   ├── preprocessing.py          # Audio loading, resampling, feature extraction
│   ├── dataset.py                # PyTorch Dataset + DataLoader factory
│   ├── models.py                 # CNN-BiLSTM and ResNet18 architectures
│   ├── train.py                  # Training script with early stopping
│   ├── evaluate.py               # Metrics computation + plot generation
│   ├── predict.py                # CLI and API inference script
│   ├── metrics.py                # Pure EER computation (no matplotlib)
│   └── eda.py                    # Exploratory Data Analysis script
├── app/
│   ├── app.py                    # Streamlit web application
│   └── requirements.txt          # Streamlit Cloud dependencies
├── notebooks/
│   └── deepfake_audio_detection.ipynb  # End-to-end runnable Colab notebook
├── models/
│   └── best_model.pt             # Best model checkpoint (saved after training)
├── reports/
│   ├── performance_report.md     # Auto-generated metrics report
│   ├── confusion_matrix_testing.png
│   ├── roc_curve_testing.png
│   ├── det_curve_testing.png
│   └── eda_waveform_spectrogram.png
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 10. Limitations & Future Work

### Current Limitations
- **Single Language:** Trained only on English speech. May not generalize well to other languages.
- **Dataset Scope:** FoR dataset covers a limited range of TTS systems. Newer GAN/diffusion-based voice cloners (e.g., VALL-E, ElevenLabs) may evade detection.
- **Cross-Dataset Generalization:** ASVspoof 2019 LA evaluation data was not used for training, so cross-dataset performance may vary.
- **Short Clips:** Model processes 3-second windows. Long audio files are center-cropped rather than processed holistically.

### Future Work
- **ASVspoof 2019 Integration:** Cross-dataset evaluation to assess generalization
- **Multi-lingual Extension:** Fine-tune on multilingual voice datasets
- **Ensemble Models:** Combine CNN-BiLSTM (logmel) + ResNet18 (LFCC) predictions
- **Real-time Streaming:** Implement sliding-window inference for live audio streams
- **Explainability:** Grad-CAM visualization of discriminative spectro-temporal regions
- **Adversarial Robustness:** Test against adaptive adversarial attacks on the detection system
