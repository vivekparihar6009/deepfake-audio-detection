# Deepfake Audio Detection — Demo Video Script (~2 minutes)

> **Format:** Screen recording with voiceover narration. Replace [ACCURACY], [EER], [F1] with actual metrics from `reports/performance_report.md`.

---

## Shot-by-Shot Script

| Timestamp | Shot | Visuals | Voiceover |
|-----------|------|---------|-----------|
| **0:00 – 0:15** | **1. Hook & Title** | Show DeepGuard Streamlit app landing page with the animated gradient title and empty audio drop zone. | *"AI-generated voices are now indistinguishable to the human ear. DeepGuard is a deep learning system that can detect them — in seconds."* |
| **0:15 – 0:30** | **2. Problem Statement** | Show spectrogram side-by-side: a genuine voice (blue) vs. a deepfake (red). Cut to a waveform comparison. | *"We built an end-to-end pipeline using CNN-BiLSTM neural networks trained on 69,000 audio clips from the Fake-or-Real benchmark dataset."* |
| **0:30 – 0:55** | **3. Genuine Sample Demo** | Upload a real human voice WAV. Show the audio player and waveform rendering. Click Analyze. Show the green "Genuine (Human)" badge with [ACCURACY_GENUINE]% confidence. | *"When we upload a genuine human voice recording, the model analyzes the Log-Mel spectrogram and correctly identifies it as authentic speech — with high confidence."* |
| **0:55 – 1:20** | **4. Deepfake Sample Demo** | Upload a synthetic voice WAV. Show the waveform looks similar but spectrogram differs in high-frequency detail. Show the red "Deepfake (AI-Generated)" badge with confidence. | *"Now with a deepfake voice sample. Despite sounding convincing to human ears, the spectrogram reveals subtle artifacts — uniform frequency patterns that AI synthesizers leave behind. Our model catches them."* |
| **1:20 – 1:40** | **5. Performance Report** | Open `reports/performance_report.md`. Zoom to the metric table: Accuracy [ACCURACY]%, EER [EER]%, F1 [F1]%. Show confusion matrix and ROC curve plots. | *"The model achieves [ACCURACY]% accuracy with an Equal Error Rate of [EER]% on the held-out test set — exceeding all performance thresholds."* |
| **1:40 – 1:55** | **6. Architecture Walkthrough** | Show CNN-BiLSTM diagram from the README. Briefly show `src/train.py` and `src/models.py` file structure. | *"The architecture combines 2D CNNs to extract spatial spectrogram features, with Bidirectional LSTMs to capture temporal speech dynamics — making it robust across diverse speaking styles."* |
| **1:55 – 2:00** | **7. Closing** | Show the GitHub repo structure and the Streamlit app URL in the browser bar. | *"The full pipeline — training, evaluation, and deployment — is reproducible via a single Colab notebook. DeepGuard: protecting trust in audio."* |

---

## Key Metrics to Fill In (Post-Evaluation)

After running `python src/evaluate.py ...`, fill in from `reports/performance_report.md`:

| Placeholder | Value |
|-------------|-------|
| `[ACCURACY]` | `__.__%` |
| `[EER]` | `__.__%` |
| `[F1]` | `__.__%` |
| `[ACCURACY_GENUINE]` | `__.__%` |
| `[ACCURACY_DEEPFAKE]` | `__.__%` |

---

## Production Notes for Recording

- **Screen Resolution:** 1920×1080 or higher
- **Recording Tool:** OBS Studio (free) or Loom
- **Audio:** Clear voiceover — no background noise
- **Duration:** Keep tight to 2 minutes
- **Demo Files:** Use real files from the FoR dataset testing split  
  - Genuine: `D:\kaggle-data\for-norm\for-norm\testing\real\<any>.wav`
  - Deepfake: `D:\kaggle-data\for-norm\for-norm\testing\fake\<any>.wav`
