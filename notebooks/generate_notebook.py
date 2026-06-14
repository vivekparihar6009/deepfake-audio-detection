import json
import os

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Deepfake Audio Detection: Genuine vs. AI-Generated Speech\n",
                    "\n",
                    "This notebook implements and runs a complete end-to-end deep learning pipeline to distinguish genuine human voices from synthetic, AI-generated speech using the Fake-or-Real (FoR) dataset.\n",
                    "\n",
                    "### Pipeline Overview:\n",
                    "1. **Environment Setup & Dependencies**\n",
                    "2. **Dataset Download & Extraction** (via Kaggle API)\n",
                    "3. **Exploratory Data Analysis (EDA)** (Waveform & Spectrogram analysis)\n",
                    "4. **Model Training** (CNN-BiLSTM with Log-Mel Spectrogram features)\n",
                    "5. **Evaluation** (EER, F1, Accuracy, ROC, DET curves)\n",
                    "6. **Inference** (Predicting on single audio files)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Phase 1: Environment Setup & Dependencies\n",
                    "\n",
                    "First, we clone the repository and install all required python libraries. If running in Google Colab, make sure to enable the **GPU** runtime under *Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU*."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Clone repo (adjust URL if needed)\n",
                    "!git clone https://github.com/vivekparihar6009/deepfake-audio-detection.git\n",
                    "%cd deepfake-audio-detection"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install dependencies\n",
                    "!pip install -q -r requirements.txt\n",
                    "# Install system libraries for audio on Colab\n",
                    "!apt-get install -y libsndfile1 ffmpeg > /dev/null 2>&1"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Phase 2: Dataset Download & Extraction\n",
                    "\n",
                    "To download the Fake-or-Real dataset programmatically, you need to upload your Kaggle API token (`kaggle.json`).\n",
                    "\n",
                    "Run the cell below to upload your `kaggle.json` file."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from google.colab import files\n",
                    "import os\n",
                    "\n",
                    "if not os.path.exists('/root/.kaggle/kaggle.json'):\n",
                    "    print(\"Please upload your kaggle.json file:\")\n",
                    "    uploaded = files.upload()\n",
                    "    for fn in uploaded.keys():\n",
                    "        os.makedirs('/root/.kaggle', exist_ok=True)\n",
                    "        with open('/root/.kaggle/kaggle.json', 'wb') as f:\n",
                    "            f.write(uploaded[fn])\n",
                    "        os.chmod('/root/.kaggle/kaggle.json', 0o600)\n",
                    "        print(\"Kaggle token saved successfully!\")\n",
                    "else:\n",
                    "    print(\"Kaggle token already exists!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Download the dataset using download_data.py\n",
                    "!python data/download_data.py --data_dir ./data/"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Phase 3: Exploratory Data Analysis (EDA)\n",
                    "\n",
                    "We run the EDA script to check dataset size, class balance, duration distribution, and plot sample waveforms + spectrograms."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run EDA script\n",
                    "!python src/eda.py --data_dir ./data/"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Display generated EDA plots\n",
                    "from IPython.display import Image, display\n",
                    "display(Image('reports/eda_waveform_spectrogram.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Phase 4: Model Training\n",
                    "\n",
                    "We train the CNN-BiLSTM classifier using Log-Mel spectrogram features. The script uses Cosine Annealing learning rate scheduling and early stopping by validation Equal Error Rate (EER)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Train model (using a subset of 10,000 files for faster training on Colab, run on GPU)\n",
                    "!python src/train.py --data_dir ./data/ --arch cnn_bilstm --feature logmel --epochs 10 --batch_size 64 --num_workers 2 --max_samples 10000"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Phase 5: Evaluation\n",
                    "\n",
                    "We evaluate our best model checkpoint on the testing set to calculate Accuracy, EER, and F1 Score, and plot the Confusion Matrix, ROC, and DET curves."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run evaluation on the test split\n",
                    "!python src/evaluate.py --data_dir ./data/ --checkpoint models/best_model_cnn_bilstm_logmel.pt --arch cnn_bilstm --feature logmel"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Display evaluation report\n",
                    "with open('reports/performance_report.md', 'r') as f:\n",
                    "    print(f.read())"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Display ROC & Confusion Matrix plots\n",
                    "display(Image('reports/roc_curve_testing.png'))\n",
                    "display(Image('reports/confusion_matrix_testing.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Phase 6: Single File Inference\n",
                    "\n",
                    "We can use `predict.py` to classify any local audio recording."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Predict on a sample wav file (e.g. from testing set)\n",
                    "import glob\n",
                    "test_files = glob.glob('data/for-norm/testing/**/*.wav', recursive=True)\n",
                    "if test_files:\n",
                    "    sample = test_files[0]\n",
                    "    !python src/predict.py --audio \"{sample}\" --checkpoint models/best_model.pt\n",
                    "else:\n",
                    "    print(\"No test files found, please ensure dataset is downloaded.\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    os.makedirs('notebooks', exist_ok=True)
    out_path = 'notebooks/deepfake_audio_detection.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created notebook at: {out_path}")

if __name__ == "__main__":
    create_notebook()
