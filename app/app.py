"""
app.py - Deepfake Audio Detection — Streamlit Web Application

Premium dark-mode UI with:
  - Audio file uploader (wav/mp3/flac)
  - Audio playback
  - Waveform + Log-Mel spectrogram visualization
  - Deepfake/Genuine classification with animated confidence gauge
  - Model loading via @st.cache_resource
"""

# Import pandas and numpy FIRST to prevent DLL conflict on Windows
import pandas as pd
import numpy as np

import os
import sys
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import soundfile as sf

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(SRC_DIR))

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepGuard — Deepfake Audio Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-card: rgba(255,255,255,0.04);
    --bg-card-hover: rgba(255,255,255,0.07);
    --accent-blue: #4f8ef7;
    --accent-purple: #8b5cf6;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: rgba(255,255,255,0.08);
    --gradient-hero: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp { background: var(--gradient-hero) !important; }

/* Hide default Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 1100px !important; }

/* Hero section */
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4f8ef7, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.3rem;
    line-height: 1.1;
}

.hero-sub {
    text-align: center;
    color: var(--text-secondary);
    font-size: 1.05rem;
    font-weight: 300;
    margin-bottom: 2.5rem;
}

/* Cards */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.2rem;
    transition: all 0.3s ease;
}
.glass-card:hover { background: var(--bg-card-hover); transform: translateY(-2px); }

/* Result badge */
.result-badge-genuine {
    display: inline-block;
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1.5px solid #10b981;
    color: #6ee7b7;
    padding: 0.6rem 1.5rem;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.result-badge-deepfake {
    display: inline-block;
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1.5px solid #ef4444;
    color: #fca5a5;
    padding: 0.6rem 1.5rem;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Confidence bar */
.conf-bar-container {
    background: rgba(255,255,255,0.06);
    border-radius: 50px;
    height: 16px;
    width: 100%;
    margin: 0.8rem 0;
    overflow: hidden;
}
.conf-bar-fill-genuine {
    background: linear-gradient(90deg, #059669, #10b981);
    height: 100%;
    border-radius: 50px;
    transition: width 0.8s ease;
}
.conf-bar-fill-deepfake {
    background: linear-gradient(90deg, #b91c1c, #ef4444);
    height: 100%;
    border-radius: 50px;
    transition: width 0.8s ease;
}

/* Section header */
.section-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.8rem;
}

/* Metric pill */
.metric-pill {
    background: rgba(79,142,247,0.12);
    border: 1px solid rgba(79,142,247,0.25);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    text-align: center;
    margin: 0.3rem;
}
.metric-pill .value { font-size: 1.4rem; font-weight: 700; color: #4f8ef7; }
.metric-pill .label { font-size: 0.75rem; color: var(--text-secondary); }

/* Upload area styling */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(79,142,247,0.35) !important;
    border-radius: 16px !important;
    background: rgba(79,142,247,0.04) !important;
    padding: 1rem !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model with caching. Downloads weights if not present locally."""
    import importlib
    import predict
    importlib.reload(predict)
    from predict import load_model_cached
    checkpoint = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pt')
    return load_model_cached(checkpoint_path=checkpoint)


# ─── Visualization helpers ─────────────────────────────────────────────────────

def plot_waveform(y: np.ndarray, sr: int = 16000) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    t = np.linspace(0, len(y) / sr, len(y))
    ax.plot(t, y, color='#4f8ef7', linewidth=0.8, alpha=0.9)
    ax.fill_between(t, y, alpha=0.2, color='#4f8ef7')
    ax.set_xlabel('Time (s)', color='#94a3b8', fontsize=9)
    ax.set_ylabel('Amplitude', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color((1.0, 1.0, 1.0, 0.15))
    ax.set_xlim([0, max(t)])
    plt.tight_layout(pad=0.5)
    return fig


def plot_spectrogram(y: np.ndarray, sr: int = 16000) -> plt.Figure:
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel',
                                    fmax=8000, ax=ax, cmap='magma')
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='#94a3b8', labelsize=8)
    cbar.set_label('dB', color='#94a3b8', fontsize=9)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8')
    
    ax.set_title('Log-Mel Spectrogram', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color((1.0, 1.0, 1.0, 0.15))
    plt.tight_layout(pad=0.5)
    return fig


def fig_to_base64_src(fig) -> str:
    import io
    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, transparent=True)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


def get_audio_base64_src(audio_path: str) -> str:
    import base64
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    mime_type = "audio/wav"
    if audio_path.endswith('.mp3'):
        mime_type = "audio/mp3"
    elif audio_path.endswith('.flac'):
        mime_type = "audio/flac"
    return f"data:{mime_type};base64,{audio_b64}"


# ─── App Layout ────────────────────────────────────────────────────────────────

def main():
    # Hero header
    st.markdown('<div class="hero-title">🎙️ DeepGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AI-Powered Deepfake Audio Detection · Genuine vs. Synthetic Speech</div>', unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🧠 Model Info")
        st.markdown("""
        **Architecture:** CNN-BiLSTM  
        **Feature:** Log-Mel Spectrogram  
        **Parameters:** 3.8M  
        **Training set:** 53,868 audio clips  
        **Classes:** Genuine · Deepfake
        """)
        st.markdown("---")
        st.markdown("### ⚙️ How It Works")
        st.markdown("""
        1. **Load** audio (resample to 16kHz)
        2. **Trim** silence (energy-based)
        3. **Extract** 128-band Log-Mel spectrogram
        4. **CNN** extracts spatial patterns
        5. **BiLSTM** models temporal context
        6. **Classify** as Genuine or Deepfake
        """)
        st.markdown("---")
        st.markdown("### 📊 Performance Targets")
        st.markdown("""
        - ✅ Accuracy ≥ 80%  
        - ✅ EER ≤ 12%  
        - ✅ F1 Score ≥ 80%  
        - ✅ Per-class Acc ≥ 75%
        """)
        st.markdown("---")
        st.caption("Built with PyTorch · Streamlit · librosa")

    # Load model (show spinner only first time)
    model_loaded = False
    try:
        with st.spinner("Loading detection model..."):
            model = load_model()
        model_loaded = True
    except FileNotFoundError:
        st.warning("⚠️ No trained model found at `models/best_model.pt`. Please train the model first: `python src/train.py --data_dir <path>`")

    # ── Upload section ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Upload Audio</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a voice recording here — WAV, MP3, or FLAC",
        type=["wav", "mp3", "flac"],
        label_visibility="collapsed"
    )

    if uploaded is not None:
        # Save to temp file
        suffix = '.' + uploaded.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        col1, col2 = st.columns([1.1, 1])

        # ── Left column: audio uploader and visualizations ───────────────────
        with col1:
            try:
                import librosa
                y_vis, sr_vis = librosa.load(tmp_path, sr=16000, mono=True)
                
                audio_src = get_audio_base64_src(tmp_path)
                fig_wave = plot_waveform(y_vis, sr_vis)
                wave_src = fig_to_base64_src(fig_wave)
                
                fig_spec = plot_spectrogram(y_vis, sr_vis)
                spec_src = fig_to_base64_src(fig_spec)

                col1_html = f"""<div class="glass-card">
<div class="section-label">🔊 Audio Playback</div>
<audio src="{audio_src}" controls style="width:100%; margin-bottom:1.5rem; border-radius: 8px;"></audio>
<div class="section-label" style="margin-top:1rem;">Waveform</div>
<img src="{wave_src}" style="width:100%; border-radius:8px; margin-bottom:1.5rem;" />
<div class="section-label">Log-Mel Spectrogram</div>
<img src="{spec_src}" style="width:100%; border-radius:8px;" />
</div>"""
                st.markdown(col1_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Visualization error: {e}")

        # ── Right column: prediction results ─────────────────────────────────
        with col2:
            if model_loaded:
                with st.spinner("Analyzing audio..."):
                    try:
                        import importlib
                        import predict
                        importlib.reload(predict)
                        from predict import predict as run_predict
                        result = run_predict(
                            audio_path=tmp_path,
                            checkpoint_path=os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pt'),
                            model=model
                        )

                        label      = result['label']
                        confidence = result['confidence']
                        is_fake    = result['label_id'] == 1
                        gen_prob   = result['genuine_prob']
                        fake_prob  = result['deepfake_prob']

                        # Result badge styling
                        badge_class = 'result-badge-deepfake' if is_fake else 'result-badge-genuine'
                        icon = '⚠️' if is_fake else '✅'
                        bar_class = 'conf-bar-fill-deepfake' if is_fake else 'conf-bar-fill-genuine'
                        text_color = '#ef4444' if is_fake else '#10b981'
                        
                        interpretation_html = f"""<div style="background-color: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); padding: 1rem; border-radius: 8px; color: #fca5a5; margin-top: 1.2rem; font-size:0.9rem;">
⚠️ <strong>Deepfake Detected</strong> — This audio appears to be AI-generated or voice-cloned.
</div>""" if is_fake else f"""<div style="background-color: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding: 1rem; border-radius: 8px; color: #6ee7b7; margin-top: 1.2rem; font-size:0.9rem;">
✅ <strong>Genuine Voice</strong> — This audio appears to be authentic human speech.
</div>"""

                        col2_html = f"""<div class="glass-card">
<div class="section-label">🔍 Analysis Result</div>
<div style="text-align:center;margin:1.5rem 0;">
<span class="{badge_class}">{icon} {label}</span>
</div>
<div style="text-align:center;font-size:2.5rem;font-weight:700;color:{text_color};line-height:1.2;">
{confidence:.1f}%
</div>
<div style="text-align:center;color:#94a3b8;font-size:0.85rem;margin-bottom:1rem;">
Confidence
</div>
<div class="conf-bar-container">
<div class="{bar_class}" style="width:{confidence}%"></div>
</div>
<div style="margin-top:1.8rem">
<div class="section-label">Probability Breakdown</div>
<div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
<div class="metric-pill" style="flex: 1;">
<div class="value" style="color:#10b981">{gen_prob*100:.1f}%</div>
<div class="label">Genuine</div>
</div>
<div class="metric-pill" style="flex: 1;">
<div class="value" style="color:#ef4444">{fake_prob*100:.1f}%</div>
<div class="label">Deepfake</div>
</div>
</div>
</div>
{interpretation_html}
</div>"""
                        st.markdown(col2_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Inference error: {e}")
            else:
                st.info("Model not loaded. Please train the model first.")

        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    else:
        # Empty state
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">🎤</div>
            <div style="font-size:1.1rem;color:#94a3b8;margin-bottom:0.5rem;">
                Upload an audio file to begin analysis
            </div>
            <div style="font-size:0.85rem;color:#475569;">
                Supported formats: WAV · MP3 · FLAC
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#475569;font-size:0.8rem;">'
        'DeepGuard · Built with PyTorch · CNN-BiLSTM · Log-Mel Spectrograms · '
        '<a href="https://github.com" style="color:#4f8ef7;text-decoration:none;">GitHub</a>'
        '</div>', unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
