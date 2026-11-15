"""
Adversarial Vision Lab — Neo-Brutalism Demo UI (Mock Data Only)
----------------------------------------------------------------
Single-page Streamlit prototype inspired by Irene Adversarial Vision Lab,
styled in bold Neo-Brutalism for live academic demonstrations.

All predictions, perturbations, and defenses are simulated via `mock_data.py`.
"""

from __future__ import annotations

import base64
from io import BytesIO
import streamlit as st
from PIL import Image

from mock_data import (
    ATTACK_METHODS,
    DEFENSE_METHODS,
    MODELS,
    SAMPLES,
    PredictionState,
    compute_mock_analysis,
)

st.set_page_config(
    page_title="Adversarial Vision Lab · Demo Mode",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEO_BRUTALISM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@700;800;900&family=Public+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
  --bg-canvas: #FAF9F5;
  --bg-card: #FFFFFF;
  --ink: #1E242B;
  --ink-muted: #5A626E;
  --accent-yellow: #FEF3C7;
  --accent-blue: #60A5FA;
  --accent-coral: #FFE4DF;
  --accent-mint: #DCFCE7;
  --border-w: 2px;
  --shadow: 3px 3px 0px #1E242B;
  --shadow-sm: 2px 2px 0px #1E242B;
}

html, body, [class*="css"] {
  font-family: 'Public Sans', sans-serif;
  color: var(--ink);
}

.stApp {
  background-color: var(--bg-canvas);
}

.block-container {
  padding-top: 1.4rem !important;
  padding-bottom: 2.2rem !important;
  max-width: 1400px;
}

section[data-testid="stSidebar"] {
  background-color: #FFFFFF !important;
  border-right: 2px solid #1E242B !important;
}

section[data-testid="stSidebar"] .block-container {
  padding-top: 1.2rem !important;
}

div[data-testid="stExpander"] {
  background: #FAF9F5;
  border: 2px solid #1E242B !important;
  border-radius: 6px !important;
  box-shadow: 2px 2px 0px #1E242B;
}

div[data-baseweb="select"] > div,
div[data-testid="stFileUploaderDropzone"] {
  border: 2px solid #1E242B !important;
  border-radius: 6px !important;
  background-color: #FFFFFF !important;
  box-shadow: 2px 2px 0px #1E242B;
}

button[kind="primary"], button[kind="secondary"] {
  border: 2px solid #1E242B !important;
  border-radius: 6px !important;
  box-shadow: 2px 2px 0px #1E242B !important;
  font-family: 'Archivo', sans-serif !important;
  font-weight: 800 !important;
}

.nb-header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  background: #FFFFFF;
  border: 2px solid #1E242B;
  border-radius: 8px;
  box-shadow: 3px 3px 0px #1E242B;
  padding: 1rem 1.3rem;
  margin-bottom: 1.2rem;
}

.nb-title {
  font-family: 'Archivo', sans-serif;
  font-weight: 900;
  font-size: 1.7rem;
  line-height: 1.05;
  letter-spacing: -0.02em;
  margin: 0;
  color: #1E242B;
}

.nb-subtitle {
  font-size: 0.9rem;
  color: #4B5563;
  margin: 0.3rem 0 0 0;
  font-weight: 500;
}

.nb-demo-banner {
  background: #FEF3C7;
  border: 2px solid #1E242B;
  border-radius: 6px;
  box-shadow: 2px 2px 0px #1E242B;
  padding: 0.45rem 0.85rem;
  font-family: 'Space Mono', monospace;
  font-weight: 700;
  font-size: 0.78rem;
  color: #1E242B;
  white-space: nowrap;
}

.nb-card {
  background: #FFFFFF;
  border: 2px solid #1E242B;
  border-radius: 8px;
  box-shadow: 3px 3px 0px #1E242B;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  height: 100%;
}

.nb-card-band {
  border-bottom: 2px solid #1E242B;
  padding: 0.6rem 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: 'Archivo', sans-serif;
  font-weight: 800;
  font-size: 0.98rem;
  letter-spacing: 0.01em;
}

.band-original { background: #F1F5F9; color: #1E242B; }
.band-attacked { background: #FFE4DF; color: #1E242B; }
.band-defended { background: #DCFCE7; color: #1E242B; }

.nb-card-tag {
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  font-weight: 700;
  background: #FFFFFF;
  border: 1.5px solid #1E242B;
  border-radius: 4px;
  padding: 0.12rem 0.42rem;
}

.nb-img-wrap {
  padding: 1rem 1rem 0.7rem 1rem;
  background: #FFFFFF;
}

.nb-img-frame {
  border: 2px solid #1E242B;
  border-radius: 6px;
  background: #181A1F;
  aspect-ratio: 1 / 1;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.nb-img-frame img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.nb-card-body {
  padding: 0.3rem 1rem 1rem 1rem;
}

.nb-label-kicker {
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  font-weight: 700;
  color: #5A626E;
  text-transform: uppercase;
  margin-bottom: 0.15rem;
}

.nb-pred-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 0.5rem;
  margin-bottom: 0.45rem;
}

.nb-pred-class {
  font-family: 'Archivo', sans-serif;
  font-weight: 800;
  font-size: 1.32rem;
  line-height: 1.1;
  color: #1E242B;
}

.nb-conf-num {
  font-family: 'Space Mono', monospace;
  font-weight: 700;
  font-size: 1.1rem;
  color: #1E242B;
}

.nb-bar-track {
  width: 100%;
  height: 12px;
  background: #F3F2EC;
  border: 1.5px solid #1E242B;
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 0.75rem;
}

.nb-bar-fill {
  height: 100%;
  border-right: 1.5px solid #1E242B;
}

.nb-meta-box {
  background: #FAF9F5;
  border: 1.5px solid #1E242B;
  border-radius: 6px;
  padding: 0.55rem 0.75rem;
  font-family: 'Space Mono', monospace;
  font-size: 0.74rem;
  line-height: 1.5;
  color: #2C333D;
}

.nb-status-line {
  font-family: 'Space Mono', monospace;
  font-size: 0.72rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  color: #333A44;
}

.nb-summary {
  margin-top: 1.3rem;
  background: #FFFFFF;
  border: 2px solid #1E242B;
  border-radius: 8px;
  box-shadow: 3px 3px 0px #1E242B;
  overflow: hidden;
}

.nb-summary-head {
  background: #FEF3C7;
  border-bottom: 2px solid #1E242B;
  padding: 0.65rem 1.2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: 'Archivo', sans-serif;
  font-weight: 800;
  font-size: 0.98rem;
}

.nb-summary-body {
  padding: 1.1rem 1.25rem;
}

.nb-summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.85rem;
  margin-bottom: 0.95rem;
}

.nb-stat-block {
  background: #FAF9F5;
  border: 1.5px solid #1E242B;
  border-radius: 6px;
  padding: 0.6rem 0.8rem;
}

.nb-stat-k {
  font-family: 'Space Mono', monospace;
  font-size: 0.68rem;
  font-weight: 700;
  color: #5A626E;
  text-transform: uppercase;
}

.nb-stat-v {
  font-family: 'Archivo', sans-serif;
  font-weight: 800;
  font-size: 1rem;
  margin-top: 0.2rem;
  color: #1E242B;
}

.nb-explanation {
  border-top: 1px solid #D6D3C9;
  padding-top: 0.85rem;
  font-size: 0.94rem;
  line-height: 1.55;
  color: #333A44;
}

@media (max-width: 900px) {
  .nb-summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
"""

st.markdown(NEO_BRUTALISM_CSS, unsafe_allow_html=True)


def _img_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _render_panel_card(
    title: str,
    band_class: str,
    badge_text: str,
    image: Image.Image,
    pred: PredictionState,
    bar_color: str,
) -> None:
    b64 = _img_to_base64(image)
    conf_pct = max(0, min(100, int(round(pred.confidence * 100))))
    truth_note = ""
    if pred.is_correct is True:
        truth_note = " · GROUND TRUTH MATCH"
    elif pred.is_correct is False:
        truth_note = " · MISCLASSIFIED"

    meta_html = "<br>".join(pred.metadata_lines)

    html = f"""
    <div class="nb-card">
      <div class="nb-card-band {band_class}">
        <span>{title}</span>
        <span class="nb-card-tag">{badge_text}</span>
      </div>
      <div class="nb-img-wrap">
        <div class="nb-img-frame">
          <img src="data:image/png;base64,{b64}" alt="{title} sample preview" />
        </div>
      </div>
      <div class="nb-card-body">
        <div class="nb-label-kicker">Predicted Class</div>
        <div class="nb-pred-row">
          <div class="nb-pred-class">{pred.predicted_class}</div>
          <div class="nb-conf-num">{pred.confidence:.1%}</div>
        </div>
        <div class="nb-bar-track">
          <div class="nb-bar-fill" style="width: {conf_pct}%; background: {bar_color};"></div>
        </div>
        <div class="nb-status-line">{pred.status_tag}{truth_note}</div>
        <div class="nb-meta-box">{meta_html}</div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# --- Session State Defaults ---
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = "sample_cifar_plane"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "cifar10_resnet18"


# --- Left Configuration Sidebar ---
with st.sidebar:
    st.markdown(
        """
        <div style="border:2px solid #1E242B; border-radius:6px; background:#FEF3C7; padding:0.65rem 0.85rem; box-shadow:2px 2px 0px #1E242B; margin-bottom:1rem;">
          <div style="font-family:'Archivo',sans-serif; font-weight:800; font-size:0.95rem; line-height:1.1; color:#1E242B;">
            CONFIGURATION PANEL
          </div>
          <div style="font-family:'Space Mono',monospace; font-size:0.7rem; font-weight:700; margin-top:0.2rem; color:#4B5563;">
            MOCK SIMULATION CONTROLS
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**1. Input Image**")
    uploaded_file = st.file_uploader(
        "Upload custom image (replaces sample)",
        type=["png", "jpg", "jpeg", "webp"],
        help="Uploaded images stay in browser session memory for mock preview only.",
    )

    sample_keys = list(SAMPLES.keys())
    selected_sample = st.selectbox(
        "Or pick a built-in demo sample",
        options=sample_keys,
        index=sample_keys.index(st.session_state.selected_sample),
        format_func=lambda k: SAMPLES[k].label,
        disabled=uploaded_file is not None,
    )
    if selected_sample != st.session_state.selected_sample and uploaded_file is None:
        st.session_state.selected_sample = selected_sample
        st.session_state.selected_model = SAMPLES[selected_sample].model_id

    st.markdown("---")
    st.markdown("**2. Target Classifier (Mock)**")
    model_keys = list(MODELS.keys())
    selected_model = st.selectbox(
        "Model architecture",
        options=model_keys,
        index=model_keys.index(st.session_state.selected_model),
        format_func=lambda k: MODELS[k].name,
    )
    st.session_state.selected_model = selected_model

    st.markdown("---")
    st.markdown("**3. Adversarial Attack (Mock)**")
    attack_method = st.radio(
        "Attack method",
        options=ATTACK_METHODS,
        horizontal=True,
    )

    epsilon = st.slider(
        "Perturbation budget (Epsilon ε)",
        min_value=0.00,
        max_value=0.30,
        value=0.08,
        step=0.01,
        format="%.2f",
    )

    st.markdown("---")
    st.markdown("**4. Defense Mechanism (Mock)**")
    defense_method = st.selectbox(
        "Defense strategy",
        options=DEFENSE_METHODS,
        index=1,
    )

    with st.expander("Advanced Settings", expanded=False):
        pgd_steps = st.slider("PGD Steps", min_value=5, max_value=40, value=10, step=1)
        pgd_alpha = st.slider("PGD Step Size (α)", min_value=0.002, max_value=0.050, value=0.010, step=0.002, format="%.3f")
        pgd_restarts = st.slider("PGD Restarts", min_value=1, max_value=5, value=1, step=1)
        blur_radius = st.slider("Gaussian Blur σ", min_value=0.5, max_value=4.0, value=1.8, step=0.1)
        bit_depth = st.slider("Bit-depth Precision", min_value=1, max_value=6, value=3, step=1)
        visual_amp = st.slider("Visual Noise Amplification (Preview)", min_value=0.5, max_value=2.5, value=1.2, step=0.1)


# --- Header Bar ---
st.markdown(
    """
    <div class="nb-header">
      <div>
        <h1 class="nb-title">Adversarial Vision Lab</h1>
        <p class="nb-subtitle">Single-page attack &amp; defense comparison workspace · Inspired by Irene Adversarial Vision Lab</p>
      </div>
      <div class="nb-demo-banner">
        DEMO MODE · MOCK DATA ONLY
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Compute Mock Analysis ---
uploaded_pil = Image.open(uploaded_file) if uploaded_file is not None else None

result = compute_mock_analysis(
    model_id=selected_model,
    attack_method=attack_method,
    epsilon=epsilon,
    defense_method=defense_method,
    sample_id=selected_sample,
    uploaded_image=uploaded_pil,
    pgd_steps=pgd_steps,
    pgd_alpha=pgd_alpha,
    pgd_restarts=pgd_restarts,
    blur_radius=blur_radius,
    bit_depth=bit_depth,
    visual_amp=visual_amp,
)

# --- Three-Panel Comparison Layout ---
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    _render_panel_card(
        title="ORIGINAL",
        band_class="band-original",
        badge_text="CLEAN",
        image=result.original_image,
        pred=result.original_pred,
        bar_color="#60A5FA",
    )

with col2:
    _render_panel_card(
        title="ATTACKED",
        band_class="band-attacked",
        badge_text=f"{attack_method} · ε={epsilon:.2f}",
        image=result.attacked_image,
        pred=result.attacked_pred,
        bar_color="#F87171",
    )

with col3:
    _render_panel_card(
        title="DEFENDED",
        band_class="band-defended",
        badge_text=defense_method.upper(),
        image=result.defended_image,
        pred=result.defended_pred,
        bar_color="#34D399",
    )

# --- Analysis Summary Section ---
shift_text = (
    f"{result.original_pred.predicted_class} → "
    f"{result.attacked_pred.predicted_class} → "
    f"{result.defended_pred.predicted_class}"
)
conf_delta = result.defended_pred.confidence - result.attacked_pred.confidence
conf_delta_str = f"{conf_delta:+.1%}"

st.markdown(
    f"""
    <div class="nb-summary">
      <div class="nb-summary-head">
        <span>ANALYSIS SUMMARY</span>
        <span class="nb-card-tag">SIMULATED METRICS</span>
      </div>
      <div class="nb-summary-body">
        <div class="nb-summary-grid">
          <div class="nb-stat-block">
            <div class="nb-stat-k">Selected Attack</div>
            <div class="nb-stat-v">{attack_method} (ε = {epsilon:.2f})</div>
          </div>
          <div class="nb-stat-block">
            <div class="nb-stat-k">Selected Defense</div>
            <div class="nb-stat-v">{defense_method}</div>
          </div>
          <div class="nb-stat-block">
            <div class="nb-stat-k">Mock Prediction Shift</div>
            <div class="nb-stat-v">{shift_text}</div>
          </div>
          <div class="nb-stat-block">
            <div class="nb-stat-k">Defense Confidence Delta</div>
            <div class="nb-stat-v">{conf_delta_str} vs Attacked</div>
          </div>
        </div>
        <div class="nb-explanation">
          <strong>{result.summary_headline}</strong><br />
          {result.summary_explanation}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
