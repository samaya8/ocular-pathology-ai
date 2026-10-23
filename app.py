"""Point-of-Care Clinical Screening & Triage Station (Streamlit Application)."""

import io
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit as st

from src.preprocessor import CLASS_LABELS, CLASS_DISPLAY_NAMES, RetinaPreprocessor
from src.retina_engine import RetinaEngine

st.set_page_config(
    page_title="OcularMed AI | Retinal Pathology Diagnostic Suite",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_DIR = Path(__file__).resolve().parent / "data" / "sample_fundus"

# Custom CSS styling for medical interface
st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.2rem; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    .badge-urgent { background-color: #FEE2E2; color: #991B1B; padding: 4px 10px; border-radius: 9999px; font-weight: 600; }
    .badge-routine { background-color: #FEF3C7; color: #92400E; padding: 4px 10px; border-radius: 9999px; font-weight: 600; }
    .badge-normal { background-color: #DCFCE7; color: #166534; padding: 4px 10px; border-radius: 9999px; font-weight: 600; }
    .disclaimer-box { background-color: #FFFBEB; border-left: 4px solid #F59E0B; padding: 12px; margin-bottom: 16px; font-size: 0.9rem; color: #92400E; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">👁️ Retinal CAD Screening & Clinical Triage Station</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Computer-aided multi-class retinal screening with Monte Carlo Dropout uncertainty and Grad-CAM pathology localization.</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ Investigational Research Protocol:</strong> This software is designed strictly for computer-aided clinical decision support and educational demonstration. It is not an autonomous diagnostic device. All recommendations must be confirmed by a board-certified ophthalmologist.
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_retina_engine():
    return RetinaEngine()


engine = get_retina_engine()

# --- Sidebar Controls ---
st.sidebar.header("🔬 Examination Settings")

sample_files = {
    "Cataract (Lens Opacity)": SAMPLE_DIR / "cataract.png",
    "Diabetic Retinopathy (Exudates)": SAMPLE_DIR / "diabetic_retinopathy.png",
    "Glaucoma (Optic Cupping)": SAMPLE_DIR / "glaucoma.png",
    "Normal Retina (Healthy Control)": SAMPLE_DIR / "normal.png",
}

input_mode = st.sidebar.radio("Select Input Source", ["Clinical Preset Case", "Upload Patient Fundus Scan"])
selected_image: Image.Image | None = None
source_name = "Patient Exam"

if input_mode == "Clinical Preset Case":
    preset_choice = st.sidebar.selectbox("Choose Standard Case", list(sample_files.keys()))
    preset_path = sample_files[preset_choice]
    if preset_path.exists():
        selected_image = Image.open(preset_path).convert("RGB")
        source_name = preset_choice
else:
    uploaded = st.sidebar.file_uploader("Upload Retinal Photograph", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        selected_image = Image.open(uploaded).convert("RGB")
        source_name = uploaded.name

st.sidebar.markdown("---")
st.sidebar.subheader("Diagnostic Parameters")
apply_clahe = st.sidebar.checkbox("Enhance Contrast (CLAHE)", value=False, help="Enhances microvascular contrast on the green channel")
run_mc = st.sidebar.checkbox("Run Monte Carlo Dropout", value=True, help="Samples posterior predictive distribution across stochastic passes")
mc_passes = st.sidebar.slider("MC Dropout Passes", min_value=5, max_value=30, value=12, step=1, disabled=not run_mc)

st.sidebar.markdown("---")
st.sidebar.subheader("Explainability (Grad-CAM)")
cam_colormap = st.sidebar.selectbox("Heatmap Colormap", ["jet", "turbo", "viridis"], index=0)
cam_alpha = st.sidebar.slider("Overlay Opacity (α)", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

if selected_image is None:
    st.info("👈 Please select a clinical preset case or upload a retinal fundus image from the sidebar to begin screening.")
    st.stop()

# --- Execute Analysis ---
with st.spinner("Analyzing retinal pathology, computing Monte Carlo uncertainty, and generating Grad-CAM..."):
    res = engine.analyze_image(
        selected_image,
        run_mc_dropout=run_mc,
        run_gradcam=True,
        apply_clahe=apply_clahe,
        num_mc_samples=mc_passes,
    )
    # Update Grad-CAM overlay opacity if customized
    if cam_alpha != res["gradcam"]["blend_alpha"] or cam_colormap != res["gradcam"].get("colormap"):
        from src.gradcam_explainer import GradCAMExplainer
        re_cam = GradCAMExplainer(colormap=cam_colormap, alpha=cam_alpha).compute_gradcam(
            engine.model,
            engine.preprocessor.preprocess_image(selected_image, apply_clahe=apply_clahe),
            selected_image,
            target_class=res["predicted_class"],
        )
        res["gradcam"] = re_cam

quality = res["image_quality"]
triage = res["clinical_triage"]
unc = res["uncertainty"]
cam = res["gradcam"]

# --- Top Visual Display ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 📸 Original Fundus Photo")
    st.image(selected_image, use_container_width=True, caption=f"{source_name} ({selected_image.size[0]}x{selected_image.size[1]})")
    q_color = "#15803D" if quality["grade"] == "A" else ("#B45309" if quality["grade"] == "B" else "#B91C1C")
    st.markdown(f"**Image Quality**: <span style='color:{q_color}; font-weight:600;'>Grade {quality['grade']} ({quality['status']})</span> | Sharpness: `{quality['laplacian_sharpness_score']:.1f}`", unsafe_allow_html=True)

with col2:
    st.markdown("##### 🔬 Preprocessed Model Input")
    pre_preview = engine.preprocessor.enhance_clahe(selected_image) if apply_clahe else selected_image.resize((224, 224))
    st.image(pre_preview, use_container_width=True, caption="Normalized 224x224 (CLAHE: Active)" if apply_clahe else "Normalized 224x224 (Standard)")
    st.markdown(f"**Mean Luminance**: `{quality['mean_luminance']:.1f}` | Dynamic Range: `{quality['dynamic_range']:.1f}`")

with col3:
    st.markdown(f"##### 🔥 Grad-CAM Heatmap ({cam['target_class_display']})")
    st.image(cam["overlay_pil"], use_container_width=True, caption=f"top_conv activations (α={cam['blend_alpha']:.2f})")
    boxes = cam["focal_pathology_boxes"]
    if boxes:
        st.markdown(f"**Focal Pathology**: `{len(boxes)}` cluster(s) detected (Peak: `{boxes[0]['peak_intensity']:.2f}`)")
    else:
        st.markdown("**Focal Pathology**: Diffuse activation pattern across retina")

st.markdown("---")

# --- Middle Section: Predictions & Uncertainty ---
mid_col1, mid_col2 = st.columns([5, 4])

with mid_col1:
    st.subheader("📊 Diagnostic Class Probabilities")
    prob_df = {
        "Disease Entity": [CLASS_DISPLAY_NAMES[k] for k in res["deterministic_probabilities"].keys()],
        "Probability (%)": list(res["deterministic_probabilities"].values()),
    }
    fig = px.bar(
        prob_df,
        x="Probability (%)",
        y="Disease Entity",
        orientation="h",
        color="Probability (%)",
        color_continuous_scale="Blues",
        text="Probability (%)",
        range_x=[0, 100],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=260, margin=dict(l=10, r=20, t=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

with mid_col2:
    st.subheader("🎯 Diagnostic Confidence & Stability")
    top_pred = res["predicted_display_name"]
    top_conf = res["confidence_percentage"]

    # Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=top_conf,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Primary: {top_pred}", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#1E3A8A"},
            'steps': [
                {'range': [0, 50], 'color': '#FEE2E2'},
                {'range': [50, 80], 'color': '#FEF3C7'},
                {'range': [80, 100], 'color': '#DCFCE7'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 3},
                'thickness': 0.75,
                'value': 85.0
            }
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- Lower Section: Clinical Triage & Action Plan ---
st.markdown("---")
st.subheader("📋 Clinical Triage & Ophthalmological Referral Protocol")

t_col1, t_col2 = st.columns([5, 5])

with t_col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    badge_style = "badge-urgent" if triage["urgency_code"].startswith("URGENT") or triage["second_opinion_required"] else ("badge-routine" if "ROUTINE" in triage["urgency_code"] else "badge-normal")
    st.markdown(f"### Referral Urgency: <span class='{badge_style}'>{triage['urgency_tier']}</span>", unsafe_allow_html=True)
    st.markdown(f"**Response Target (SLA)**: `{triage['response_sla']}`")
    st.markdown(f"**ICD-10 Diagnostic Guidance**: `{triage['icd10_code']}`")
    
    opinion_badge = "🔴 **MANDATORY SECOND SPECIALIST OPINION**" if triage["second_opinion_required"] else "🟢 **Standard Protocol Clear**"
    st.markdown(f"**Second-Opinion Safeguard**: {opinion_badge}")
    st.markdown(f"**Clinical Rationale**:\n{triage['clinical_rationale']}")
    st.markdown('</div>', unsafe_allow_html=True)

with t_col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### 🩺 Frontline Action Items")
    for idx, item in enumerate(triage["action_checklist"], 1):
        st.markdown(f"**{idx}.** {item}")

    st.markdown("#### Recommended Diagnostic Modalities")
    for mod in triage["recommended_modalities"]:
        st.markdown(f"- {mod}")

    if unc:
        st.markdown("---")
        st.markdown(
            f"**Uncertainty Profile**: Entropy `{unc['shannon_entropy_bits']:.3f}` bits | "
            f"Epistemic Std `{unc['epistemic_uncertainty_std']:.3f}` | "
            f"Reliability Index `{unc['reliability_score']:.1f}%`"
        )
    st.markdown('</div>', unsafe_allow_html=True)
