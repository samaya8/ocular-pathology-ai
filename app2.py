"""Ophthalmology AI Research & Diagnostic Studio (Executive / Specialist Studio)."""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit as st

from src.preprocessor import CLASS_LABELS, CLASS_DISPLAY_NAMES, RetinaPreprocessor
from src.retina_engine import RetinaEngine

st.set_page_config(
    page_title="OcularMed AI | Clinical Specialist Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_DIR = Path(__file__).resolve().parent / "data" / "sample_fundus"
REPORT_PATH = Path(__file__).resolve().parent / "evals" / "BENCHMARK_RESULTS.md"


@st.cache_resource
def get_retina_engine():
    return RetinaEngine()


engine = get_retina_engine()

# Custom Styling
st.markdown(
    """
    <style>
    .studio-header { font-size: 2.1rem; font-weight: 700; color: #0F172A; margin-bottom: 0.1rem; }
    .studio-sub { font-size: 1.0rem; color: #64748B; margin-bottom: 1.0rem; }
    .kpi-box { background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; text-align: center; }
    .kpi-val { font-size: 1.7rem; font-weight: 700; color: #1E3A8A; }
    .kpi-lbl { font-size: 0.85rem; color: #64748B; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="studio-header">🔬 Ophthalmology AI Research & Diagnostic Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="studio-sub">Clinical Decision Support, Diagnostic Uncertainty Decomposition, and Explainable Pathology Localization.</div>', unsafe_allow_html=True)

# Preset options
preset_options = {
    "Cataract (Lens Opacity)": SAMPLE_DIR / "cataract.png",
    "Diabetic Retinopathy (Exudates)": SAMPLE_DIR / "diabetic_retinopathy.png",
    "Glaucoma (Optic Cupping)": SAMPLE_DIR / "glaucoma.png",
    "Normal Retina (Healthy Control)": SAMPLE_DIR / "normal.png",
}

# Sidebar Global Settings
st.sidebar.header("📁 Active Examination")
exam_source = st.sidebar.radio("Input Examination", ["Standard Research Cases", "Upload New Image"])
active_img: Image.Image | None = None
active_name = "Patient Exam"

if exam_source == "Standard Research Cases":
    chosen = st.sidebar.selectbox("Select Patient Case", list(preset_options.keys()))
    p = preset_options[chosen]
    if p.exists():
        active_img = Image.open(p).convert("RGB")
        active_name = chosen
else:
    up = st.sidebar.file_uploader("Upload Retinal Scan", type=["png", "jpg", "jpeg"])
    if up is not None:
        active_img = Image.open(up).convert("RGB")
        active_name = up.name

st.sidebar.markdown("---")
st.sidebar.subheader("Clinical Studio Controls")
global_clahe = st.sidebar.checkbox("Enable CLAHE Normalization", value=False)
global_mc_passes = st.sidebar.slider("MC Dropout Sample Size", min_value=5, max_value=40, value=15, step=5)
cam_cmap = st.sidebar.selectbox("Grad-CAM Colormap", ["jet", "turbo", "viridis"])
cam_alpha = st.sidebar.slider("Heatmap Blending (α)", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

if active_img is None:
    st.warning("Please choose a standard research case or upload a fundus scan from the sidebar.")
    st.stop()

# Run Engine Analysis
with st.spinner("Executing deep retinal inference pipeline..."):
    res = engine.analyze_image(
        active_img,
        run_mc_dropout=True,
        run_gradcam=True,
        apply_clahe=global_clahe,
        num_mc_samples=global_mc_passes,
    )
    if cam_alpha != res["gradcam"]["blend_alpha"] or cam_cmap != res["gradcam"].get("colormap"):
        from src.gradcam_explainer import GradCAMExplainer
        res["gradcam"] = GradCAMExplainer(colormap=cam_cmap, alpha=cam_alpha).compute_gradcam(
            engine.model,
            engine.preprocessor.preprocess_image(active_img, apply_clahe=global_clahe),
            active_img,
            target_class=res["predicted_class"],
        )

# 5-Tab Navigation Layout
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏥 Clinical Screening Console",
    "📈 Uncertainty & MC Dropout Lab",
    "🔥 Multi-Class Explainability Arena",
    "📋 Cohort Screening & Batch Triage",
    "🛡️ Model Governance & Safety",
])

# ==================== TAB 1: Clinical Screening Console ====================
with tab1:
    st.subheader(f"Patient Exam: {active_name}")

    # Top KPI Metrics
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-box"><div class="kpi-val">{res["predicted_display_name"]}</div><div class="kpi-lbl">Top CAD Diagnosis</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-box"><div class="kpi-val">{res["confidence_percentage"]:.1f}%</div><div class="kpi-lbl">Diagnostic Confidence</div></div>', unsafe_allow_html=True)
    with k3:
        sla_val = res["clinical_triage"]["response_sla"]
        st.markdown(f'<div class="kpi-box"><div class="kpi-val">{sla_val}</div><div class="kpi-lbl">Target Referral SLA</div></div>', unsafe_allow_html=True)
    with k4:
        ent_val = res["uncertainty"]["shannon_entropy_bits"] if res["uncertainty"] else 0.0
        st.markdown(f'<div class="kpi-box"><div class="kpi-val">{ent_val:.3f}</div><div class="kpi-lbl">Predictive Entropy (bits)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3-Way Image Inspector
    im_c1, im_c2, im_c3 = st.columns(3)
    with im_c1:
        st.markdown("##### 📸 Raw Fundus Photograph")
        st.image(active_img, use_container_width=True, caption=f"Dimensions: {active_img.size[0]}x{active_img.size[1]}")
    with im_c2:
        st.markdown("##### 🔬 Preprocessed Model Input")
        preview_pre = engine.preprocessor.enhance_clahe(active_img) if global_clahe else active_img.resize((224, 224))
        st.image(preview_pre, use_container_width=True, caption="Normalized to [0, 1] (224x224)")
    with im_c3:
        st.markdown("##### 🔥 Grad-CAM Pathology Overlay")
        st.image(res["gradcam"]["overlay_pil"], use_container_width=True, caption=f"Class: {res['gradcam']['target_class_display']} (α={cam_alpha:.2f})")

    st.markdown("---")

    # Diagnostic Probabilities and Triage Protocol
    diag_c1, diag_c2 = st.columns([5, 5])
    with diag_c1:
        st.markdown("#### Multi-Class Distribution")
        probs_df = pd.DataFrame([
            {"Disease": CLASS_DISPLAY_NAMES[k], "Confidence (%)": v}
            for k, v in res["deterministic_probabilities"].items()
        ])
        fig_bar = px.bar(
            probs_df,
            x="Confidence (%)",
            y="Disease",
            orientation="h",
            color="Confidence (%)",
            color_continuous_scale="Teal",
            text="Confidence (%)",
            range_x=[0, 100],
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(height=260, margin=dict(l=10, r=20, t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with diag_c2:
        st.markdown("#### Clinical Action Protocol")
        t = res["clinical_triage"]
        st.info(f"**Referral Tier**: {t['urgency_tier']}\n\n**Response SLA Target**: {t['response_sla']}\n\n**ICD-10 Code**: `{t['icd10_code']}`")
        if t["second_opinion_required"]:
            st.error("⚠️ **Specialist Safeguard Active**: Elevated predictive entropy requires mandatory senior ophthalmologist sign-off.")
        else:
            st.success("✅ **Primary Care Protocol**: Standard risk profile verified.")

# ==================== TAB 2: Uncertainty & MC Dropout Lab ====================
with tab2:
    st.subheader("📈 Monte Carlo Dropout Uncertainty & Entropy Lab")
    st.markdown(
        "Standard deep neural networks tend to yield overconfident point estimates. Monte Carlo Dropout "
        "samples the network with active dropout layers during inference, approximating the posterior distribution over network weights."
    )

    unc = res["uncertainty"]
    if unc:
        u1, u2, u3 = st.columns(3)
        with u1:
            st.metric("Total MC Passes", f"{unc['num_mc_samples']} stochastic runs")
            st.metric("Shannon Predictive Entropy", f"{unc['shannon_entropy_bits']:.4f} bits", help="Measures total predictive uncertainty (max 2.0 bits for 4 classes)")
        with u2:
            st.metric("Epistemic Std Dev (σ)", f"{unc['epistemic_uncertainty_std']:.4f}", help="Model weight uncertainty across stochastic passes")
            st.metric("Mutual Information", f"{unc['mutual_information']:.4f}", help="Information gained about model weights from observation")
        with u3:
            st.metric("Diagnostic Reliability", f"{unc['reliability_score']:.1f}%")
            st.metric("Uncertainty Tier", unc["uncertainty_tier"])

        st.markdown("---")
        st.markdown("#### Stochastic Sample Variance per Disease Class")
        mc_mat = np.array(unc["mc_samples_matrix"])  # (N, 4)
        mc_df_list = []
        for i in range(mc_mat.shape[0]):
            for c_idx in range(4):
                c_lbl = CLASS_LABELS[c_idx]
                mc_df_list.append({
                    "Sample Pass": i + 1,
                    "Disease Entity": CLASS_DISPLAY_NAMES[c_lbl],
                    "Probability (%)": mc_mat[i, c_idx] * 100.0,
                })
        mc_df = pd.DataFrame(mc_df_list)

        fig_box = px.box(
            mc_df,
            x="Disease Entity",
            y="Probability (%)",
            color="Disease Entity",
            points="all",
            title=f"Posterior Probability Spread across {unc['num_mc_samples']} MC Dropout Passes",
        )
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

# ==================== TAB 3: Multi-Class Explainability Arena ====================
with tab3:
    st.subheader("🔥 Multi-Class Grad-CAM Comparison Arena")
    st.markdown(
        "Observe how the deep convolutional feature maps in `top_conv` activate differently when queried "
        "for different clinical hypotheses on the **exact same fundus photograph**."
    )

    from src.gradcam_explainer import GradCAMExplainer
    explainer = GradCAMExplainer(colormap=cam_cmap, alpha=cam_alpha)
    pre_tensor = engine.preprocessor.preprocess_image(active_img, apply_clahe=global_clahe)

    arena_cols = st.columns(4)
    for idx, (lbl, disp) in enumerate(CLASS_DISPLAY_NAMES.items()):
        with arena_cols[idx]:
            cam_eval = explainer.compute_gradcam(
                engine.model,
                pre_tensor,
                active_img,
                target_class=lbl,
                alpha=cam_alpha,
            )
            prob_val = res["deterministic_probabilities"].get(lbl, 0.0)
            is_top = lbl == res["predicted_class"]
            header_prefix = "⭐ " if is_top else ""
            st.markdown(f"##### {header_prefix}{disp}")
            st.image(cam_eval["overlay_pil"], use_container_width=True, caption=f"P({disp}) = {prob_val:.1f}%")
            boxes = cam_eval["focal_pathology_boxes"]
            if boxes:
                st.caption(f"Peak Intensity: `{boxes[0]['peak_intensity']:.2f}` | Center: `{boxes[0]['center_point']}`")
            else:
                st.caption("Diffuse activation across retina")

# ==================== TAB 4: Cohort Screening & Batch Triage ====================
with tab4:
    st.subheader("📋 Cohort Screening & Batch Triage")
    st.markdown(
        "Upload a batch of fundus images or triage the standard patient cohort. Cases are sorted by clinical urgency, "
        "ensuring high-risk sight-threatening emergencies are reviewed first."
    )

    if st.button("🚀 Execute Cohort Screening Evaluation"):
        cohort_files = list(SAMPLE_DIR.glob("*.png"))
        with st.spinner(f"Evaluating {len(cohort_files)} examinations..."):
            batch_out = engine.analyze_batch(cohort_files, run_mc_dropout=True, run_gradcam=False)
            b_df = batch_out["triage_summary_table"]

            st.markdown(f"**Total Exams**: `{batch_out['total_exams_evaluated']}` | **Urgent / Flagged**: `{batch_out['urgent_cases_count']}`")

            st.dataframe(
                b_df.style.applymap(
                    lambda v: "background-color: #FEE2E2; color: #991B1B; font-weight: bold;" if v is True else "",
                    subset=["Second_Opinion_Required"],
                ),
                use_container_width=True,
            )

            # Export to CSV
            csv_data = b_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Triage Ledger (CSV)",
                data=csv_data,
                file_name="retinal_cohort_triage_ledger.csv",
                mime="text/csv",
            )

# ==================== TAB 5: Model Governance & Safety ====================
with tab5:
    st.subheader("🛡️ Model Governance, Architecture & Clinical Compliance")

    gov_c1, gov_c2 = st.columns([5, 5])
    with gov_c1:
        st.markdown(
            """
            ### Model Metadata & Architecture
            - **Base Backbone**: `EfficientNetB3 (Tan & Le, 2019)`
            - **Input Dimensions**: `224 x 224 x 3` (RGB normalized to `[0.0, 1.0]`)
            - **Pretrained Weights**: ImageNet initialization
            - **Regularization**: BatchNormalization + Dropout (rate = 0.45)
            - **Target Classes**: `[Cataract, Diabetic Retinopathy, Glaucoma, Normal]`
            - **Inference Latency**: Sub-100ms CPU inference per fundus examination
            """
        )

    with gov_c2:
        st.markdown(
            """
            ### Clinical Safety Principles
            1. **Second-Opinion Trigger**: Automatically escalates ambiguous presentations with high Shannon entropy (>1.10 bits) or optical blur.
            2. **Visual Explainability**: Every diagnostic prediction can be paired with an anatomical Grad-CAM heatmap to prevent black-box reliance.
            3. **Optical Quality Screening**: Pre-filters blurry, underexposed, or low-resolution images before clinical review.
            4. **Regulatory Classification**: Designed under FDA Software as a Medical Device (SaMD) Class II decision-support guidelines.
            """
        )

    st.markdown("---")
    st.subheader("Empirical Benchmark Audit Report (`BENCHMARK_RESULTS.md`)")
    if REPORT_PATH.exists():
        st.markdown(REPORT_PATH.read_text(encoding="utf-8"))
    else:
        st.info("Benchmark report not yet generated. Run `cli.py benchmark` to produce empirical results.")
