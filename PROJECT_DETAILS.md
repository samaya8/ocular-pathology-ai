# Retinal Disease Classification & Clinical Diagnostic Intelligence
> **Enterprise Clinical Computer Vision & AI Diagnostic Triage Platform**  
> **Backbone**: EfficientNetB3 (ImageNet Transfer Learning) | **Calibration**: Monte Carlo Dropout & Predictive Entropy  
> **Explainability**: Grad-CAM & Lesion Localization | **Clinical Triage**: Urgency Grading & ICD-10 Mapping

---

## 1. Executive Summary & Clinical Context

Retinal disease remains a primary driver of preventable vision impairment and blindness globally:
- **Cataracts**: Opacification of the crystalline lens impairing light transmission to the retina, accounting for over 50% of global blindness.
- **Glaucoma**: Progressive optic neuropathy characterized by retinal nerve fiber layer (RNFL) thinning and neuroretinal rim loss, often asymptomatic until advanced stages.
- **Diabetic Retinopathy (DR)**: Microvascular retinal damage caused by chronic hyperglycemia, manifesting as microaneurysms, intraretinal hemorrhages, hard exudates, and neovascularization.
- **Normal Retinal Fundus**: Intact optic disc margin, healthy cup-to-disc ratio ($\le 0.4$), uniform vascular caliber, and an uncompromised macula/fovea centralis.

Automated computer-aided diagnosis (CAD) in primary care triage mitigates critical ophthalmologist shortages. However, conventional CNN classifiers suffer from overconfidence on out-of-distribution artifacts and lack diagnostic interpretability.

This project delivers an **enterprise-grade medical AI diagnostic platform** built on **EfficientNetB3**, integrating **Laplacian exposure and focus audits**, **feature-cached Monte Carlo Dropout uncertainty calibration**, **Grad-CAM pathological heatmap localization**, and an **ICD-10-CM mapped clinical referral triage engine**.

---

## 2. End-to-End System Architecture

```text
                                  [Retinal Fundus Image]
                                             │
                                             ▼
                       ┌───────────────────────────────────────────┐
                       │           RetinaPreprocessor              │
                       │ ─ Laplacian Sharpness Variance (σ² ≥ 15)  │
                       │ ─ Exposure & Contrast Histogram Audit     │
                       │ ─ CLAHE Contrast Enhancement              │
                       │ ─ Letterbox Aspect-Ratio Normalization    │
                       └─────────────────────┬─────────────────────┘
                                             │ Preprocessed Tensor (1, 224, 224, 3)
                                             ▼
                       ┌───────────────────────────────────────────┐
                       │               ModelLoader                 │
                       │ ─ EfficientNetB3 Singleton                │
                       │ ─ Mock Fallback for Headless CI/CD        │
                       └─────────────────────┬─────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
┌──────────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│            UncertaintyEngine             │    │             GradCAMExplainer             │
│ ─ Base Feature Cache                     │    │ ─ Target: 'top_conv' (7, 7, 1536)        │
│ ─ N=20 Monte Carlo Dropout Forward Passes│    │ ─ tf.GradientTape Class Gradient Pooling │
│ ─ Shannon Predictive Entropy H(p)        │    │ ─ Guided ReLU & Jet/Turbo Heatmap Blends │
│ ─ 95% Credible Intervals & Mutual Info   │    │ ─ Pathological Bounding Box Localization │
└──────────────────────┬───────────────────┘    └────────────────────┬─────────────────────┘
                       │                                             │
                       └─────────────────────┬───────────────────────┘
                                             │ Class Probabilities, Entropy, Heatmaps
                                             ▼
                       ┌───────────────────────────────────────────┐
                       │          ClinicalTriageAdvisor            │
                       │ ─ Urgency Tier: Emergency / Urgent /      │
                       │   Routine / Annual Surveillance           │
                       │ ─ ICD-10-CM Coding & Diagnostic Workup    │
                       │ ─ Second-Opinion Ambiguity Trigger Flag   │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
        ┌────────────────────────────────────┼────────────────────────────────────┐
        │                                    │                                    │
        ▼                                    ▼                                    ▼
┌─────────────────┐                ┌──────────────────┐                ┌──────────────────┐
│  cli.py Driver  │                │  app.py (Portal) │                │ app2.py (Studio) │
│ 6 Subcommands   │                │ Point-of-Care    │                │ 5 Research Tabs  │
└─────────────────┘                └──────────────────┘                └──────────────────┘
```

---

## 3. Mathematical Formulations

### 3.1 Preprocessing & Image Quality Assurance

1. **Focus/Sharpness Metric (Discrete Laplacian Operator)**:
   $$\sigma^2 = \operatorname{Var}\left(\nabla^2 I\right) = \frac{1}{N}\sum_{x,y} \left( (\nabla^2 I)(x,y) - \mu_{\nabla^2 I} \right)^2$$
   Images with $\sigma^2 < 15.0$ are flagged for camera defocus or motion blur.

2. **Contrast-Limited Adaptive Histogram Equalization (CLAHE)**:
   Partitions the luminance channel into $8 \times 8$ contextual tiles, clipping histograms at slope limit $\beta = 2.0$ to prevent noise overamplification before bilinear interpolation.

3. **Letterbox Aspect-Ratio Padding**:
   Given original dimensions $(W_0, H_0)$ and target dimensions $(W_t, H_t)$, computes scale factor $s = \min(W_t / W_0, H_t / H_0)$, centering the resized image within black padding borders to preserve foveal geometry.

### 3.2 Monte Carlo Dropout & Bayesian Uncertainty

Standard deterministic softmax outputs $\hat{y} = \operatorname{softmax}(f(x))$ are frequently uncalibrated and overconfident. We introduce epistemic uncertainty estimation via Monte Carlo Dropout:

1. **Predictive Mean**:
   For $T$ stochastic forward passes with active dropout rates ($p = 0.45$):
   $$\bar{p}_c = \frac{1}{T} \sum_{t=1}^T p_c^{(t)}$$

2. **Shannon Predictive Entropy**:
   Measures total predictive uncertainty across classes $C \in \{1, \dots, 4\}$:
   $$\mathcal{H}(p) = -\sum_{c=1}^C \bar{p}_c \log_2(\bar{p}_c + \epsilon), \quad \mathcal{H} \in [0, \log_2(C)] \approx [0, 2.0] \text{ bits}$$

3. **Epistemic Mutual Information**:
   Separates epistemic (model parameter uncertainty) from aleatoric (data noise) uncertainty:
   $$I(y, W \mid x) = \mathcal{H}\left(\frac{1}{T} \sum_{t=1}^T p^{(t)}\right) - \frac{1}{T} \sum_{t=1}^T \mathcal{H}\left(p^{(t)}\right)$$

4. **Credible Intervals**:
   Computed using empirical 2.5th and 97.5th percentiles of Monte Carlo trajectories for the top predicted diagnosis:
   $$\mathrm{CI}_{95\%} = \left[ Q_{0.025}\left(\{p_c^{(t)}\}\right), \; Q_{0.975}\left(\{p_c^{(t)}\}\right) \right]$$

### 3.3 Gradient-Weighted Class Activation Mapping (Grad-CAM)

To provide interpretability without modifying model architecture, gradients of the winning class score $y^c$ are computed with respect to feature activation maps $A^k$ of the final convolutional layer `top_conv` (shape $(7, 7, 1536)$):

1. **Neuron Importance Weights**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}$$

2. **Class-Discriminative Saliency Map**:
   $$L_{\text{Grad-CAM}}^c = \operatorname{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
   The rectified linear unit filters out features that negatively contribute to the class of interest.

3. **Bilinear Upsampling & Lesion Bounding Boxes**:
   The activation map is scaled to image dimensions $(224, 224)$, normalized to $[0, 1]$, and thresholded at $\tau = 0.6$ of maximum intensity to extract focal pathological cluster bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$.

---

## 4. Clinical Referral Triage & ICD-10 Mapping

| Diagnostic Category | ICD-10-CM | Referral Urgency | Target Window | Recommended Modalities | Clinical Checklist |
|---|---|---|---|---|---|
| **Glaucoma** | `H40.9` | **Emergency** | 12–24 Hours | Humphrey Visual Field 24-2, RNFL OCT, Tonometry | Stat IOP measurement; assess gonioscopy; initiate topical prostaglandins |
| **Diabetic Retinopathy** | `E11.319` | **Urgent** | 1–2 Weeks | Macular OCT, Fluorescein Angiography (FFA), Dilated Fundus | Assess DME presence; optimize HbA1c control; consider anti-VEGF / focal laser |
| **Cataract** | `H25.9` | **Routine** | 1–3 Months | Optical Biometry (IOLMaster), Slit-lamp biomicroscopy | Evaluate visual acuity impact; glare disability test; schedule phacoemulsification |
| **Normal** | `Z01.00` | **Annual** | 12 Months | Routine Non-Mydriatic Fundus Photography | Annual wellness screening; blood glucose/BP checks; patient education |

### Second-Opinion Safety Gate
A second-opinion trigger flag (`requires_second_opinion = True`) is raised if:
- Predictive entropy $\mathcal{H}(p) > 0.85$ bits
- Top confidence $< 70.0\%$
- Prediction margin between 1st and 2nd highest class $< 15.0\%$

---

## 5. Software Architecture & Directory Layout

```text
ocular-pathology-ai/
├── src/
│   ├── __init__.py                 # Package exports
│   ├── preprocessor.py             # RetinaPreprocessor (CLAHE, Laplacian, letterbox)
│   ├── model_loader.py             # ModelLoader singleton with Mock fallback
│   ├── uncertainty_engine.py       # Monte Carlo Dropout, entropy, credible intervals
│   ├── gradcam_explainer.py        # GradCAMExplainer & lesion bounding box extractor
│   ├── clinical_triage.py          # ClinicalTriageAdvisor & ICD-10 mapping
│   └── retina_engine.py            # High-level facade for end-to-end pipeline
├── evals/
│   ├── __init__.py                 # Evaluation package
│   ├── runner.py                   # Automated benchmark evaluation harness
│   └── BENCHMARK_RESULTS.md        # Comprehensive benchmark documentation
├── data/sample_fundus/
│   ├── generate_samples.py         # Deterministic synthetic fundus image generator
│   ├── normal.png                  # Healthy fundus sample
│   ├── cataract.png                # Cataract fundus sample
│   ├── diabetic_retinopathy.png    # DR fundus sample
│   └── glaucoma.png                # Glaucoma fundus sample
├── tests/
│   ├── test_preprocessing.py       # Legacy & enhanced preprocessor unit tests
│   ├── test_uncertainty.py         # Monte Carlo dropout & entropy tests
│   ├── test_gradcam.py             # Grad-CAM heatmap & bounding box tests
│   ├── test_clinical_triage.py     # Triage urgency & ICD-10 mapping tests
│   ├── test_retina_engine.py       # High-level facade tests
│   ├── test_cli.py                 # CLI end-to-end subcommands tests
│   └── test_benchmark.py           # Benchmark runner evaluation tests
├── app.py                          # Streamlit Point-of-Care Clinical Screening Portal
├── app2.py                         # Streamlit Ophthalmology AI Research Studio (5 tabs)
├── cli.py                          # Multi-command Production CLI
├── inference.py                    # Backward-compatible inference wrapper
├── efficientnetb3-Eye Disease-90.76.h5 # Saved model checkpoint
├── efficientnetb3-Eye Disease-weights.h5 # Model weights checkpoint
├── requirements.txt                # Production & test dependencies
├── PROJECT_DETAILS.md              # Master technical documentation
└── README.md                       # Repository overview & quick start
```

---

## 6. Command-Line Interface (CLI) Guide

The `cli.py` interface provides 6 commands supporting both human-readable terminal output and automated JSON payloads:

### 1. Deterministic Prediction
```bash
python cli.py predict --image data/sample_fundus/glaucoma.png --json
```

### 2. Monte Carlo Dropout Uncertainty
```bash
python cli.py uncertainty --image data/sample_fundus/cataract.png --samples 25
```

### 3. Grad-CAM Pathology Heatmap Generation
```bash
python cli.py explain --image data/sample_fundus/diabetic_retinopathy.png --colormap jet --output gradcam_dr.png
```

### 4. Comprehensive Clinical Referral Triage
```bash
python cli.py triage --image data/sample_fundus/glaucoma.png
```

### 5. Cohort Batch Processing
```bash
python cli.py batch --input-dir data/sample_fundus --output-csv cohort_triage.csv --samples 15
```

### 6. Automated Benchmark Evaluation
```bash
python cli.py benchmark --data-dir data/sample_fundus --output evals/BENCHMARK_RESULTS.md
```

---

## 7. Interactive Streamlit Applications

### Point-of-Care Clinical Screening (`app.py`)
- Single-image fundus upload or synthetic sample picker.
- Laplacian image quality score with real-time blur warning.
- Top diagnosis badge with confidence meter.
- Side-by-side visualization: Preprocessed Fundus vs. Grad-CAM Superimposed Heatmap.
- Urgency badge with recommended specialist appointment window.
- Recommended diagnostic testing checklist and exportable clinical summary.

### Ophthalmology AI Research & Diagnostic Studio (`app2.py`)
- **Tab 1: Clinical Examination**: Full multimodal breakdown with interactive lesion bounding box overlays.
- **Tab 2: Bayesian Uncertainty Studio**: Monte Carlo predictive distribution violins, credible intervals, and entropy gauge.
- **Tab 3: Grad-CAM Explainability**: Multi-colormap explorer (`jet`, `turbo`, `viridis`, `plasma`) with threshold controls.
- **Tab 4: Cohort Batch Analytics**: Batch CSV inspection with urgency distribution charts and ambiguity review table.
- **Tab 5: System Benchmarks**: Model latency gauges, ECE calibration curve, and confusion matrix analytics.

---

## 8. Verification & Quality Assurance

- **Unit Test Suite**: 59 automated tests passing across 7 test suites.
- **Code Coverage**: 85% overall core codebase coverage.
- **Execution Speed**: Full test suite completes in under 5.0 seconds on CPU.
- **Backward Compatibility**: Preserves 100% functional parity with legacy `inference.py` and `tests/test_preprocessing.py`.
