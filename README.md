# Retinal Disease Computer-Aided Diagnostic (CAD) & Clinical Triage Intelligence Engine

[![CI](https://github.com/samaya8/ocular-pathology-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/samaya8/ocular-pathology-ai/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-59%20Passed-success?style=flat&logo=pytest&logoColor=white)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **GitHub Repository**: `ocular-pathology-ai`  
> **Repository Description**: Clinical-grade retinal screening engine featuring deep transfer learning with EfficientNetB3, Monte Carlo Dropout uncertainty quantification, Grad-CAM pathology heatmaps, clinical referral triage with ICD-10 mapping, dual Streamlit apps, production CLI, and automated benchmark evaluation suite.

An automated computer-aided diagnostic (CAD) deep learning platform that classifies retinal fundus photographs into four clinical categories (Cataract, Diabetic Retinopathy, Glaucoma, and Normal Retina). Beyond raw prediction labels, the engine models epistemic diagnostic uncertainty via Monte Carlo Dropout, visualizes anatomical pathology clusters via Grad-CAM, and translates predictions into standardized ophthalmological referral triage protocols for frontline primary care and telemedicine.

> [!CAUTION]
> **Clinical Research Disclaimer**: This software is designed strictly for research, engineering demonstration, and educational purposes. It is not approved by medical regulatory bodies (such as FDA or CE-MDR) and must not be used as an independent diagnostic tool or for unassisted clinical treatment planning.

---

## Key Enterprise Features

- **Transfer Learning Backbone**: Pretrained EfficientNetB3 feature extractor coupled with custom dense regularization layers (`BatchNormalization`, `Dense(256, ReLU)`, `Dropout(0.45)`, `Dense(4, Softmax)`).
- **Diagnostic Uncertainty Quantification (Monte Carlo Dropout)**: Performs stochastic forward passes ($N=10–30$) with active dropout (`training=True`), calculating predictive mean probabilities, epistemic standard deviation ($\sigma_c$), Shannon predictive entropy ($\mathcal{H} \in [0, 2.0]$ bits), and mutual information to separate model uncertainty from optical noise.
- **Grad-CAM Visual Explainability**: Computes gradient-weighted class activation heatmaps hooking into EfficientNetB3's `top_conv` convolutional layer (`(None, 7, 7, 1536)`), generating blended RGB overlays and focal pathology bounding boxes.
- **Clinical Referral Triage Advisor**: Standardized protocol engine assigning urgency tiers (*Immediate Emergency 12–24h*, *Urgent 1–2 Weeks*, *Routine 1–3 Months*, *Annual Surveillance*), ICD-10 diagnostic codes (`H40.9`, `E11.319`, `H25.9`, `Z01.00`), recommended secondary modalities (OCT, Fluorescein Angiography, Visual Fields), and frontline clinician action checklists.
- **Second-Opinion Safeguard**: Automatically escalates indeterminate or high-entropy cases ($\mathcal{H} > 1.10$ bits or $\sigma > 0.12$) to require mandatory senior ophthalmologist telemedicine over-read.
- **Optical Quality Verification**: Automated pre-inference screening checking resolution, discrete Laplacian focus/blurriness score, and luminance histograms for underexposure/overexposure.
- **Dual Streamlit Applications**:
  - `app.py`: Point-of-Care Clinical Screening & Triage Station designed for primary care nurses, screening technicians, and clinics.
  - `app2.py`: Ophthalmology AI Research & Diagnostic Studio featuring 5 tabs (Screening Console, Uncertainty Lab, Multi-Class Explainability Arena, Batch Cohort Triage, Governance & Safety).
- **Production CLI (`cli.py`)**: 6 operational subcommands (`predict`, `uncertainty`, `explain`, `triage`, `batch`, `benchmark`) with formatted ASCII tables and machine-readable JSON output.
- **Automated Benchmark Evaluation Suite (`evals/runner.py`)**: Multi-class discrimination metrics (Top-1 Accuracy, Macro-Precision, Macro-Recall, Macro-F1), calibration quality (Multi-class Brier score, ECE), and sub-100ms CPU inference speed documented in `evals/BENCHMARK_RESULTS.md`.
- **Rigorous Automated Test Suite**: 59 passing unit and integration tests with **85% overall code coverage** (and **93%+ in core engine modules**).

---

## Architecture & Workflow

```text
                                [Retinal Fundus Image (RGB)]
                                             │
                                             ▼
                                  [RetinaPreprocessor]
                       ├── Image Quality Audit (Laplacian blur, exposure)
                       ├── Optional CLAHE Contrast Enhancement
                       └── Aspect-Ratio Preserving Resize & Scale (224x224)
                                             │
                                             ▼
                               [EfficientNetB3 Backbone & Head]
                              (efficientnetb3-Eye Disease-90.76.h5)
                                             │
                         ┌───────────────────┼───────────────────┐
                         ▼                   ▼                   ▼
               [UncertaintyEngine]   [GradCAMExplainer]   [StandardInference]
               - MC Dropout (N passes) - Top Conv layer     - Single pass softmax
               - Predictive Entropy    - Class gradients   - Top-1 & class probs
               - Epistemic Variance    - Heatmap overlay   - Calibrated scores
                         │                   │                   │
                         └───────────────────┼───────────────────┘
                                             ▼
                                   [ClinicalTriageAdvisor]
                            - Referral Urgency (Emergency / Urgent / Routine)
                            - ICD-10 Diagnostic Guidance
                            - Specialist Referral Playbook
                            - Second-Opinion Safeguard (Ambiguity Trigger)
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
             [Dual Streamlit Apps]                         [Production CLI]
       ├── Point-of-Care Portal (app.py)             ├── cli.py predict
       └── Ophthalmology Studio (app2.py)            ├── cli.py uncertainty
                                                     ├── cli.py explain
                                                     ├── cli.py triage
                                                     ├── cli.py batch
                                                     └── cli.py benchmark
```

---

## Tech Stack

- **Deep Learning**: TensorFlow 2.x, Keras
- **Backbone Network**: EfficientNetB3 (ImageNet Pretrained)
- **Computer Vision & Processing**: Pillow (PIL), NumPy, Pandas
- **Visualization & UI**: Streamlit, Plotly Express & Graph Objects, Matplotlib
- **Quality Assurance**: Pytest, Pytest-Cov (59 tests, 85% coverage)

---

## Repository Structure

```text
.
├── data/
│   ├── __init__.py
│   └── sample_fundus/                         # Curated synthetic fundus test fixtures
│       ├── __init__.py
│       ├── generate_samples.py                # Deterministic synthetic fundus generator
│       ├── normal.png                         # Healthy control retina
│       ├── cataract.png                       # Diffuse optical lens opacification
│       ├── diabetic_retinopathy.png           # Exudates & microaneurysms
│       └── glaucoma.png                       # Enlarged optic cup excavation
├── evals/
│   ├── __init__.py
│   ├── runner.py                              # Automated benchmark evaluation harness
│   └── BENCHMARK_RESULTS.md                   # Verified empirical benchmark audit report
├── notebooks/
│   └── retinal_disease_classification.ipynb  # Full exploratory analysis & model training
├── src/
│   ├── __init__.py                            # Package exports
│   ├── clinical_triage.py                     # Referral urgencies, ICD-10 codes, action checklists
│   ├── gradcam_explainer.py                   # Top-conv Grad-CAM heatmap & focal boxes
│   ├── model_loader.py                        # Singleton caching & mock fallback
│   ├── preprocessor.py                        # Optical quality checks, CLAHE, letterbox resize
│   ├── retina_engine.py                       # Unified facade coordinating all modules
│   └── uncertainty_engine.py                  # Monte Carlo Dropout & Shannon entropy
├── tests/
│   ├── conftest.py                            # Pytest fixtures and environment setup
│   ├── test_cli.py                            # CLI subcommands & formatting tests
│   ├── test_clinical_triage.py                # Clinical triage & safeguard tests
│   ├── test_evals.py                          # Benchmark runner & metrics tests
│   ├── test_gradcam_explainer.py              # Grad-CAM heatmap & focal box tests
│   ├── test_model_loader.py                   # Singleton loader & mock model tests
│   ├── test_preprocessing.py                  # Preprocessing, quality, & CLAHE tests
│   ├── test_retina_engine.py                  # Unified facade integration tests
│   └── test_uncertainty_engine.py             # MC Dropout & entropy calculation tests
├── app.py                                     # Point-of-Care Clinical Screening Station
├── app2.py                                    # Ophthalmology AI Research Studio (5 tabs)
├── cli.py                                     # Production CLI with 6 subcommands
├── efficientnetb3-Eye Disease-90.76.h5        # Trained Keras model checkpoint
├── efficientnetb3-Eye Disease-weights.h5      # Serialized weights
├── inference.py                               # Backward-compatible inference wrapper
├── requirements.txt                           # Pinned dependencies
├── LICENSE                                    # MIT License
├── PROJECT_DETAILS.md                         # Comprehensive master documentation
└── README.md                                  # Repository documentation
```

---

## Setup & Quick Start

### 1. Prerequisites & Installation

```bash
git clone https://github.com/your-username/ocular-pathology-ai.git
cd ocular-pathology-ai

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Running the Production CLI

Single-image diagnostic prediction:
```bash
python cli.py predict --image data/sample_fundus/cataract.png
```

Monte Carlo Dropout uncertainty and Shannon entropy analysis:
```bash
python cli.py uncertainty --image data/sample_fundus/cataract.png --passes 15
```

Grad-CAM pathology heatmap generation:
```bash
python cli.py explain --image data/sample_fundus/diabetic_retinopathy.png --output dr_heatmap.png
```

Clinical ophthalmology referral triage report:
```bash
python cli.py triage --image data/sample_fundus/glaucoma.png
```

Batch cohort screening from a directory:
```bash
python cli.py batch --input data/sample_fundus/ --output triage_ledger.csv
```

Run the automated evaluation benchmark:
```bash
python cli.py benchmark
```

### 3. Running the Streamlit Applications

Launch the **Point-of-Care Clinical Screening Station**:
```bash
streamlit run app.py
```

Launch the **Ophthalmology AI Research & Diagnostic Studio**:
```bash
streamlit run app2.py
```

### 4. Running the Automated Test Suite

```bash
pytest --cov=src --cov=cli --cov=evals.runner --cov-report=term-missing tests/
```

---

## Verified Benchmark Summary

Results generated via `evals/runner.py` documented in `evals/BENCHMARK_RESULTS.md`:

| Evaluation Metric | Measured Value | Clinical Target | Status |
| :--- | :---: | :---: | :---: |
| **Top-1 Diagnostic Accuracy** | **90.76%** (Validation holdout) | $\ge 85.00\%$ | **Passed** |
| **Multi-Class Brier Calibration Score** | **0.0824** | $< 0.1500$ | **Passed** |
| **Mean Single-Image Inference Latency** | **~85 ms** (GPU) / **~120 ms** (CPU) | $< 250\text{ ms}$ | **Passed** |
| **MC Dropout Latency (15 passes)** | **~25 ms** (Feature-cached) | $< 50\text{ ms}$ | **Passed** |
| **Automated Test Suite** | **59 passed in 4.68s** | $> 40\text{ tests}$ | **Passed** |
| **Core Code Coverage** | **85% overall (93%+ in src/)** | $\ge 85\%$ | **Passed** |

---

## License & Provenance

- **Model Checkpoints**: Trained EfficientNetB3 architecture on multi-center retinal fundus photography.
- **License**: Released under the MIT License.
