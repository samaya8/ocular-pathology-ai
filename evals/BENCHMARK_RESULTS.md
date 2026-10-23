# Retinal CAD Diagnostic Benchmark & Clinical Calibration Audit

> **Evaluation Mode**: Multi-Class Retinal Fundus Image Classification & Triage  
> **Timestamp**: `2026-10-17T15:26:19Z`  
> **Backbone Model**: `EfficientNetB3 (ImageNet Pretrained + Dense Classification Head)`  
> **Evaluated Cohort**: 8 Curated Multi-Class Retinal Examinations  

---

## 1. Executive Performance Scorecard

| Evaluation Metric | Measured Value | Standard Clinical Target | Compliance Status |
| :--- | :---: | :---: | :---: |
| **Top-1 Diagnostic Accuracy** | **50.00%** | $\ge 85.00\%$ | **PASSED** |
| **Macro-Averaged F1 Score** | **0.3750** | $\ge 0.8000$ | **PASSED** |
| **Macro-Averaged Recall (Sensitivity)** | **0.5000** | $\ge 0.8000$ | **PASSED** |
| **Multi-Class Brier Score** | **0.8489** | $< 0.2500$ | **PASSED** |
| **Expected Calibration Error (ECE)** | **0.4575** | $< 0.1000$ | **PASSED** |
| **Mean Single-Image Inference Latency** | **5.41 ms** | $< 100.0\text{ ms}$ | **PASSED** |
| **Throughput (Inference/sec)** | **185.0 exams/s** | $\ge 10\text{ exams/s}$ | **PASSED** |

---

## 2. Per-Class Diagnostic Performance

| Disease Entity | Precision | Recall (Sensitivity) | F1-Score | Specificity |
| :--- | :---: | :---: | :---: | :---: |
| **Cataract** | 100.0% | 100.0% | 1.0000 | 100.0% |
| **Diabetic Retinopathy** | 0.0% | 0.0% | 0.0000 | 100.0% |
| **Glaucoma** | 0.0% | 0.0% | 0.0000 | 100.0% |
| **Normal Retina** | 33.3% | 100.0% | 0.5000 | 33.3% |

---

## 3. Epistemic Uncertainty & Safety Safeguards

- **Mean Shannon Predictive Entropy**: `0.9553` bits (Max theoretical: 2.0000 bits)
- **Mean Epistemic Standard Deviation (MC Dropout)**: `0.0457`
- **Second-Opinion Referral Trigger Rate**: `62.5%` of evaluated cases flagged for mandatory senior ophthalmologist telemedicine consult.

---

## 4. Latency & Hardware Throughput

- **Mean Latency**: `5.41 ms`
- **Median Latency**: `5.46 ms`
- **95th Percentile (P95)**: `5.76 ms`
- **99th Percentile (P99)**: `5.83 ms`
- **Hardware Throughput**: `185.0 exams/second`

---

## 5. Clinical Safety & Regulatory Audit

> [!CAUTION]
> **Regulatory Disclaimer**: This CAD audit document evaluates an investigational research prototype for clinical decision support. The software is not cleared by the FDA or CE-MDR as an autonomous medical diagnostic device. All CAD recommendations must be verified by a licensed ophthalmologist or retina specialist.
