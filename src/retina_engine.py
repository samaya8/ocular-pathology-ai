"""Unified facade coordinating preprocessing, inference, uncertainty, Grad-CAM, and clinical triage."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image

from src.preprocessor import RetinaPreprocessor, CLASS_LABELS, CLASS_DISPLAY_NAMES
from src.model_loader import ModelLoader
from src.uncertainty_engine import UncertaintyEngine
from src.gradcam_explainer import GradCAMExplainer
from src.clinical_triage import ClinicalTriageAdvisor


class RetinaEngine:
    """High-level orchestration engine for Retinal CAD screening and diagnostic triage."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        use_mock: bool = False,
        mc_dropout_samples: int = 15,
        gradcam_colormap: str = "jet",
    ) -> None:
        self.model = ModelLoader.get_model(model_path=model_path, use_mock=use_mock)
        self.preprocessor = RetinaPreprocessor()
        self.uncertainty_engine = UncertaintyEngine(num_samples=mc_dropout_samples)
        self.gradcam_explainer = GradCAMExplainer(colormap=gradcam_colormap)
        self.triage_advisor = ClinicalTriageAdvisor()

    def analyze_image(
        self,
        image_input: Union[str, Path, Image.Image],
        run_mc_dropout: bool = True,
        run_gradcam: bool = True,
        apply_clahe: bool = False,
        num_mc_samples: Optional[int] = None,
        gradcam_target_class: Optional[Union[int, str]] = None,
    ) -> Dict[str, Any]:
        """Perform end-to-end multi-class diagnostic evaluation of a retinal fundus photograph.
        
        Args:
            image_input: Filepath string, Path, or PIL Image.
            run_mc_dropout: Whether to execute stochastic forward passes for uncertainty.
            run_gradcam: Whether to generate Grad-CAM visual pathology heatmap.
            apply_clahe: Whether to enhance contrast before inference.
            num_mc_samples: Optional override for Monte Carlo Dropout pass count.
            gradcam_target_class: Optional override target class for Grad-CAM.
        """
        # 1. Image loading and quality audit
        raw_pil = self.preprocessor.load_image(image_input)
        quality_data = self.preprocessor.assess_image_quality(raw_pil)

        # 2. Standardized preprocessing
        img_tensor = self.preprocessor.preprocess_image(
            raw_pil,
            apply_clahe=apply_clahe,
        )

        # 3. Deterministic single-pass inference
        raw_preds = self.model.predict(img_tensor, verbose=0)
        probs_1d = np.squeeze(np.asarray(raw_preds))
        if probs_1d.ndim == 0:
            probs_1d = np.array([probs_1d])
        if len(probs_1d) != len(CLASS_LABELS):
            raise ValueError(f"Model output dimension mismatch: expected {len(CLASS_LABELS)}, got {len(probs_1d)}")

        top_idx = int(np.argmax(probs_1d))
        predicted_class = CLASS_LABELS[top_idx]
        display_name = CLASS_DISPLAY_NAMES[predicted_class]
        top_conf = float(probs_1d[top_idx] * 100.0)
        formatted_preds = self.preprocessor.format_predictions(probs_1d)

        # 4. Monte Carlo Dropout Uncertainty Quantification
        uncertainty_data: Optional[Dict[str, Any]] = None
        if run_mc_dropout:
            uncertainty_data = self.uncertainty_engine.evaluate_uncertainty(
                self.model,
                img_tensor,
                num_samples=num_mc_samples,
            )

        # 5. Visual Grad-CAM Explainability
        gradcam_data: Optional[Dict[str, Any]] = None
        if run_gradcam:
            target_cls = gradcam_target_class if gradcam_target_class is not None else top_idx
            gradcam_data = self.gradcam_explainer.compute_gradcam(
                self.model,
                img_tensor,
                raw_pil,
                target_class=target_cls,
            )

        # 6. Clinical Referral Triage
        triage_data = self.triage_advisor.generate_triage_report(
            predicted_class=predicted_class,
            confidence_percentage=top_conf,
            uncertainty_data=uncertainty_data,
            image_quality_data=quality_data,
        )

        return {
            "image_quality": quality_data,
            "deterministic_probabilities": formatted_preds,
            "predicted_class": predicted_class,
            "predicted_display_name": display_name,
            "confidence_percentage": round(top_conf, 2),
            "uncertainty": uncertainty_data,
            "gradcam": gradcam_data,
            "clinical_triage": triage_data,
        }

    def analyze_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image]],
        run_mc_dropout: bool = True,
        run_gradcam: bool = False,
    ) -> Dict[str, Any]:
        """Process a cohort of retinal fundus images and compile a triage prioritization ledger."""
        records: List[Dict[str, Any]] = []
        detailed_results: List[Dict[str, Any]] = []

        urgency_rank = {
            "AMBIGUOUS": 1,
            "URGENT_GLAUCOMA": 2,
            "URGENT_DR": 3,
            "ROUTINE_GLAUCOMA": 4,
            "ROUTINE_DR": 5,
            "ROUTINE_CATARACT": 6,
            "ANNUAL_NORMAL": 7,
        }

        for idx, img_in in enumerate(image_inputs):
            name = Path(img_in).name if isinstance(img_in, (str, Path)) else f"Patient_Exam_{idx+1:03d}"
            try:
                res = self.analyze_image(
                    img_in,
                    run_mc_dropout=run_mc_dropout,
                    run_gradcam=run_gradcam,
                )
                detailed_results.append(res)
                triage = res["clinical_triage"]
                unc = res["uncertainty"] or {}

                records.append({
                    "Patient_Identifier": name,
                    "Predicted_Condition": res["predicted_display_name"],
                    "Confidence_Pct": res["confidence_percentage"],
                    "Urgency_Tier": triage["urgency_tier"],
                    "Urgency_Code": triage["urgency_code"],
                    "SLA_Target": triage["response_sla"],
                    "Second_Opinion_Required": triage["second_opinion_required"],
                    "Predictive_Entropy_Bits": unc.get("shannon_entropy_bits", 0.0),
                    "Epistemic_Std": unc.get("epistemic_uncertainty_std", 0.0),
                    "Image_Quality_Grade": res["image_quality"]["grade"],
                    "ICD10_Code": triage["icd10_code"],
                    "_urgency_rank": urgency_rank.get(triage["urgency_code"], 99),
                })
            except Exception as e:
                records.append({
                    "Patient_Identifier": name,
                    "Predicted_Condition": "Processing Error",
                    "Confidence_Pct": 0.0,
                    "Urgency_Tier": "Error — Manual Inspection Required",
                    "Urgency_Code": "AMBIGUOUS",
                    "SLA_Target": "Immediate",
                    "Second_Opinion_Required": True,
                    "Predictive_Entropy_Bits": 2.0,
                    "Epistemic_Std": 1.0,
                    "Image_Quality_Grade": "C",
                    "ICD10_Code": "N/A",
                    "_urgency_rank": 0,
                })

        df = pd.DataFrame(records).sort_values(by=["_urgency_rank", "Confidence_Pct"], ascending=[True, False])
        df = df.drop(columns=["_urgency_rank"]).reset_index(drop=True)

        return {
            "total_exams_evaluated": len(image_inputs),
            "urgent_cases_count": int(np.sum(df["Second_Opinion_Required"] | df["Urgency_Code"].str.startswith("URGENT"))),
            "triage_summary_table": df,
            "detailed_analyses": detailed_results,
        }
