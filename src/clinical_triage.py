"""Clinical triage and ophthalmological referral advisor based on CAD predictions and uncertainty."""

from typing import Any, Dict, List, Optional
from src.preprocessor import CLASS_LABELS, CLASS_DISPLAY_NAMES

# Standardized Clinical Triage Metadata
ICD10_CODES: Dict[str, str] = {
    "cataract": "H25.9 (Age-related cataract, unspecified)",
    "diabetic_retinopathy": "E11.319 (Type 2 diabetes mellitus with diabetic retinopathy)",
    "glaucoma": "H40.9 (Glaucoma, unspecified / open-angle suspect)",
    "normal": "Z01.00 (Encounter for examination of eyes without abnormal findings)",
}

DIAGNOSTIC_MODALITIES: Dict[str, List[str]] = {
    "cataract": [
        "Slit-lamp biomicroscopy under pharmacological mydriasis",
        "Best-Corrected Visual Acuity (BCVA) refraction testing",
        "Optical biometry (IOL Master / Lenstar) for intraocular lens calculation",
    ],
    "diabetic_retinopathy": [
        "High-density Spectral-Domain Macular OCT (SD-OCT)",
        "Ultra-widefield Fluorescein Angiography (UW-FA)",
        "Serum Glycated Hemoglobin (HbA1c) and blood pressure review",
    ],
    "glaucoma": [
        "Goldmann Applanation Tonometry (GAT) for Intraocular Pressure (IOP)",
        "Automated Perimetry (Humphrey Visual Field 24-2 / SITA-Faster)",
        "Peripapillary Retinal Nerve Fiber Layer (RNFL) OCT thickness scan",
        "Pachymetry (Central Corneal Thickness measurement)",
    ],
    "normal": [
        "Standard annual comprehensive dilated eye examination",
        "Routine visual acuity and screening intraocular pressure check",
    ],
}


class ClinicalTriageAdvisor:
    """Translates CAD model predictions and uncertainty metrics into actionable clinical triage protocols."""

    def __init__(self, high_urgency_confidence_threshold: float = 85.0) -> None:
        self.high_urgency_confidence_threshold = high_urgency_confidence_threshold

    def generate_triage_report(
        self,
        predicted_class: str,
        confidence_percentage: float,
        uncertainty_data: Optional[Dict[str, Any]] = None,
        image_quality_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate structured clinical referral triage report with urgency SLA and safety flags.
        
        Args:
            predicted_class: Key from CLASS_LABELS ('cataract', 'diabetic_retinopathy', etc.).
            confidence_percentage: Model confidence (0-100%).
            uncertainty_data: Optional output dictionary from UncertaintyEngine.
            image_quality_data: Optional output dictionary from RetinaPreprocessor.assess_image_quality.
        """
        p_class = predicted_class.lower()
        conf = float(confidence_percentage)

        # Extract uncertainty metrics if present
        entropy = uncertainty_data.get("shannon_entropy_bits", 0.0) if uncertainty_data else 0.0
        norm_entropy = uncertainty_data.get("normalized_entropy", 0.0) if uncertainty_data else 0.0
        epistemic_std = uncertainty_data.get("epistemic_uncertainty_std", 0.0) if uncertainty_data else 0.0
        flagged_by_engine = uncertainty_data.get("second_opinion_recommended", False) if uncertainty_data else False

        # Image quality check flags
        is_suboptimal_image = False
        if image_quality_data:
            is_suboptimal_image = image_quality_data.get("grade") == "C" or image_quality_data.get("is_blurry", False)

        # 1. Determine Second-Opinion Safeguard Trigger
        is_ambiguous = (
            conf < 60.0
            or norm_entropy > 0.65
            or epistemic_std > 0.12
            or flagged_by_engine
            or is_suboptimal_image
        )

        # 2. Determine Clinical Urgency Tier & SLA
        if is_ambiguous:
            urgency = "Secondary Specialist Consultation Mandatory"
            urgency_code = "AMBIGUOUS"
            urgency_sla = "Within 48-72 Hours"
            rationale = (
                "High predictive entropy or optical blur detected; automated classification cannot rule out "
                "subtle co-existing retinal micro-pathologies. Immediate review by a certified ophthalmologist recommended."
            )
        elif p_class == "glaucoma":
            if conf >= self.high_urgency_confidence_threshold:
                urgency = "Tier 1 - High-Priority Glaucoma Workup"
                urgency_code = "URGENT_GLAUCOMA"
                urgency_sla = "Within 1-2 Weeks"
                rationale = "High-confidence glaucomatous neuroretinal rim thinning or cupping; immediate visual field and IOP workup indicated."
            else:
                urgency = "Tier 2 - Glaucoma Suspect Evaluation"
                urgency_code = "ROUTINE_GLAUCOMA"
                urgency_sla = "Within 3-4 Weeks"
                rationale = "Moderate confidence optic disc excavation; formal gonioscopy and baseline perimetry recommended."
        elif p_class == "diabetic_retinopathy":
            if conf >= self.high_urgency_confidence_threshold:
                urgency = "Tier 1 - Expedited Diabetic Retinal Clinic"
                urgency_code = "URGENT_DR"
                urgency_sla = "Within 1-2 Weeks"
                rationale = "Prominent microvascular lesions detected; urgent macular OCT required to rule out diabetic macular edema."
            else:
                urgency = "Tier 2 - Non-Proliferative DR Screening"
                urgency_code = "ROUTINE_DR"
                urgency_sla = "Within 4-6 Weeks"
                rationale = "Early-stage retinal microaneurysms suspected; confirm staging and review glycemic HbA1c control."
        elif p_class == "cataract":
            urgency = "Tier 3 - Routine Anterior Segment / Cataract Consult"
            urgency_code = "ROUTINE_CATARACT"
            urgency_sla = "Within 1-3 Months"
            rationale = "Significant optical density attenuation observed; evaluate functional visual acuity impairment for elective phacoemulsification."
        else:  # normal
            urgency = "Tier 4 - Routine Annual Surveillance"
            urgency_code = "ANNUAL_NORMAL"
            urgency_sla = "12 Months"
            rationale = "Fundus appearance within normal anatomical limits; schedule annual preventive dilated eye exam."

        # 3. Formulate Actionable Primary Care Checklist
        action_checklist: List[str] = []
        if is_ambiguous:
            action_checklist.append("Repeat dilated fundus photograph with verified optical focus.")
            action_checklist.append("Schedule senior vitreoretinal / glaucoma specialist telemedicine review.")
        elif p_class in ["glaucoma", "diabetic_retinopathy"]:
            action_checklist.append(f"Issue referral to Comprehensive Ophthalmology department ({urgency_sla}).")
            action_checklist.append("Perform in-clinic Goldmann / non-contact tonometry and record IOP.")
            action_checklist.append("Obtain blood pressure and fasting glucose / HbA1c history.")
        elif p_class == "cataract":
            action_checklist.append("Assess subjective glare disability and visual acuity with Snellen chart.")
            action_checklist.append("Schedule anterior segment slit-lamp evaluation.")
        else:
            action_checklist.append("Provide patient with standard diabetic and ocular wellness educational material.")
            action_checklist.append("Issue reminder for next annual routine retinal screening.")

        return {
            "predicted_class": p_class,
            "display_name": CLASS_DISPLAY_NAMES.get(p_class, p_class),
            "confidence_percentage": round(conf, 2),
            "urgency_tier": urgency,
            "urgency_code": urgency_code,
            "response_sla": urgency_sla,
            "clinical_rationale": rationale,
            "second_opinion_required": is_ambiguous,
            "icd10_code": ICD10_CODES.get(p_class, "N/A"),
            "recommended_modalities": DIAGNOSTIC_MODALITIES.get(p_class, []),
            "action_checklist": action_checklist,
            "regulatory_disclaimer": (
                "RESEARCH PROTOCOL NOTICE: This computer-aided triage report is intended solely to support "
                "clinical workflow prioritization and does not constitute a standalone medical diagnosis. "
                "Final clinical determinations must be made by a licensed ophthalmologist."
            ),
        }
