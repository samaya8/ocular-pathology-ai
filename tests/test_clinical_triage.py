"""Unit tests for ClinicalTriageAdvisor."""

from src.clinical_triage import ClinicalTriageAdvisor


def test_triage_high_confidence_glaucoma():
    advisor = ClinicalTriageAdvisor(high_urgency_confidence_threshold=85.0)
    rep = advisor.generate_triage_report("glaucoma", confidence_percentage=92.5)

    assert rep["urgency_code"] == "URGENT_GLAUCOMA"
    assert "Within 1-2 Weeks" in rep["response_sla"]
    assert "H40.9" in rep["icd10_code"]
    assert len(rep["recommended_modalities"]) >= 3
    assert len(rep["action_checklist"]) >= 2
    assert rep["second_opinion_required"] is False


def test_triage_routine_glaucoma():
    advisor = ClinicalTriageAdvisor(high_urgency_confidence_threshold=85.0)
    rep = advisor.generate_triage_report("glaucoma", confidence_percentage=72.0)

    assert rep["urgency_code"] == "ROUTINE_GLAUCOMA"
    assert "Within 3-4 Weeks" in rep["response_sla"]


def test_triage_high_confidence_dr():
    advisor = ClinicalTriageAdvisor(high_urgency_confidence_threshold=85.0)
    rep = advisor.generate_triage_report("diabetic_retinopathy", confidence_percentage=90.0)

    assert rep["urgency_code"] == "URGENT_DR"
    assert "Within 1-2 Weeks" in rep["response_sla"]
    assert "E11.319" in rep["icd10_code"]


def test_triage_routine_dr():
    advisor = ClinicalTriageAdvisor(high_urgency_confidence_threshold=85.0)
    rep = advisor.generate_triage_report("diabetic_retinopathy", confidence_percentage=75.0)

    assert rep["urgency_code"] == "ROUTINE_DR"
    assert "Within 4-6 Weeks" in rep["response_sla"]


def test_triage_cataract():
    advisor = ClinicalTriageAdvisor()
    rep = advisor.generate_triage_report("cataract", confidence_percentage=88.0)

    assert rep["urgency_code"] == "ROUTINE_CATARACT"
    assert "H25.9" in rep["icd10_code"]
    assert "Within 1-3 Months" in rep["response_sla"]


def test_triage_normal_retina():
    advisor = ClinicalTriageAdvisor()
    rep = advisor.generate_triage_report("normal", confidence_percentage=95.0)

    assert rep["urgency_code"] == "ANNUAL_NORMAL"
    assert "Z01.00" in rep["icd10_code"]
    assert "12 Months" in rep["response_sla"]


def test_triage_second_opinion_trigger_on_ambiguity():
    advisor = ClinicalTriageAdvisor()
    # Low confidence triggers safeguard
    rep = advisor.generate_triage_report("diabetic_retinopathy", confidence_percentage=54.0)

    assert rep["second_opinion_required"] is True
    assert rep["urgency_code"] == "AMBIGUOUS"
    assert "48-72 Hours" in rep["response_sla"]


def test_triage_second_opinion_trigger_on_entropy():
    advisor = ClinicalTriageAdvisor()
    uncertainty_mock = {
        "shannon_entropy_bits": 1.5,
        "normalized_entropy": 0.75,  # > 0.65 threshold
        "epistemic_uncertainty_std": 0.08,
        "second_opinion_recommended": True,
    }
    rep = advisor.generate_triage_report("cataract", confidence_percentage=80.0, uncertainty_data=uncertainty_mock)

    assert rep["second_opinion_required"] is True
    assert rep["urgency_code"] == "AMBIGUOUS"


def test_triage_regulatory_disclaimer():
    advisor = ClinicalTriageAdvisor()
    rep = advisor.generate_triage_report("normal", confidence_percentage=90.0)

    assert "RESEARCH PROTOCOL NOTICE" in rep["regulatory_disclaimer"]
