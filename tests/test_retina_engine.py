"""Integration tests for RetinaEngine facade."""

from PIL import Image
from src.retina_engine import RetinaEngine


def test_retina_engine_single_image_full_pipeline(mock_retina_engine, sample_fundus_image):
    res = mock_retina_engine.analyze_image(
        sample_fundus_image,
        run_mc_dropout=True,
        run_gradcam=True,
    )

    assert "image_quality" in res
    assert "deterministic_probabilities" in res
    assert "predicted_class" in res
    assert "confidence_percentage" in res
    assert "uncertainty" in res
    assert "gradcam" in res
    assert "clinical_triage" in res


def test_retina_engine_without_mc_dropout(mock_retina_engine, sample_fundus_image):
    res = mock_retina_engine.analyze_image(
        sample_fundus_image,
        run_mc_dropout=False,
        run_gradcam=True,
    )

    assert res["uncertainty"] is None
    assert res["gradcam"] is not None


def test_retina_engine_without_gradcam(mock_retina_engine, sample_fundus_image):
    res = mock_retina_engine.analyze_image(
        sample_fundus_image,
        run_mc_dropout=True,
        run_gradcam=False,
    )

    assert res["uncertainty"] is not None
    assert res["gradcam"] is None


def test_retina_engine_with_clahe(mock_retina_engine, sample_fundus_image):
    res = mock_retina_engine.analyze_image(
        sample_fundus_image,
        apply_clahe=True,
        run_mc_dropout=False,
        run_gradcam=False,
    )

    assert res["predicted_class"] in ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]


def test_retina_engine_batch_analysis(mock_retina_engine, sample_fundus_image):
    images = [
        sample_fundus_image,
        sample_fundus_image.resize((200, 200)),
        sample_fundus_image.resize((180, 180)),
    ]

    batch_res = mock_retina_engine.analyze_batch(images, run_mc_dropout=False, run_gradcam=False)

    assert batch_res["total_exams_evaluated"] == 3
    assert len(batch_res["triage_summary_table"]) == 3
    assert "Patient_Identifier" in batch_res["triage_summary_table"].columns
    assert "Urgency_Tier" in batch_res["triage_summary_table"].columns
