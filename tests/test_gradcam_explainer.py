"""Unit tests for GradCAMExplainer."""

import numpy as np
from PIL import Image
from src.gradcam_explainer import GradCAMExplainer


def test_gradcam_output_keys(mock_model, sample_fundus_image):
    explainer = GradCAMExplainer(colormap="jet", alpha=0.45)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

    res = explainer.compute_gradcam(
        mock_model,
        dummy_input,
        sample_fundus_image,
        target_class=1,
    )

    assert "target_class_idx" in res
    assert res["target_class_idx"] == 1
    assert "target_class_label" in res
    assert res["target_class_label"] == "diabetic_retinopathy"
    assert "heatmap_matrix" in res
    assert "colored_heatmap_pil" in res
    assert "overlay_pil" in res
    assert "focal_pathology_boxes" in res


def test_gradcam_overlay_dimensions(mock_model):
    explainer = GradCAMExplainer()
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    custom_size_img = Image.new("RGB", (320, 180), color=(100, 30, 20))

    res = explainer.compute_gradcam(
        mock_model,
        dummy_input,
        custom_size_img,
        target_class="cataract",
    )

    assert res["overlay_pil"].size == (320, 180)
    assert res["heatmap_matrix"].shape == (180, 320)


def test_gradcam_alpha_clipping(mock_model, sample_fundus_image):
    explainer = GradCAMExplainer(alpha=1.5)  # Out of bounds
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

    res = explainer.compute_gradcam(
        mock_model,
        dummy_input,
        sample_fundus_image,
        alpha=1.5,
    )

    assert res["blend_alpha"] <= 0.90


def test_gradcam_focal_boxes_extraction():
    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[30:50, 40:60] = 0.95

    boxes = GradCAMExplainer._extract_focal_boxes(heatmap, threshold=0.70)

    assert len(boxes) == 1
    box = boxes[0]
    assert "bbox" in box
    assert "center_point" in box
    assert "peak_intensity" in box
    assert box["peak_intensity"] == 0.95
    assert 30 <= box["bbox"][0] <= 50
    assert 40 <= box["bbox"][1] <= 60


def test_gradcam_empty_focal_boxes_on_low_activation():
    heatmap = np.ones((50, 50), dtype=np.float32) * 0.20
    boxes = GradCAMExplainer._extract_focal_boxes(heatmap, threshold=0.60)
    assert len(boxes) == 0
