"""Unit tests for retinal image preprocessing, optical quality checks, and CLAHE."""

import numpy as np
import pytest
from PIL import Image

from src.preprocessor import (
    CLASS_LABELS,
    CLASS_DISPLAY_NAMES,
    TARGET_SIZE,
    RetinaPreprocessor,
    format_predictions,
    preprocess_image,
)


def test_class_labels_count():
    assert len(CLASS_LABELS) == 4
    assert set(CLASS_LABELS.values()) == {
        "cataract",
        "diabetic_retinopathy",
        "glaucoma",
        "normal",
    }


def test_display_names_mapping():
    assert len(CLASS_DISPLAY_NAMES) == 4
    assert CLASS_DISPLAY_NAMES["diabetic_retinopathy"] == "Diabetic Retinopathy"
    assert CLASS_DISPLAY_NAMES["normal"] == "Normal Retina"


def test_preprocess_image_shape_and_range():
    img = Image.new("RGB", (500, 400), color=(128, 64, 32))
    processed = preprocess_image(img, target_size=TARGET_SIZE)

    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == np.float32
    assert 0.0 <= processed.min()
    assert processed.max() <= 1.0


def test_preprocess_image_invalid_input():
    with pytest.raises(TypeError):
        preprocess_image("invalid_path.jpg")


def test_format_predictions():
    mock_probs = np.array([[0.1, 0.7, 0.15, 0.05]])
    results = format_predictions(mock_probs)

    assert results["diabetic_retinopathy"] == 70.0
    assert results["glaucoma"] == 15.0
    assert results["cataract"] == 10.0
    assert results["normal"] == 5.0


def test_format_predictions_invalid_dim():
    with pytest.raises(ValueError):
        format_predictions(np.array([0.5, 0.5]))


def test_assess_image_quality_optimal():
    # Crisp image with texture
    arr = np.random.randint(40, 200, (300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    quality = RetinaPreprocessor.assess_image_quality(img)

    assert quality["status"] in ["Optimal", "Acceptable"]
    assert quality["grade"] in ["A", "B"]
    assert quality["resolution"] == (300, 300)
    assert quality["laplacian_sharpness_score"] > 0


def test_assess_image_quality_blurry():
    # Uniform flat image (zero texture -> blurry)
    img = Image.new("RGB", (256, 256), color=(100, 100, 100))
    quality = RetinaPreprocessor.assess_image_quality(img)

    assert quality["is_blurry"] is True
    assert "Low optical sharpness / potential focus blur" in quality["quality_issues"]


def test_assess_image_quality_underexposed():
    # Almost black image
    img = Image.new("RGB", (256, 256), color=(5, 5, 5))
    quality = RetinaPreprocessor.assess_image_quality(img)

    assert quality["is_underexposed"] is True
    assert quality["grade"] == "C"


def test_clahe_enhancement():
    img = Image.new("RGB", (128, 128), color=(80, 50, 40))
    enhanced = RetinaPreprocessor.enhance_clahe(img)

    assert isinstance(enhanced, Image.Image)
    assert enhanced.size == (128, 128)


def test_aspect_ratio_preservation_letterbox():
    prep = RetinaPreprocessor(target_size=(224, 224), preserve_aspect_ratio=True)
    # 400x200 wide aspect image
    img = Image.new("RGB", (400, 200), color=(200, 100, 50))
    res = prep.preprocess_image(img)

    assert res.shape == (1, 224, 224, 3)
    # Check top/bottom letterbox borders have zero padding
    assert np.all(res[0, :5, :, :] == 0.0)
