"""Pytest configuration and fixtures for Retinal Disease Classification."""

import sys
from pathlib import Path
import pytest
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.model_loader import MockKerasModel, ModelLoader
from src.retina_engine import RetinaEngine


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Ensure tests run cleanly with mock mode enabled by default."""
    monkeypatch.setenv("RETINA_MOCK_MODEL", "1")
    ModelLoader.reset()
    yield
    ModelLoader.reset()


@pytest.fixture
def mock_model():
    """Return an initialized mock model."""
    return MockKerasModel()


@pytest.fixture
def sample_fundus_image():
    """Return a clean RGB PIL fundus image for test cases."""
    img = Image.new("RGB", (256, 256), color=(140, 35, 20))
    return img


@pytest.fixture
def sample_blurry_image():
    """Return a blurry image to trigger quality warnings."""
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    return img


@pytest.fixture
def mock_retina_engine():
    """Return an initialized RetinaEngine configured with mock model."""
    return RetinaEngine(use_mock=True, mc_dropout_samples=5)
