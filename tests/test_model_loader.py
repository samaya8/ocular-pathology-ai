"""Unit tests for ModelLoader and MockKerasModel."""

import numpy as np
from src.model_loader import MockKerasModel, ModelLoader


def test_mock_model_predict_shape_and_sum():
    mock = MockKerasModel()
    dummy_input = np.zeros((2, 224, 224, 3), dtype=np.float32)
    preds = mock.predict(dummy_input)

    assert preds.shape == (2, 4)
    assert np.all(preds >= 0.0)
    for p in preds:
        assert np.isclose(np.sum(p), 1.0, atol=1e-5)


def test_mock_model_deterministic_when_training_false():
    mock = MockKerasModel()
    dummy_input = np.ones((1, 224, 224, 3), dtype=np.float32) * 0.5
    p1 = mock.predict(dummy_input, training=False)
    p2 = mock.predict(dummy_input, training=False)

    assert np.allclose(p1, p2)


def test_mock_model_stochastic_when_training_true():
    mock = MockKerasModel()
    dummy_input = np.ones((1, 224, 224, 3), dtype=np.float32) * 0.5
    p1 = mock.predict(dummy_input, training=True)
    p2 = mock.predict(dummy_input, training=True)

    # Stochastic dropout adds noise
    assert not np.allclose(p1, p2)


def test_model_loader_mock_singleton():
    m1 = ModelLoader.get_model(use_mock=True)
    m2 = ModelLoader.get_model(use_mock=True)

    assert m1 is m2
    assert isinstance(m1, MockKerasModel)


def test_model_loader_reset():
    _ = ModelLoader.get_model(use_mock=True)
    ModelLoader.reset()
    assert ModelLoader._cached_model is None
