"""
# ID: TEST-MODEL-001
# Purpose: Unit tests for src/model.train_svm() and predict().
#          Uses synthetic in-memory data - no network required.
"""
import numpy as np
import pytest
from src.model import train_svm, predict


def _blobs(n=90, n_feat=20, n_classes=3, seed=7):
    """Synthetic well-separated blobs for quick SVM tests."""
    rng = np.random.default_rng(seed)
    parts = []
    for k in range(n_classes):
        centre = np.zeros(n_feat)
        centre[k % n_feat] = 5.0 * (k + 1)
        parts.append(rng.normal(centre, 0.3, (n // n_classes, n_feat)))
    X = np.vstack(parts).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms == 0, 1, norms)
    y = [k for k in range(n_classes) for _ in range(n // n_classes)]
    return X, y


class TestTrainSVM:
    def test_linear_returns_model(self):
        X, y = _blobs()
        model = train_svm(X, y, use_linear=True)
        assert hasattr(model, "predict")

    def test_rbf_returns_model(self):
        X, y = _blobs()
        model = train_svm(X, y, use_linear=False, max_iter=500)
        assert hasattr(model, "predict")

    def test_predict_shape(self):
        X, y = _blobs()
        model = train_svm(X, y, use_linear=True)
        preds = predict(model, X)
        assert preds.shape == (len(y),)

    def test_shape_mismatch_raises(self):
        X, y = _blobs()
        with pytest.raises(ValueError, match="rows but y_train has"):
            train_svm(X, y[:-1])

    def test_grid_search(self):
        X, y = _blobs()
        model = train_svm(X, y, use_linear=True, tune=True, cv_folds=2)
        preds = predict(model, X)
        assert preds.shape == (len(y),)
