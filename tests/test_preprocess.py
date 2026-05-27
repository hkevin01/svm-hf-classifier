"""
# ID: TEST-PRE-001
# Purpose: Unit tests for src/preprocess.build_features() TF-IDF mode.
#          No network access required.
"""
import numpy as np
import pytest
from src.preprocess import build_features


TEXTS = [
    "support vector machines are powerful classifiers",
    "sklearn provides many machine learning algorithms",
    "natural language processing with python",
    "text classification using tf idf features",
    "deep learning and neural networks are popular",
]


class TestBuildFeaturesTFIDF:
    def test_output_shape(self):
        X, vec = build_features(TEXTS, mode="tfidf", max_features=200)
        assert X.shape[0] == len(TEXTS)
        assert X.shape[1] <= 200

    def test_dtype_float32(self):
        X, _ = build_features(TEXTS, mode="tfidf", max_features=200)
        assert X.dtype == np.float32

    def test_reuse_vectorizer_shape(self):
        # Use min_df=1 override by using all 5 texts for fit (enough for min_df=2)
        X_tr, vec = build_features(TEXTS, mode="tfidf", max_features=200)
        X_te, vec2 = build_features(TEXTS[:2], mode="tfidf",
                                     max_features=200, vectorizer=vec)
        # Feature dim must match train
        assert X_tr.shape[1] == X_te.shape[1]
        assert vec2 is vec  # same object

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_features([], mode="tfidf")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            build_features(TEXTS, mode="foo")
