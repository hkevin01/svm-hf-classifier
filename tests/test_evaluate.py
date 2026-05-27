"""
# ID: TEST-EVAL-001
# Purpose: Unit tests for src/evaluate.evaluate().
"""
import pytest
from src.evaluate import evaluate


class TestEvaluate:
    def test_perfect_predictions(self):
        y = [0, 1, 2, 0, 1, 2]
        m = evaluate(y, y)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"]       == pytest.approx(1.0)

    def test_keys_present(self):
        m = evaluate([0, 1, 0, 1], [0, 0, 0, 1])
        for k in ("accuracy", "precision", "recall", "f1", "report"):
            assert k in m

    def test_accuracy_in_range(self):
        y_true = [0, 1, 2] * 10
        y_pred = [0, 1, 0] * 10
        m = evaluate(y_true, y_pred)
        assert 0.0 <= m["accuracy"] <= 1.0
