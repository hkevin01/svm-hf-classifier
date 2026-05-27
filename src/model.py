"""
# ============================================================
# ID            : MOD-MODEL-001
# Requirement   : Train an SVM classifier on a pre-built feature
#                 matrix; optionally tune C and kernel via
#                 GridSearchCV.
# Purpose       : Isolate all sklearn SVM code so kernel choice,
#                 hyperparameters, and tuning strategy can be
#                 changed without touching other modules.
# Rationale     :
#   LinearSVC   - O(N) training; ideal for large TF-IDF matrices.
#   SVC(kernel) - kernel trick enables non-linear boundaries;
#                 better for dense SBERT embeddings.
#   GridSearchCV- exhaustive C / kernel search with cross-
#                 validation to find the best generalising model.
# Inputs        :
#   X_train      - np.ndarray (N_train, F) float32.
#   y_train      - array-like integer labels (N_train,).
#   use_linear   - bool : use LinearSVC (fast) vs SVC (flexible).
#   C            - float: regularisation inverse strength.
#   max_iter     - int  : solver iteration budget.
#   tune         - bool : run GridSearchCV instead of direct fit.
#   cv_folds     - int  : cross-validation fold count.
# Outputs       :
#   fitted model (LinearSVC | SVC | GridSearchCV best estimator).
# Side Effects  : None (pure computation; may take several minutes
#                 when tune=True on large datasets).
# Failure Modes : ConvergenceWarning if max_iter too low.
# Verification  : tests/test_model.py
# References    : https://scikit-learn.org/stable/modules/svm.html
# ============================================================
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_svm(
    X_train: np.ndarray,
    y_train: Any,
    use_linear: bool = True,
    C: float = 1.0,
    max_iter: int = 2000,
    tune: bool = False,
    cv_folds: int = 3,
    random_state: int = 42,
) -> Any:
    """
    # ID: FUNC-TRAIN-001
    # Requirement   : Fit an SVM and return the trained model.
    # Purpose       : Core training step invoked by main.py.
    # Inputs        :
    #   X_train    - (N, F) float32 feature matrix.
    #   y_train    - integer label array of length N.
    #   use_linear - True  -> LinearSVC (fast, sparse-friendly).
    #                False -> SVC with RBF kernel (dense embeddings).
    #   C          - Regularisation strength inverse (default 1.0).
    #   max_iter   - Maximum solver iterations.
    #   tune       - If True, run GridSearchCV over C / kernel grid.
    #   cv_folds   - Number of CV folds used when tune=True.
    #   random_state - RNG seed for reproducibility.
    # Outputs       : fitted model with .predict() method.
    # Preconditions : X_train.shape[0] == len(y_train).
    # Error Handling: Raises ValueError on shape mismatch.
    """
    # --- Shape guard ---
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train has {len(X_train)} rows but y_train has {len(y_train)} labels."
        )

    n_classes = len(set(y_train))
    print(
        f"[model] SVM training: n_samples={len(X_train)}, "
        f"n_features={X_train.shape[1]}, n_classes={n_classes}, "
        f"use_linear={use_linear}, tune={tune}"
    )

    if tune:
        return _grid_search(X_train, y_train, use_linear, cv_folds, random_state)

    if use_linear:
        # LinearSVC does not support predict_proba natively;
        # wrap with CalibratedClassifierCV for probability estimates.
        base = LinearSVC(C=C, max_iter=max_iter, random_state=random_state)
        clf = CalibratedClassifierCV(base, cv=3)
    else:
        # SVC with RBF kernel - better for dense embedding features
        clf = SVC(
            C=C,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=random_state,
            max_iter=max_iter,
        )

    clf.fit(X_train, y_train)
    print("[model] Training complete.")
    return clf


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """
    # ID: FUNC-PREDICT-001
    # Purpose   : Convenience wrapper around model.predict().
    # Inputs    : model - any fitted sklearn estimator; X - (N, F) array.
    # Outputs   : np.ndarray (N,) integer predictions.
    """
    return model.predict(X)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _grid_search(
    X_train: np.ndarray,
    y_train: Any,
    use_linear: bool,
    cv_folds: int,
    random_state: int,
) -> Any:
    """
    # ID: HELP-GRID-001
    # Purpose   : Run GridSearchCV to tune C (and kernel for SVC).
    # Rationale : Cross-validated grid search prevents overfitting to
    #             training-set hyperparameter choices.
    # Inputs    : X_train, y_train, use_linear, cv_folds, random_state.
    # Outputs   : Best estimator from GridSearchCV.
    # Side Effects: Verbose progress printed to stdout.
    # Constraints : May be slow for large datasets; reduce cv_folds or
    #               the param_grid if needed.
    """
    if use_linear:
        param_grid = {"estimator__C": [0.01, 0.1, 1.0, 10.0]}
        base = LinearSVC(max_iter=2000, random_state=random_state)
        estimator = CalibratedClassifierCV(base, cv=3)
        grid = GridSearchCV(
            estimator,
            param_grid,
            cv=cv_folds,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
        )
    else:
        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        }
        estimator = SVC(probability=True, random_state=random_state)
        grid = GridSearchCV(
            estimator,
            param_grid,
            cv=cv_folds,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
        )

    print(f"[model] GridSearchCV (cv={cv_folds}) - this may take a few minutes ...")
    grid.fit(X_train, y_train)
    print(f"[model] Best params : {grid.best_params_}")
    print(f"[model] Best CV F1  : {grid.best_score_:.4f}")
    return grid.best_estimator_
