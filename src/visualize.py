"""
# ============================================================
# ID            : MOD-VIZ-001
# Requirement   : Generate and save two diagnostic plots:
#                 1. Normalised confusion matrix heatmap.
#                 2. Top-N TF-IDF feature weights per class
#                    (skipped automatically in SBERT mode).
# Purpose       : Provide visual diagnostics for both overall
#                 classification quality and feature attribution.
# Inputs        :
#   y_true        - true labels (N,).
#   y_pred        - predicted labels (N,).
#   label_names   - class name strings.
#   output_path   - PNG save path (None = interactive display).
#   model         - fitted SVM (for feature importance).
#   vectorizer    - fitted TfidfVectorizer or None.
#   top_n         - number of top features to show per class.
# Outputs       : PNG files written to output_dir.
# Side Effects  : File I/O.
# Failure Modes : OSError if output_dir not writable.
# Verification  : tests/test_visualize.py
# ============================================================
"""

from __future__ import annotations

from typing import Any, List, Optional

import matplotlib
import numpy as np


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    label_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> str:
    """
    # ID: FUNC-CM-001
    # Requirement   : Render and save a row-normalised confusion matrix.
    # Inputs        :
    #   y_true      - ground-truth labels.
    #   y_pred      - model predictions.
    #   label_names - display names for axes.
    #   output_path - save path; None for interactive display.
    #   title       - figure title string.
    # Outputs       : output_path string.
    # Postconditions: PNG written when output_path is not None.
    """
    if output_path is not None:
        matplotlib.use("Agg")

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n_classes = cm.shape[0]
    fig_size = max(6, n_classes * 1.3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names or range(n_classes),
        yticklabels=label_names or range(n_classes),
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"[visualize] Confusion matrix saved to '{output_path}'.")
    else:
        plt.show()

    plt.close(fig)
    return output_path or ""


# ---------------------------------------------------------------------------
# TF-IDF feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model: Any,
    vectorizer: Any,
    label_names: Optional[List[str]] = None,
    top_n: int = 15,
    output_path: Optional[str] = None,
    title: str = "Top TF-IDF Features per Class (SVM weights)",
) -> str:
    """
    # ID: FUNC-FEAT-001
    # Requirement   : Plot LinearSVC coefficient magnitudes as a proxy
    #                 for feature importance per class.
    # Purpose       : Audit which tokens most influence SVM decisions.
    # Inputs        :
    #   model       - fitted estimator with a coef_ attribute, or a
    #                 CalibratedClassifierCV wrapping LinearSVC.
    #   vectorizer  - fitted TfidfVectorizer (None = SBERT mode, skip).
    #   label_names - class display names.
    #   top_n       - features per class subplot.
    #   output_path - PNG save path (None = interactive).
    # Outputs       : output_path string (empty string if skipped).
    # Error Handling: Returns '' silently when vectorizer is None.
    """
    if vectorizer is None:
        print("[visualize] Skipping feature importance (SBERT mode - no vectoriser).")
        return ""

    if output_path is not None:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # Extract coefficient matrix - handle CalibratedClassifierCV wrapper
    coefs = _extract_coef(model)
    if coefs is None:
        print("[visualize] Cannot extract coef_ from this model type - skipping.")
        return ""

    feature_names = np.array(vectorizer.get_feature_names_out())
    n_classes = coefs.shape[0]
    names = label_names or [f"Class {i}" for i in range(n_classes)]

    fig, axes = plt.subplots(
        1, n_classes, figsize=(max(top_n * 0.9, 8), 5), sharey=False
    )
    if n_classes == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        top_idx = np.argsort(coefs[i])[-top_n:][::-1]
        weights = coefs[i][top_idx]
        feats = feature_names[top_idx]
        colors = ["#1976D2" if w >= 0 else "#D32F2F" for w in weights]

        ax.barh(range(top_n), weights[::-1], color=colors[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feats[::-1], fontsize=8)
        ax.set_title(names[i], fontsize=10, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.6)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Feature importance saved to '{output_path}'.")
    else:
        plt.show()

    plt.close(fig)
    return output_path or ""


def _extract_coef(model: Any) -> Optional[np.ndarray]:
    """
    # ID: HELP-COEF-001
    # Purpose   : Extract the coefficient matrix from LinearSVC or a
    #             CalibratedClassifierCV wrapper around LinearSVC.
    # Inputs    : model - any fitted sklearn estimator.
    # Outputs   : np.ndarray (n_classes, n_features) or None if unavailable.
    """
    # Direct coef_ (LinearSVC or other linear models)
    if hasattr(model, "coef_"):
        return model.coef_

    # CalibratedClassifierCV: average calibrated estimator coefs
    if hasattr(model, "calibrated_classifiers_"):
        coefs = []
        for cal in model.calibrated_classifiers_:
            base = cal.estimator
            if hasattr(base, "coef_"):
                coefs.append(base.coef_)
        if coefs:
            return np.mean(coefs, axis=0)

    return None
