"""
# ============================================================
# ID            : MAIN-001
# Requirement   : Single runnable script executing the full SVM
#                 classification pipeline end-to-end.
# Purpose       : Wire together the src/ modules; expose all
#                 user-tunable options in one PIPELINE_CONFIG
#                 dict so nothing else needs editing.
# Inputs        : None - configured via PIPELINE_CONFIG below.
# Outputs       :
#   - Console evaluation report (accuracy, precision, recall, F1).
#   - outputs/confusion_matrix.png
#   - outputs/feature_importance.png  (TF-IDF + linear SVM only)
# Preconditions : pip install -r requirements.txt completed.
# Side Effects  : HF/model caches, PNG files in outputs/.
# Verification  : Run `python main.py` and inspect outputs/.
# ============================================================
"""

from __future__ import annotations

import os

from sklearn.model_selection import train_test_split

from src.data_loader  import load_hf_dataset
from src.preprocess   import build_features
from src.model        import train_svm, predict
from src.evaluate     import evaluate
from src.visualize    import plot_confusion_matrix, plot_feature_importance


# ============================================================
# Pipeline Configuration
# - Edit ANY value here to swap dataset, features, or SVM type.
# ============================================================

PIPELINE_CONFIG = {
    # ---- Dataset --------------------------------------------------------
    # ag_news  : 4-class news topic classification (fast, good baseline)
    "dataset_name":  "ag_news",
    "split":         "train",
    "text_column":   "text",
    "label_column":  "label",
    "max_samples":   4000,       # lower for quick smoke-test

    # Human-readable class names (None = use integers).
    # ag_news: 0=World  1=Sports  2=Business  3=Sci/Tech
    "label_names":   ["World", "Sports", "Business", "Sci/Tech"],

    # ---- Feature engineering -------------------------------------------
    # 'tfidf' - fast sparse TF-IDF; recommended with use_linear=True
    # 'sbert' - dense semantic embeddings; better with use_linear=False
    "embedding_mode":     "tfidf",
    "tfidf_max_features": 20_000,
    "sbert_model":        "sentence-transformers/all-MiniLM-L6-v2",

    # ---- Train / test split --------------------------------------------
    "test_size":    0.20,
    "random_state": 42,

    # ---- SVM hyperparameters -------------------------------------------
    # use_linear=True  -> LinearSVC (fast, great for TF-IDF)
    # use_linear=False -> SVC RBF   (slower, better for SBERT)
    "use_linear": True,
    "C":          1.0,
    "max_iter":   2000,

    # ---- Optional GridSearchCV hyperparameter tuning -------------------
    # Set tune=True to search C / kernel automatically (slower).
    # Reduce max_samples and cv_folds for faster tuning runs.
    "tune":     False,
    "cv_folds": 3,

    # ---- Output --------------------------------------------------------
    "output_dir": "outputs",
}


# ============================================================
# Alternative dataset snippets
# (replace the dataset block in PIPELINE_CONFIG)
# ============================================================
# imdb (2-class sentiment):
#   "dataset_name": "imdb", "label_names": ["negative", "positive"],
#
# emotion (6-class):
#   "dataset_name": "dair-ai/emotion",
#   "label_names": ["sadness","joy","love","anger","fear","surprise"],
#
# 20 newsgroups via HF:
#   "dataset_name": "SetFit/20_newsgroups", "label_names": None,
# ============================================================


def main() -> None:
    """
    # ID: FUNC-MAIN-001
    # Requirement   : Execute the full SVM pipeline with PIPELINE_CONFIG.
    # Preconditions : requirements.txt packages installed.
    # Postconditions: Metrics printed; PNGs saved in outputs/.
    """
    cfg = PIPELINE_CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 - Load dataset
    # ------------------------------------------------------------------
    texts, labels = load_hf_dataset(
        dataset_name=cfg["dataset_name"],
        split=cfg["split"],
        text_column=cfg["text_column"],
        label_column=cfg["label_column"],
        max_samples=cfg["max_samples"],
    )

    # ------------------------------------------------------------------
    # Step 2 - Train / test split
    # Split BEFORE feature engineering to prevent leakage.
    # ------------------------------------------------------------------
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=labels,
    )
    print(
        f"[main] Train: {len(texts_train)} samples | "
        f"Test:  {len(texts_test)} samples"
    )

    # ------------------------------------------------------------------
    # Step 3 - Feature engineering
    # Fit only on training data; transform test separately.
    # ------------------------------------------------------------------
    X_train, vectorizer = build_features(
        texts=texts_train,
        mode=cfg["embedding_mode"],
        max_features=cfg["tfidf_max_features"],
        model_name=cfg["sbert_model"],
        vectorizer=None,      # fit on train
    )

    X_test, _ = build_features(
        texts=texts_test,
        mode=cfg["embedding_mode"],
        max_features=cfg["tfidf_max_features"],
        model_name=cfg["sbert_model"],
        vectorizer=vectorizer,  # transform-only on test
    )

    # ------------------------------------------------------------------
    # Step 4 - Train SVM
    # ------------------------------------------------------------------
    model = train_svm(
        X_train=X_train,
        y_train=y_train,
        use_linear=cfg["use_linear"],
        C=cfg["C"],
        max_iter=cfg["max_iter"],
        tune=cfg["tune"],
        cv_folds=cfg["cv_folds"],
        random_state=cfg["random_state"],
    )

    # ------------------------------------------------------------------
    # Step 5 - Predict on the held-out test set
    # ------------------------------------------------------------------
    y_pred = predict(model, X_test)

    # ------------------------------------------------------------------
    # Step 6 - Evaluate
    # ------------------------------------------------------------------
    metrics = evaluate(
        y_true=y_test,
        y_pred=y_pred,
        label_names=cfg["label_names"],
    )

    # ------------------------------------------------------------------
    # Step 7 - Confusion matrix
    # ------------------------------------------------------------------
    cm_path = os.path.join(cfg["output_dir"], "confusion_matrix.png")
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        label_names=cfg["label_names"],
        output_path=cm_path,
        title=(
            f"SVM Confusion Matrix - {cfg['dataset_name']} "
            f"(F1={metrics['f1']:.3f})"
        ),
    )

    # ------------------------------------------------------------------
    # Step 8 - Feature importance (linear SVM + TF-IDF only)
    # ------------------------------------------------------------------
    fi_path = os.path.join(cfg["output_dir"], "feature_importance.png")
    plot_feature_importance(
        model=model,
        vectorizer=vectorizer,
        label_names=cfg["label_names"],
        top_n=15,
        output_path=fi_path,
        title=f"Top TF-IDF Features - {cfg['dataset_name']} (SVM weights)",
    )

    print(f"[main] Done. Outputs saved to '{cfg['output_dir']}/'.")


if __name__ == "__main__":
    main()
