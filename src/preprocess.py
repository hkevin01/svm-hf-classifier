"""
# ============================================================
# ID            : MOD-PREPROCESS-001
# Requirement   : Transform raw text samples into a numeric
#                 feature matrix suitable for sklearn SVM.
# Purpose       : Decouple feature engineering from model code;
#                 supports two modes switchable via a config key.
# Rationale     :
#   tfidf - fast, interpretable, good SVM baseline.
#           LinearSVC trains in O(N*F) with sparse matrices.
#   sbert - dense semantic embeddings; better recall on short
#           or paraphrased text; normalised for cosine SVM.
# Inputs        :
#   texts        - list[str] of N text samples.
#   mode         - 'tfidf' | 'sbert'.
#   max_features - TF-IDF vocabulary cap.
#   model_name   - SentenceTransformer model id (SBERT mode).
#   vectorizer   - Pre-fitted vectoriser (test-set transform).
# Outputs       :
#   X          - np.ndarray (N, F) float32 feature matrix.
#   vectorizer - fitted TfidfVectorizer or None (SBERT mode).
# Side Effects  : SBERT model cached on first call.
# Failure Modes : ValueError on unknown mode; ImportError if
#                 sentence-transformers absent in SBERT mode.
# Verification  : tests/test_preprocess.py
# ============================================================
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    texts: list,
    mode: str = "tfidf",
    max_features: int = 20_000,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    vectorizer: Optional[Any] = None,
) -> Tuple[np.ndarray, Any]:
    """
    # ID: FUNC-FEATURES-001
    # Requirement   : Produce a float32 feature matrix from text samples.
    # Purpose       : Single entry-point for feature engineering; called
    #                 separately on train and test splits.
    # Inputs        :
    #   texts       - non-empty list[str].
    #   mode        - 'tfidf' or 'sbert'.
    #   max_features- TF-IDF vocabulary size (ignored in SBERT mode).
    #   model_name  - HF model identifier (SBERT mode only).
    #   vectorizer  - If not None, call .transform() to avoid leakage.
    # Outputs       :
    #   X           - (N, F) float32 array.
    #   vectorizer  - fitted TfidfVectorizer | None.
    # Preconditions : len(texts) >= 1.
    # Error Handling: ValueError on empty input or unknown mode.
    """
    if not texts:
        raise ValueError("build_features: texts list is empty.")

    mode = mode.lower()
    if mode == "tfidf":
        return _tfidf_features(texts, max_features, vectorizer)
    elif mode == "sbert":
        return _sbert_features(texts, model_name)
    else:
        raise ValueError(f"Unknown mode='{mode}'. Choose 'tfidf' or 'sbert'.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _tfidf_features(
    texts: list,
    max_features: int,
    vectorizer: Optional[Any],
) -> Tuple[np.ndarray, Any]:
    """
    # ID: HELP-TFIDF-001
    # Purpose   : Fit (or reuse) a TF-IDF vectoriser; return dense float32.
    # Rationale : sublinear_tf + bigrams boost SVM performance on news text.
    # Inputs    : texts, max_features, optional pre-fitted vectorizer.
    # Outputs   : (X float32, fitted TfidfVectorizer).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # deferred

    if vectorizer is None:
        print(f"[preprocess] Fitting TF-IDF (max_features={max_features}) ...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
        )
        X_sparse = vectorizer.fit_transform(texts)
    else:
        print("[preprocess] Transforming with pre-fitted TF-IDF ...")
        X_sparse = vectorizer.transform(texts)

    X = X_sparse.toarray().astype(np.float32)
    print(f"[preprocess] TF-IDF shape: {X.shape}")
    return X, vectorizer


def _sbert_features(
    texts: list,
    model_name: str,
) -> Tuple[np.ndarray, None]:
    """
    # ID: HELP-SBERT-001
    # Purpose   : Encode texts with a SentenceTransformer model.
    # Rationale : L2-normalised SBERT vectors work well with a linear
    #             SVM using cosine distance (equivalent to dot product).
    # Outputs   : (X float32 unit-normalised, None).
    """
    from sentence_transformers import SentenceTransformer  # deferred

    print(f"[preprocess] Loading SBERT model '{model_name}' ...")
    model = SentenceTransformer(model_name)
    print(f"[preprocess] Encoding {len(texts)} texts ...")
    X = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"[preprocess] SBERT shape: {X.shape}")
    return X, None
