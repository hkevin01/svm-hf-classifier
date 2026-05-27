"""
# ============================================================
# ID            : MOD-DATA-001
# Requirement   : Load a Hugging Face dataset and return a clean
#                 (texts, labels) tuple for downstream processing.
# Purpose       : Centralise all dataset I/O so any HF text
#                 dataset can be swapped by changing one config
#                 key without touching other modules.
# Rationale     : Strict separation of I/O from ML logic enables
#                 easy unit testing without network access.
# Inputs        :
#   dataset_name - str   : HF dataset slug (e.g. 'ag_news').
#   split        - str   : 'train' | 'test' | 'validation'.
#   text_column  - str   : Column containing raw text.
#   label_column - str   : Column containing integer labels.
#   max_samples  - int   : Row cap - keeps RAM bounded.
# Outputs       :
#   texts  - list[str]  : Cleaned text, length <= max_samples.
#   labels - list[int]  : Aligned integer labels.
# Side Effects  : HF cache written to ~/.cache/huggingface/.
# Failure Modes : ValueError on bad column; network error on
#                 first download.
# Verification  : tests/test_data_loader.py
# References    : https://huggingface.co/docs/datasets
# ============================================================
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    # ID: HELP-CLEAN-001
    # Purpose   : Strip surrounding whitespace; collapse internal
    #             whitespace runs (spaces, tabs, newlines) to one space.
    # Inputs    : text - raw string.
    # Outputs   : normalised string.
    """
    return re.sub(r"\s+", " ", text.strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_hf_dataset(
    dataset_name: str = "ag_news",
    split: str = "train",
    text_column: str = "text",
    label_column: Optional[str] = "label",
    max_samples: int = 5000,
) -> Tuple[List[str], List[int]]:
    """
    # ID: FUNC-LOAD-001
    # Requirement   : Return (texts, labels) for the requested HF dataset.
    # Inputs        :
    #   dataset_name  - HF dataset identifier.
    #   split         - Dataset split to load.
    #   text_column   - Column name for raw text.
    #   label_column  - Column name for integer labels (None = no labels).
    #   max_samples   - Maximum rows; use a small value for quick tests.
    # Outputs        :
    #   texts  - list[str] of cleaned text samples.
    #   labels - list[int] of ground-truth labels (empty if None).
    # Preconditions  : Network available on first call.
    # Postconditions : len(texts) == len(labels) when label_column given.
    # Error Handling : Raises ValueError listing available columns on mismatch.
    """
    if max_samples < 2:
        raise ValueError(f"max_samples must be >= 2, got {max_samples}.")

    # Deferred import - module importable without 'datasets' installed
    from datasets import load_dataset

    print(f"[data_loader] Loading '{dataset_name}' split='{split}' ...")
    dataset = load_dataset(dataset_name, split=split)

    available = dataset.column_names
    if text_column not in available:
        raise ValueError(
            f"text_column='{text_column}' not found. Available: {available}"
        )
    if label_column is not None and label_column not in available:
        raise ValueError(
            f"label_column='{label_column}' not found. Available: {available}"
        )

    n = min(max_samples, len(dataset))
    dataset = dataset.select(range(n))
    print(f"[data_loader] Using {n} samples.")

    texts: List[str] = [_clean_text(str(row[text_column])) for row in dataset]
    labels: List[int] = (
        [int(row[label_column]) for row in dataset]
        if label_column is not None
        else []
    )
    return texts, labels
