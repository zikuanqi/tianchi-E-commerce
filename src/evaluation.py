"""Competition-style set evaluation metrics.

The Tianchi 移动推荐算法 competition scores submissions with a set-based
F1 over (user_id, item_id) recommendation pairs — *not* a per-row
binary-classification F1. This module computes the official metric.

Definitions:
    PredictionSet = set of (user_id, item_id) pairs our model recommends
    ReferenceSet  = set of (user_id, item_id) pairs the user actually
                    purchased on the held-out day (within item subset P)

    Precision = |PredictionSet ∩ ReferenceSet| / |PredictionSet|
    Recall    = |PredictionSet ∩ ReferenceSet| / |ReferenceSet|
    F1        = 2 · P · R / (P + R)
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple, Union

import pandas as pd

PairLike = Union[pd.DataFrame, Iterable[Tuple[str, str]]]


def _to_pair_set(pairs: PairLike) -> set:
    """Normalize either a DataFrame or an iterable to a set of str tuples."""
    if isinstance(pairs, pd.DataFrame):
        if 'user_id' not in pairs.columns or 'item_id' not in pairs.columns:
            raise ValueError("DataFrame must have 'user_id' and 'item_id' columns")
        return set(
            (str(u), str(i))
            for u, i in zip(pairs['user_id'].tolist(), pairs['item_id'].tolist())
        )
    return set((str(u), str(i)) for u, i in pairs)


def set_precision_recall_f1(predicted: PairLike,
                            reference: PairLike) -> Dict[str, float]:
    """Compute set-based precision, recall, and F1.

    Inputs may be either pandas DataFrames (with ``user_id`` and
    ``item_id`` columns) or any iterable of (user_id, item_id) tuples.
    Pairs are compared as strings so the (str, int) mismatch between
    a freshly built submission and a freshly loaded ground truth
    doesn't silently produce zero recall.

    Returns a dict with keys: precision, recall, f1, tp,
    predicted_size, reference_size.
    """
    pred_set = _to_pair_set(predicted)
    ref_set = _to_pair_set(reference)
    tp = len(pred_set & ref_set)

    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(ref_set) if ref_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0
          else 0.0)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'predicted_size': len(pred_set),
        'reference_size': len(ref_set),
    }


__all__ = ['set_precision_recall_f1']
