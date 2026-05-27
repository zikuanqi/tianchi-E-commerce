"""Set-based F1 evaluation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation import set_precision_recall_f1


def test_perfect_match():
    pairs = [('u1', 'i1'), ('u2', 'i2')]
    m = set_precision_recall_f1(pairs, pairs)
    assert m['precision'] == 1.0
    assert m['recall'] == 1.0
    assert m['f1'] == 1.0
    assert m['tp'] == 2


def test_no_overlap():
    m = set_precision_recall_f1([('u1', 'i1')], [('u2', 'i2')])
    assert m['precision'] == 0.0
    assert m['recall'] == 0.0
    assert m['f1'] == 0.0
    assert m['tp'] == 0


def test_half_overlap():
    predicted = [('u1', 'i1'), ('u2', 'i2'), ('u3', 'i3'), ('u4', 'i4')]
    reference = [('u1', 'i1'), ('u2', 'i2')]
    m = set_precision_recall_f1(predicted, reference)
    # tp=2, |pred|=4, |ref|=2 → P=0.5, R=1.0, F1=2/3
    assert m['precision'] == pytest.approx(0.5)
    assert m['recall'] == pytest.approx(1.0)
    assert m['f1'] == pytest.approx(2 / 3)


def test_dataframe_input():
    predicted = pd.DataFrame({'user_id': ['u1', 'u2'], 'item_id': ['i1', 'i2']})
    reference = pd.DataFrame({'user_id': ['u1'], 'item_id': ['i1']})
    m = set_precision_recall_f1(predicted, reference)
    assert m['precision'] == pytest.approx(0.5)
    assert m['recall'] == pytest.approx(1.0)


def test_dtype_coercion_no_silent_zero_recall():
    """Common pitfall: pandas writes ids as ints, ground-truth has strings.
    Without coercion this collapses to zero. We coerce to str on both
    sides so the metric reports the real overlap."""
    predicted = pd.DataFrame({'user_id': [1, 2], 'item_id': [10, 20]})
    reference = pd.DataFrame({'user_id': ['1', '2'], 'item_id': ['10', '20']})
    m = set_precision_recall_f1(predicted, reference)
    assert m['f1'] == 1.0


def test_empty_sets():
    m = set_precision_recall_f1([], [])
    assert m['precision'] == 0.0
    assert m['recall'] == 0.0
    assert m['f1'] == 0.0


def test_dataframe_missing_columns_raises():
    bad = pd.DataFrame({'foo': ['u1'], 'bar': ['i1']})
    with pytest.raises(ValueError, match='user_id'):
        set_precision_recall_f1(bad, [('u1', 'i1')])
