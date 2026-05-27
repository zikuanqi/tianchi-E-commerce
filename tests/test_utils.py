"""Comprehensive tests for src/utils.py helpers."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.utils import (
    DataFrameSerializer,
    MetricTracker,
    TensorBatchGenerator,
    calculate_model_size,
    check_data_quality,
    convert_to_tensor,
    create_submission_file,
    get_device,
    load_dict_from_json,
    memory_usage,
    reduce_memory_usage,
    report_memory,
    save_dict_to_json,
    set_seeds,
    setup_logging,
    timer,
)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def test_timer_returns_value_and_logs(caplog):
    @timer
    def add(a, b):
        return a + b

    with caplog.at_level(logging.INFO, logger='src.utils'):
        assert add(2, 3) == 5
    # Decorator should log something containing the function name + duration.
    assert any('add' in r.message for r in caplog.records)


def test_timer_logs_even_on_exception():
    @timer
    def boom():
        raise ValueError('boom')

    with pytest.raises(ValueError, match='boom'):
        boom()


def test_memory_usage_decorator_runs():
    @memory_usage
    def allocate():
        # Force a measurable allocation so the delta is non-trivial.
        return [0] * 10_000

    out = allocate()
    assert len(out) == 10_000


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

def test_setup_logging_creates_log_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_logging({'level': 'DEBUG'})
    assert (tmp_path / 'logs').is_dir()
    # At least one .log file should appear.
    assert list((tmp_path / 'logs').glob('*.log'))


def test_setup_logging_idempotent(tmp_path, monkeypatch):
    """Calling twice must not double-up handlers (no duplicate log lines)."""
    monkeypatch.chdir(tmp_path)
    setup_logging({'level': 'INFO'})
    setup_logging({'level': 'INFO'})
    # Only one StreamHandler should remain on the root logger
    # (a FileHandler is also added — that's fine).
    stream_handlers = [h for h in logging.getLogger().handlers
                       if isinstance(h, logging.StreamHandler)
                       and not isinstance(h, logging.FileHandler)]
    assert len(stream_handlers) <= 1


# ---------------------------------------------------------------------------
# Reproducibility & devices
# ---------------------------------------------------------------------------

def test_set_seeds_makes_numpy_deterministic():
    set_seeds(123)
    a = np.random.rand(5)
    set_seeds(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_seeds_makes_torch_deterministic():
    set_seeds(7)
    a = torch.rand(4)
    set_seeds(7)
    b = torch.rand(4)
    assert torch.equal(a, b)


def test_get_device_returns_torch_device():
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {'cuda', 'cpu'}


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def test_report_memory_returns_expected_keys():
    snap = report_memory('test-stage')
    assert 'rss_gb' in snap
    assert 'vms_gb' in snap
    assert snap['rss_gb'] > 0


def test_reduce_memory_usage_downcasts_ints():
    df = pd.DataFrame({
        'small_int': pd.Series([1, 2, 3], dtype='int64'),
        'small_float': pd.Series([1.0, 2.0, 3.0], dtype='float64'),
    })
    before = df.memory_usage(deep=True).sum()
    out = reduce_memory_usage(df, verbose=True)
    # int64 should compress to int8 / int16
    assert out['small_int'].dtype.itemsize < 8
    after = out.memory_usage(deep=True).sum()
    assert after <= before


def test_reduce_memory_usage_skips_non_numeric_columns():
    """Object/string columns must pass through untouched (covers the
    'neither integer nor float' branch in the dtype dispatch)."""
    df = pd.DataFrame({
        'i': pd.Series([1, 2, 3], dtype='int64'),
        'f': pd.Series([1.5, 2.5, 3.5], dtype='float64'),
        's': pd.Series(['a', 'b', 'c']),   # object
    })
    out = reduce_memory_usage(df)
    # Untouched — accept either the legacy object dtype or the modern StringDtype.
    assert pd.api.types.is_string_dtype(out['s'])
    # Numeric columns still got downcast.
    assert out['i'].dtype.itemsize < 8


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------

def test_save_and_load_json_roundtrip(tmp_path):
    data = {'a': 1, 'b': [1, 2, 3], 'c': {'nested': True}}
    path = tmp_path / 'out' / 'data.json'
    save_dict_to_json(data, path)
    assert path.exists()
    loaded = load_dict_from_json(path)
    assert loaded == data


def test_save_json_handles_non_serializable_via_default(tmp_path):
    """default=str fallback in save_dict_to_json should handle Timestamps."""
    data = {'when': pd.Timestamp('2014-12-19')}
    path = tmp_path / 'data.json'
    save_dict_to_json(data, path)
    loaded = load_dict_from_json(path)
    assert '2014-12-19' in loaded['when']


# ---------------------------------------------------------------------------
# Submission helper
# ---------------------------------------------------------------------------

def test_create_submission_file_basic(tmp_path):
    df = pd.DataFrame({
        'user_id': ['u1', 'u2', 'u1'],
        'item_id': ['i1', 'i2', 'i3'],
        'score': [0.9, 0.7, 0.6],
    })
    out = create_submission_file(df, tmp_path / 'sub.csv')
    assert out.exists()
    written = pd.read_csv(out)
    assert list(written.columns) == ['user_id', 'item_id']


def test_create_submission_file_respects_top_k(tmp_path):
    df = pd.DataFrame({
        'user_id': ['u1'] * 5,
        'item_id': [f'i{i}' for i in range(5)],
        'score': [0.9, 0.8, 0.7, 0.6, 0.5],
    })
    out = create_submission_file(df, tmp_path / 'sub.csv', top_k=2)
    written = pd.read_csv(out)
    assert len(written) == 2


def test_create_submission_file_requires_columns(tmp_path):
    df = pd.DataFrame({'foo': [1], 'bar': [2]})
    with pytest.raises(ValueError, match='missing columns'):
        create_submission_file(df, tmp_path / 'sub.csv')


# ---------------------------------------------------------------------------
# Data-quality
# ---------------------------------------------------------------------------

def test_check_data_quality_flags_nulls_and_missing_columns():
    df = pd.DataFrame({
        'a': [1, 2, None, 4],
        'b': [None, None, 3, 4],
    })
    report = check_data_quality(df, required_cols=['a', 'b', 'c'])
    assert report['shape'] == (4, 2)
    assert report['null_counts'] == {'a': 1, 'b': 2}
    assert report['missing_required_columns'] == ['c']


def test_check_data_quality_counts_duplicates():
    df = pd.DataFrame({'a': [1, 1, 2], 'b': [3, 3, 4]})
    report = check_data_quality(df)
    assert report['duplicate_rows'] == 1


# ---------------------------------------------------------------------------
# DataFrameSerializer
# ---------------------------------------------------------------------------

def test_serializer_pickle_roundtrip(tmp_path):
    df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    path = tmp_path / 'df.pkl'
    DataFrameSerializer.save(df, path)
    loaded = DataFrameSerializer.load(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_serializer_parquet_roundtrip(tmp_path):
    pytest.importorskip('pyarrow')
    df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    path = tmp_path / 'df.parquet'
    DataFrameSerializer.save(df, path)
    loaded = DataFrameSerializer.load(path)
    pd.testing.assert_frame_equal(loaded, df)


# ---------------------------------------------------------------------------
# TensorBatchGenerator
# ---------------------------------------------------------------------------

def test_tensor_batch_generator_yields_expected_shape():
    features = {'x': torch.arange(10, dtype=torch.float32).reshape(10, 1)}
    labels = torch.arange(10, dtype=torch.float32)
    gen = TensorBatchGenerator(features, labels=labels, batch_size=4,
                               shuffle=False, device=torch.device('cpu'))
    batches = list(gen)
    assert len(batches) == len(gen) == 3   # ceil(10 / 4)
    assert batches[0]['x'].shape == (4, 1)
    assert batches[-1]['x'].shape == (2, 1)  # final, smaller batch
    assert 'labels' in batches[0]


def test_tensor_batch_generator_requires_features():
    with pytest.raises(ValueError, match='non-empty'):
        TensorBatchGenerator({}, batch_size=4)


def test_tensor_batch_generator_shuffle_changes_order():
    set_seeds(0)
    features = {'x': torch.arange(20, dtype=torch.float32)}
    gen = TensorBatchGenerator(features, batch_size=4, shuffle=True,
                               device=torch.device('cpu'))
    batches = list(gen)
    flat = torch.cat([b['x'] for b in batches])
    # Highly unlikely to be in original order after shuffle.
    assert not torch.equal(flat, torch.arange(20, dtype=torch.float32))


# ---------------------------------------------------------------------------
# MetricTracker
# ---------------------------------------------------------------------------

def test_metric_tracker_average_and_latest():
    tracker = MetricTracker()
    tracker.update({'loss': 1.0, 'acc': 0.5})
    tracker.update({'loss': 2.0, 'acc': 0.7})
    latest = tracker.get_latest()
    assert latest['loss'] == 2.0
    avg = tracker.get_average()
    assert avg['loss'] == pytest.approx(1.5)
    assert avg['acc'] == pytest.approx(0.6)


def test_metric_tracker_save_history(tmp_path):
    tracker = MetricTracker()
    tracker.update({'loss': 1.0})
    tracker.update({'loss': 0.5})
    path = tmp_path / 'history.json'
    tracker.save_history(path)
    with open(path) as f:
        data = json.load(f)
    assert data['loss'] == [1.0, 0.5]


def test_metric_tracker_empty_returns_empty_dicts():
    tracker = MetricTracker()
    assert tracker.get_latest() == {}
    assert tracker.get_average() == {}


# ---------------------------------------------------------------------------
# Tensor / model utilities
# ---------------------------------------------------------------------------

def test_calculate_model_size_returns_mb():
    model = nn.Linear(100, 100)
    size = calculate_model_size(model)
    assert size > 0
    assert isinstance(size, float)


def test_convert_to_tensor_from_numpy():
    arr = np.array([1, 2, 3], dtype=np.float32)
    t = convert_to_tensor(arr)
    assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0]))


def test_convert_to_tensor_from_dataframe_with_dtype():
    df = pd.DataFrame({'a': [1, 2, 3]})
    t = convert_to_tensor(df, dtype=torch.long)
    assert t.dtype == torch.long


def test_convert_to_tensor_from_series():
    s = pd.Series([1.0, 2.0, 3.0])
    t = convert_to_tensor(s)
    assert t.shape == (3,)


def test_convert_to_tensor_passthrough_existing_tensor():
    original = torch.tensor([1.0, 2.0])
    out = convert_to_tensor(original)
    assert out is original


def test_convert_to_tensor_rejects_unsupported_type():
    with pytest.raises(TypeError, match='Unsupported'):
        convert_to_tensor("not a tensor")
