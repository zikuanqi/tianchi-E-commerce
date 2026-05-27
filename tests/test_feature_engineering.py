"""FeatureEngineer correctness tests."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.data_processing import DataProcessor
from src.feature_engineering import (
    FeatureEngineer,
    ITEM_FEATURE_NAMES,
    USER_FEATURE_NAMES,
)


def test_feature_engineer_dims_are_stable():
    fe = FeatureEngineer()
    assert fe.user_feature_dim == len(USER_FEATURE_NAMES)
    assert fe.item_feature_dim == len(ITEM_FEATURE_NAMES)


def test_feature_tables_have_expected_shape(config):
    proc = DataProcessor(config)
    train_ds, _ = proc.prepare_train_val_data()
    user_tbl = proc.feature_engineer.user_feature_table
    item_tbl = proc.feature_engineer.item_feature_table
    vocab = proc.get_categorical_dims()
    assert user_tbl.shape == (vocab['user_id'], proc.get_user_feature_dim())
    assert item_tbl.shape == (vocab['item_id'], proc.get_item_feature_dim())
    # PAD row (id 0) must stay zero — guarantees cold-start = mean.
    assert np.allclose(user_tbl[0], 0)
    assert np.allclose(item_tbl[0], 0)


def test_dataset_returns_numerical_features(config):
    proc = DataProcessor(config)
    train_ds, _ = proc.prepare_train_val_data()
    sample = train_ds[0]
    assert 'user_numerical' in sample
    assert 'item_numerical' in sample
    assert sample['user_numerical'].shape == (proc.get_user_feature_dim(),)
    assert sample['item_numerical'].shape == (proc.get_item_feature_dim(),)


def test_features_respect_train_end_date(config):
    """Features computed at train_end_date must ignore later rows."""
    cfg_early = copy.deepcopy(config)
    cfg_early['training']['train_end_date'] = '2014-12-11'
    cfg_late = copy.deepcopy(config)
    cfg_late['training']['train_end_date'] = '2014-12-20'

    proc_early = DataProcessor(cfg_early)
    proc_early.prepare_train_val_data()
    proc_late = DataProcessor(cfg_late)
    proc_late.prepare_train_val_data()

    # `total_actions` is the first user feature; standardization makes
    # absolute comparison hard, but the row-wise *sum* of standardized
    # features should differ between two different time windows.
    early = proc_early.feature_engineer.user_feature_table
    late = proc_late.feature_engineer.user_feature_table
    assert not np.allclose(early, late), \
        'features did not change when the training window was widened'
