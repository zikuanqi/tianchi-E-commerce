"""Negative-sampling correctness tests."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.data_processing import DataProcessor


def _prepare(cfg):
    proc = DataProcessor(cfg)
    user_data, item_data = proc.load_data()
    proc.preprocess_data(user_data, item_data)
    return proc, user_data


def test_uniform_sampling_avoids_user_positives(config):
    cfg = copy.deepcopy(config)
    cfg['model']['training']['negative_sampling_strategy'] = 'uniform'
    cfg['model']['training']['negative_sampling_ratio'] = 5
    proc, user_data = _prepare(cfg)
    import pandas as pd
    target = pd.to_datetime(cfg['training']['train_end_date'])
    interactions = proc.sample_interactions(user_data, target)

    # Per user, no negative should coincide with one of that user's positives.
    for user_id, group in interactions.groupby('user_id'):
        positives = set(group.loc[group['label'] == 1, 'item_id'])
        negatives = set(group.loc[group['label'] == 0, 'item_id'])
        assert not (positives & negatives), \
            f'user {user_id}: negative collides with positive'


def test_popularity_sampling_favors_frequent_items(config):
    cfg = copy.deepcopy(config)
    cfg['model']['training']['negative_sampling_strategy'] = 'popularity'
    cfg['model']['training']['negative_sampling_alpha'] = 1.0
    cfg['model']['training']['negative_sampling_ratio'] = 10
    proc, user_data = _prepare(cfg)

    import pandas as pd
    target = pd.to_datetime(cfg['training']['train_end_date'])
    interactions = proc.sample_interactions(user_data, target)
    negatives = interactions.loc[interactions['label'] == 0, 'item_id']
    assert len(negatives) > 0

    # Frequent items should appear in negatives more often than the
    # all-items median frequency. Use a Spearman-style proxy: the top
    # 20% of items by global popularity should account for a
    # disproportionately large share of negatives.
    item_counts = user_data['item_id_encoded'].value_counts()
    top_quintile = set(item_counts.head(max(1, len(item_counts) // 5)).index.astype(int))
    neg_in_top = negatives.isin(top_quintile).mean()
    assert neg_in_top > 0.20, \
        f'popularity sampling underweights top-quintile items: {neg_in_top:.2%}'


def test_invalid_strategy_raises(config):
    cfg = copy.deepcopy(config)
    cfg['model']['training']['negative_sampling_strategy'] = 'gibberish'
    with pytest.raises(ValueError, match='negative_sampling_strategy'):
        DataProcessor(cfg)
