"""Shared fixtures: a synthetic Tianchi-shaped dataset on disk."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import yaml


def _make_synthetic_users(n_users: int = 50,
                          n_items: int = 100,
                          n_categories: int = 10,
                          seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp('2014-12-10 00')
    for u in range(1, n_users + 1):
        n_actions = int(rng.integers(5, 30))
        for _ in range(n_actions):
            item = int(rng.integers(1, n_items + 1))
            category = int(rng.integers(1, n_categories + 1))
            behavior = int(rng.choice([1, 2, 3, 4], p=[0.6, 0.2, 0.1, 0.1]))
            hour_offset = int(rng.integers(0, 24 * 10))
            rows.append({
                'user_id': u,
                'item_id': item,
                'behavior_type': behavior,
                'user_geohash': '',
                'item_category': category,
                'time': base + pd.Timedelta(hours=hour_offset),
            })
    return pd.DataFrame(rows)


def _make_synthetic_items(user_data: pd.DataFrame) -> pd.DataFrame:
    """Match the official Tianchi P table schema:
    `item_id, item_geohash, item_category`. `load_data` then renames
    `item_category` → `category` — so we exercise that rename branch."""
    items = user_data[['item_id', 'item_category']].drop_duplicates()
    items['item_geohash'] = ''
    return items[['item_id', 'item_geohash', 'item_category']]


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Tuple[Path, Path]:
    user = _make_synthetic_users()
    item = _make_synthetic_items(user)
    user_csv = tmp_path / 'user.csv'
    item_csv = tmp_path / 'item.csv'
    user['time'] = user['time'].dt.strftime('%Y-%m-%d %H')
    user.to_csv(user_csv, index=False)
    item.to_csv(item_csv, index=False)
    return user_csv, item_csv


@pytest.fixture
def base_config() -> dict:
    """Minimal config sufficient to drive the pipeline end-to-end on CPU."""
    return {
        'system': {'seed': 42, 'num_workers': 0, 'pin_memory': False, 'prefetch_factor': 2},
        'data': {
            'paths': {
                'raw_user_data': '',  # filled per-test
                'raw_item_data': '',
                'processed_data_dir': '',
                'output_dir': '',
                'checkpoint_dir': '',
                'log_dir': '',
            },
            'features': {
                'categorical': ['user_id', 'item_id', 'category'],
                'sequence': {'max_length': 8, 'num_behaviors': 5, 'behavior_dim': 4},
                'user': {'categorical': ['user_id'], 'numerical': [], 'embedding_dim': 8},
                'item': {'categorical': ['item_id', 'category'], 'numerical': [],
                         'embedding_dim': 8},
            },
        },
        'device': {'accelerator': 'cpu', 'devices': 1, 'strategy': 'auto', 'precision': 32},
        'model': {
            'architecture': {
                'embedding_dim': 8,
                'hidden_dims': [16, 8],
                'dropout': 0.0,
                'num_attention_heads': 2,
                'num_transformer_layers': 1,
                'transformer_ff_dim': 16,
            },
            'training': {
                'batch_size': 16,
                'num_epochs': 1,
                'learning_rate': 0.001,
                'weight_decay': 0.0,
                'negative_sampling_ratio': 2,
                'early_stopping': {'patience': 1, 'min_delta': 0.0},
                'optimization': {'accumulate_grad_batches': 1, 'gradient_clip_val': 1.0},
                'scheduler': {'type': 'cosine', 'T_max': 1, 'eta_min': 1e-6},
            },
        },
        'training': {
            'train_end_date': '2014-12-15',
            'pred_date': '2014-12-16',
            'validation_days': 1,
            'top_k': 5,
        },
        'logging': {'level': 'WARNING'},
    }


@pytest.fixture
def config(base_config: dict, synthetic_dataset, tmp_path: Path) -> dict:
    cfg = copy.deepcopy(base_config)
    user_csv, item_csv = synthetic_dataset
    cfg['data']['paths']['raw_user_data'] = str(user_csv)
    cfg['data']['paths']['raw_item_data'] = str(item_csv)
    cfg['data']['paths']['processed_data_dir'] = str(tmp_path / 'processed')
    cfg['data']['paths']['output_dir'] = str(tmp_path / 'output')
    cfg['data']['paths']['checkpoint_dir'] = str(tmp_path / 'ckpt')
    cfg['data']['paths']['log_dir'] = str(tmp_path / 'logs')
    return cfg
