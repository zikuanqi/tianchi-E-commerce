"""Per-user and per-item numerical feature tables.

This module computes dense numerical features keyed by *encoded* user_id
and item_id (i.e. the same int ids the model's embedding tables use).
Features are standardized so they're directly usable as model inputs.

The output is a pair of numpy tables that can be indexed directly:

    user_feature_table[encoded_user_id]   -> [user_feature_dim]
    item_feature_table[encoded_item_id]   -> [item_feature_dim]

Row 0 (the PAD sentinel) is always zero, so a cold-start lookup sees the
post-standardization mean.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


USER_FEATURE_NAMES: Sequence[str] = (
    'total_actions',
    'unique_items',
    'unique_categories',
    'behavior_1_ratio',
    'behavior_2_ratio',
    'behavior_3_ratio',
    'behavior_4_ratio',
    'avg_hour',
    'weekend_ratio',
    'action_days',
    'avg_time_diff_hours',
)

ITEM_FEATURE_NAMES: Sequence[str] = (
    'total_actions',
    'unique_users',
    'behavior_1_ratio',
    'behavior_2_ratio',
    'behavior_3_ratio',
    'behavior_4_ratio',
    'user_diversity',
)


def _behavior_ratio_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """For each grouping key, return ratios of behaviors 1..4 (one column each)."""
    pivot = (df.assign(_n=1)
               .pivot_table(index=key, columns='behavior_type',
                            values='_n', aggfunc='sum', fill_value=0))
    totals = pivot.sum(axis=1).replace(0, np.nan)
    ratios = pivot.div(totals, axis=0).fillna(0.0)
    for b in (1, 2, 3, 4):
        if b not in ratios.columns:
            ratios[b] = 0.0
    return ratios[[1, 2, 3, 4]]


class FeatureEngineer:
    """Compute, standardize, and serve per-id numerical feature tables."""

    def __init__(self, config: Union[str, Path, dict, None] = None):
        if isinstance(config, (str, Path)):
            with open(config, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        elif config is None:
            self.config = {}
        else:
            raise TypeError('config must be a path, dict, or None')

        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        self.user_feature_table: Optional[np.ndarray] = None  # [n_users, D_u]
        self.item_feature_table: Optional[np.ndarray] = None  # [n_items, D_i]

    # ------------------------------------------------------------------
    # Public dims (constants — kept as properties so callers don't depend
    # on whether `fit` has run yet)
    # ------------------------------------------------------------------
    @property
    def user_feature_dim(self) -> int:
        return len(USER_FEATURE_NAMES)

    @property
    def item_feature_dim(self) -> int:
        return len(ITEM_FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self,
            user_data: pd.DataFrame,
            vocab_sizes: Dict[str, int],
            end_date: Optional[pd.Timestamp] = None) -> 'FeatureEngineer':
        """Compute and standardize feature tables from `user_data`.

        `vocab_sizes` is the dict returned by
        :meth:`DataProcessor.get_categorical_dims`. It pins the table
        sizes so unseen ids at inference time get a zero feature row.

        `end_date` (optional) restricts the window used to compute
        training features — pass it as the train cut-off so we never
        leak validation/test rows into the standardization fit.
        """
        df = user_data
        if end_date is not None:
            df = df[df['time'] <= end_date]

        n_users = int(vocab_sizes.get('user_id', 0))
        n_items = int(vocab_sizes.get('item_id', 0))
        if n_users == 0 or n_items == 0:
            raise ValueError(f'vocab_sizes must include user_id and item_id; got {vocab_sizes}')

        self.user_feature_table = self._fit_user_features(df, n_users)
        self.item_feature_table = self._fit_item_features(df, n_items)
        logger.info('FeatureEngineer.fit: user table %s, item table %s',
                    self.user_feature_table.shape, self.item_feature_table.shape)
        return self

    # ------------------------------------------------------------------
    # User-side feature computation
    # ------------------------------------------------------------------
    def _fit_user_features(self, df: pd.DataFrame, n_users: int) -> np.ndarray:
        out = np.zeros((n_users, self.user_feature_dim), dtype=np.float32)
        if df.empty:
            return out

        grouped = df.groupby('user_id_encoded', sort=False)
        total = grouped.size().rename('total_actions')
        unique_items = grouped['item_id_encoded'].nunique().rename('unique_items')
        if 'category_encoded' in df.columns:
            unique_cats = grouped['category_encoded'].nunique().rename('unique_categories')
        else:
            unique_cats = pd.Series(0, index=total.index, name='unique_categories')

        beh_ratios = _behavior_ratio_table(df, 'user_id_encoded')
        avg_hour = (grouped['hour'].mean().rename('avg_hour')
                    if 'hour' in df.columns
                    else pd.Series(0.0, index=total.index, name='avg_hour'))

        if 'weekday' in df.columns:
            df_local = df.assign(_wknd=df['weekday'].isin([5, 6]).astype(float))
            weekend_ratio = df_local.groupby('user_id_encoded')['_wknd'].mean().rename('weekend_ratio')
        else:
            weekend_ratio = pd.Series(0.0, index=total.index, name='weekend_ratio')

        action_days = (df.assign(_d=df['time'].dt.date)
                         .groupby('user_id_encoded')['_d']
                         .nunique()
                         .rename('action_days'))

        # Average inter-action gap in hours, per user.
        sorted_df = df.sort_values(['user_id_encoded', 'time'])
        gaps = sorted_df.groupby('user_id_encoded')['time'].diff().dt.total_seconds() / 3600.0
        avg_diff = gaps.groupby(sorted_df['user_id_encoded']).mean().fillna(0.0).rename('avg_time_diff_hours')

        index = total.index.to_numpy()
        cols = np.column_stack([
            total.to_numpy(dtype=np.float32),
            unique_items.reindex(index).fillna(0).to_numpy(dtype=np.float32),
            unique_cats.reindex(index).fillna(0).to_numpy(dtype=np.float32),
            beh_ratios.reindex(index).fillna(0)[1].to_numpy(dtype=np.float32),
            beh_ratios.reindex(index).fillna(0)[2].to_numpy(dtype=np.float32),
            beh_ratios.reindex(index).fillna(0)[3].to_numpy(dtype=np.float32),
            beh_ratios.reindex(index).fillna(0)[4].to_numpy(dtype=np.float32),
            avg_hour.reindex(index).fillna(0).to_numpy(dtype=np.float32),
            weekend_ratio.reindex(index).fillna(0).to_numpy(dtype=np.float32),
            action_days.reindex(index).fillna(0).to_numpy(dtype=np.float32),
            avg_diff.reindex(index).fillna(0).to_numpy(dtype=np.float32),
        ])

        cols = self.user_scaler.fit_transform(cols).astype(np.float32)
        # Clamp the standardized values so a single outlier user can't
        # blow up gradients. ±5 σ is generous but bounded.
        np.clip(cols, -5.0, 5.0, out=cols)

        mask = (index >= 0) & (index < n_users)
        out[index[mask]] = cols[mask]
        return out

    # ------------------------------------------------------------------
    # Item-side feature computation
    # ------------------------------------------------------------------
    def _fit_item_features(self, df: pd.DataFrame, n_items: int) -> np.ndarray:
        out = np.zeros((n_items, self.item_feature_dim), dtype=np.float32)
        if df.empty:
            return out

        grouped = df.groupby('item_id_encoded', sort=False)
        total = grouped.size().rename('total_actions')
        unique_users = grouped['user_id_encoded'].nunique().rename('unique_users')
        beh_ratios = _behavior_ratio_table(df, 'item_id_encoded')
        user_diversity = (unique_users / total.replace(0, np.nan)).fillna(0.0).rename('user_diversity')

        index = total.index.to_numpy()
        cols = np.column_stack([
            total.to_numpy(dtype=np.float32),
            unique_users.reindex(index).fillna(0).to_numpy(dtype=np.float32),
            beh_ratios.reindex(index).fillna(0)[1].to_numpy(dtype=np.float32),
            beh_ratios.reindex(index).fillna(0)[2].to_numpy(dtype=np.float32),
            beh_ratios.reindex(index).fillna(0)[3].to_numpy(dtype=np.float32),
            beh_ratios.reindex(index).fillna(0)[4].to_numpy(dtype=np.float32),
            user_diversity.reindex(index).fillna(0).to_numpy(dtype=np.float32),
        ])

        cols = self.item_scaler.fit_transform(cols).astype(np.float32)
        np.clip(cols, -5.0, 5.0, out=cols)

        mask = (index >= 0) & (index < n_items)
        out[index[mask]] = cols[mask]
        return out

    # ------------------------------------------------------------------
    # Tensor accessors
    # ------------------------------------------------------------------
    def user_features_tensor(self) -> torch.Tensor:
        if self.user_feature_table is None:
            raise RuntimeError('FeatureEngineer.fit was not called')
        return torch.from_numpy(self.user_feature_table)

    def item_features_tensor(self) -> torch.Tensor:
        if self.item_feature_table is None:
            raise RuntimeError('FeatureEngineer.fit was not called')
        return torch.from_numpy(self.item_feature_table)


__all__ = [
    'FeatureEngineer',
    'USER_FEATURE_NAMES',
    'ITEM_FEATURE_NAMES',
]
