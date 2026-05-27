"""PyTorch Dataset over (user, item, label) interactions with per-user sequences."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RecommendationDataset(Dataset):
    """Pair-wise recommendation dataset.

    Each item is one (user, candidate-item, label) row plus the user's
    recent behavior sequence and optional dense numerical features for
    the user and the item.

    Parameters
    ----------
    interactions:
        DataFrame with columns ``user_id``, ``item_id``, ``category_id``,
        ``label``. All ids are already label-encoded ints.
    sequences:
        Mapping ``user_id -> {items, behaviors, mask, length}`` produced
        by :meth:`DataProcessor.build_user_sequences`.
    max_seq_length:
        Length of the (padded) per-user sequence — used to synthesize a
        zero sequence for users absent from ``sequences``.
    user_features:
        Optional ``[num_users, D_u]`` tensor of dense per-user features
        indexed by encoded ``user_id``. Pass ``None`` to disable
        numerical user features (the model then sees a zero vector).
    item_features:
        Same shape contract as ``user_features`` but for items.
    """

    def __init__(self,
                 interactions: pd.DataFrame,
                 sequences: Mapping[int, Dict[str, np.ndarray]],
                 max_seq_length: int,
                 user_features: Optional[torch.Tensor] = None,
                 item_features: Optional[torch.Tensor] = None):
        required = {'user_id', 'item_id', 'category_id', 'label'}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f'interactions missing columns: {missing}')

        # `.to_numpy(copy=True)` to make the underlying array writable —
        # otherwise PyTorch emits an "non-writable tensor" warning.
        self.user_ids = torch.as_tensor(interactions['user_id'].to_numpy(copy=True), dtype=torch.long)
        self.item_ids = torch.as_tensor(interactions['item_id'].to_numpy(copy=True), dtype=torch.long)
        self.category_ids = torch.as_tensor(interactions['category_id'].to_numpy(copy=True), dtype=torch.long)
        self.labels = torch.as_tensor(interactions['label'].to_numpy(copy=True), dtype=torch.float)

        self.sequences = sequences
        self.max_seq_length = int(max_seq_length)
        self._empty_seq = {
            'items': torch.zeros(self.max_seq_length, dtype=torch.long),
            'behaviors': torch.zeros(self.max_seq_length, dtype=torch.long),
            'mask': torch.ones(self.max_seq_length, dtype=torch.bool),
            'length': torch.tensor(0, dtype=torch.long),
        }

        # Numerical feature tables are shared across all rows; storing
        # them once here keeps __getitem__ O(1) and avoids per-worker
        # duplication.
        self.user_features = user_features.float() if user_features is not None else None
        self.item_features = item_features.float() if item_features is not None else None
        self.user_feature_dim = int(self.user_features.shape[1]) if self.user_features is not None else 0
        self.item_feature_dim = int(self.item_features.shape[1]) if self.item_features is not None else 0

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_id = int(self.user_ids[idx].item())
        seq_np = self.sequences.get(user_id)
        if seq_np is None:
            sequence = self._empty_seq
        else:
            sequence = {
                'items': torch.as_tensor(seq_np['items'], dtype=torch.long),
                'behaviors': torch.as_tensor(seq_np['behaviors'], dtype=torch.long),
                'mask': torch.as_tensor(seq_np['mask'], dtype=torch.bool),
                'length': torch.as_tensor(seq_np['length'], dtype=torch.long),
            }

        sample: Dict[str, torch.Tensor] = {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'category_id': self.category_ids[idx],
            'sequence': sequence,
            'label': self.labels[idx],
        }
        if self.user_features is not None:
            sample['user_numerical'] = self.user_features[user_id]
        if self.item_features is not None:
            item_id = int(self.item_ids[idx].item())
            sample['item_numerical'] = self.item_features[item_id]
        return sample
