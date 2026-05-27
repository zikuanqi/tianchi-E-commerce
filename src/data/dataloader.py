"""Thin convenience wrapper around `torch.utils.data.DataLoader`."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.data.dataset import RecommendationDataset


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of per-sample dicts into a batched dict.

    `default_collate` handles most of this for us but doesn't recurse into
    nested dicts the way we want, so we do it explicitly.
    """
    out = {
        'user_id': torch.stack([b['user_id'] for b in batch]),
        'item_id': torch.stack([b['item_id'] for b in batch]),
        'category_id': torch.stack([b['category_id'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'sequence': {
            'items': torch.stack([b['sequence']['items'] for b in batch]),
            'behaviors': torch.stack([b['sequence']['behaviors'] for b in batch]),
            'mask': torch.stack([b['sequence']['mask'] for b in batch]),
            'length': torch.stack([b['sequence']['length'] for b in batch]),
        },
    }
    if 'user_numerical' in batch[0]:
        out['user_numerical'] = torch.stack([b['user_numerical'] for b in batch])
    if 'item_numerical' in batch[0]:
        out['item_numerical'] = torch.stack([b['item_numerical'] for b in batch])
    return out


def build_dataloader(dataset: RecommendationDataset,
                     batch_size: int,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     pin_memory: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )


class RecommendationDataLoader:
    """Class-style wrapper kept for backward compatibility with src.__init__."""

    def __init__(self,
                 dataset: RecommendationDataset,
                 batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = False):
        self.dataloader = build_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)
