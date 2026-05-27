"""Tests for the optional in-batch InfoNCE auxiliary loss."""

from __future__ import annotations

import copy

import torch

from src.data.dataloader import build_dataloader
from src.data_processing import DataProcessor
from src.models.deep_recommender import DeepRecommender
from src.trainer import ModelTrainer


def test_infonce_off_by_default(config):
    model = DeepRecommender(
        config,
        vocab_sizes={'user_id': 10, 'item_id': 10, 'category': 5},
    )
    assert not model.use_in_batch_negatives


def test_infonce_loss_is_finite_and_positive(config):
    cfg = copy.deepcopy(config)
    cfg['model']['training']['use_in_batch_negatives'] = True
    cfg['model']['training']['infonce_weight'] = 1.0

    proc = DataProcessor(cfg)
    train_ds, _ = proc.prepare_train_val_data()
    model = DeepRecommender(
        cfg,
        vocab_sizes=proc.get_categorical_dims(),
        user_feature_dim=proc.get_user_feature_dim(),
        item_feature_dim=proc.get_item_feature_dim(),
    )
    loader = build_dataloader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    user_repr = model._compute_user_repr(batch)
    item_repr = model._compute_item_repr(batch)
    loss = model._infonce_loss(user_repr, item_repr, batch)

    # The synthetic dataset should produce >= 2 positives in a batch of 32.
    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_training_with_infonce_runs(config):
    cfg = copy.deepcopy(config)
    cfg['model']['training']['use_in_batch_negatives'] = True
    cfg['model']['training']['infonce_weight'] = 0.5

    trainer = ModelTrainer(cfg)
    metrics = trainer.train()
    assert 'train_loss' in metrics or 'train_loss_epoch' in metrics
