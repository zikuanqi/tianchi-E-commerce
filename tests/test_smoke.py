"""End-to-end smoke tests against a tiny synthetic dataset.

These tests assert that the pipeline *runs* and produces tensors of the
right shape, not that the model has any predictive accuracy.
"""

from __future__ import annotations

import torch

from src.data.dataloader import build_dataloader
from src.data_processing import DataProcessor
from src.models.deep_recommender import DeepRecommender
from src.trainer import ModelTrainer


def test_data_processor_builds_datasets(config):
    proc = DataProcessor(config)
    train_ds, val_ds = proc.prepare_train_val_data()
    assert len(train_ds) > 0
    assert len(val_ds) >= 0  # may be zero on tiny synthetic data
    vocab = proc.get_categorical_dims()
    assert 'user_id' in vocab and vocab['user_id'] > 1
    assert 'item_id' in vocab and vocab['item_id'] > 1


def test_model_forward_pass(config):
    proc = DataProcessor(config)
    train_ds, _ = proc.prepare_train_val_data()
    model = DeepRecommender(config, vocab_sizes=proc.get_categorical_dims())
    loader = build_dataloader(train_ds, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    logits = model(batch)
    assert logits.shape == (batch['label'].shape[0],)
    assert torch.isfinite(logits).all()


def test_training_step_runs_one_epoch(config):
    trainer = ModelTrainer(config)
    metrics = trainer.train()
    assert 'train_loss_epoch' in metrics or 'train_loss' in metrics
    # After fit, a checkpoint should have been written.
    from pathlib import Path
    ckpt_dir = Path(config['data']['paths']['checkpoint_dir'])
    assert any(ckpt_dir.glob('*.ckpt'))


def test_predict_runs_after_train(config):
    trainer = ModelTrainer(config)
    trainer.train()
    test_ds = trainer.data_processor.prepare_test_data()
    scores = trainer.predict(test_ds)
    assert scores.shape[0] == len(test_ds)
    assert (scores >= 0).all() and (scores <= 1).all()  # sigmoid output
