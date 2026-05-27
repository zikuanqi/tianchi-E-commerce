"""Trainer edge-case tests: bad config, device fallback, checkpoint flow."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.trainer import ModelTrainer


def test_bad_config_type_raises():
    with pytest.raises(TypeError, match='path or a dict'):
        ModelTrainer(12345)


def test_load_checkpoint_missing_file_raises(config):
    trainer = ModelTrainer(config)
    with pytest.raises(FileNotFoundError):
        trainer.load_checkpoint('/no/such/file.ckpt')


def test_predict_before_model_raises(config):
    trainer = ModelTrainer(config)
    train_ds, _ = trainer.prepare_data()
    with pytest.raises(RuntimeError, match='predict called before'):
        trainer.predict(train_ds)


def test_build_model_before_prepare_raises(config):
    """Vocab sizes only appear after preprocess_data has run."""
    trainer = ModelTrainer(config)
    with pytest.raises(RuntimeError, match='No vocab sizes'):
        trainer.build_model()


def test_gpu_request_downgrades_to_cpu_when_no_cuda(config, monkeypatch):
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    cfg = copy.deepcopy(config)
    cfg['device']['accelerator'] = 'gpu'
    cfg['device']['precision'] = 16
    trainer = ModelTrainer(cfg)
    assert trainer.config['device']['accelerator'] == 'cpu'
    # Precision must be reset to 32 — 16-bit on CPU is not portable.
    assert trainer.config['device']['precision'] == 32


def test_auto_accelerator_picks_cpu_when_no_cuda(config, monkeypatch):
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    cfg = copy.deepcopy(config)
    cfg['device']['accelerator'] = 'auto'
    trainer = ModelTrainer(cfg)
    assert trainer.config['device']['accelerator'] == 'cpu'


def test_load_checkpoint_reconstructs_model(config, tmp_path):
    """A round-trip: train → predict (warm) → load_checkpoint → predict
    again, this time using only the on-disk checkpoint as ground truth."""
    trainer = ModelTrainer(config)
    trainer.train()

    # Trainer's ModelCheckpoint(save_last=True) writes last.ckpt.
    ckpt_path = Path(config['data']['paths']['checkpoint_dir']) / 'last.ckpt'
    assert ckpt_path.exists()

    fresh = ModelTrainer(config)
    fresh.load_checkpoint(ckpt_path)
    assert fresh.model is not None

    test_ds = fresh.data_processor.prepare_test_data()
    scores = fresh.predict(test_ds)
    assert scores.shape[0] == len(test_ds)
    assert (scores >= 0).all() and (scores <= 1).all()
