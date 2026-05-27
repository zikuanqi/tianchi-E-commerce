"""Targeted edge-case tests to close remaining coverage gaps."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data.dataloader import RecommendationDataLoader
from src.data.dataset import RecommendationDataset
from src.data_processing import DataProcessor, PadLabelEncoder
from src.feature_engineering import FeatureEngineer
from src.trainer import ModelTrainer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_dataset_rejects_interactions_missing_columns():
    bad = pd.DataFrame({'user_id': [1], 'item_id': [2]})  # missing label, category_id
    with pytest.raises(ValueError, match='missing columns'):
        RecommendationDataset(bad, sequences={}, max_seq_length=5)


def test_dataset_returns_empty_seq_for_missing_user():
    interactions = pd.DataFrame({
        'user_id': [99],
        'item_id': [1],
        'category_id': [1],
        'label': [0.0],
    })
    # user 99 not in sequences → fallback to the all-PAD sentinel.
    ds = RecommendationDataset(interactions, sequences={}, max_seq_length=4)
    sample = ds[0]
    assert sample['sequence']['length'].item() == 0
    assert sample['sequence']['mask'].all()


def test_dataset_without_numerical_features_omits_keys():
    interactions = pd.DataFrame({
        'user_id': [1], 'item_id': [2], 'category_id': [3], 'label': [1.0],
    })
    ds = RecommendationDataset(interactions, sequences={}, max_seq_length=2,
                               user_features=None, item_features=None)
    sample = ds[0]
    assert 'user_numerical' not in sample
    assert 'item_numerical' not in sample
    assert ds.user_feature_dim == 0
    assert ds.item_feature_dim == 0


# ---------------------------------------------------------------------------
# DataLoader class wrapper
# ---------------------------------------------------------------------------

def test_recommendation_dataloader_class_wrapper():
    interactions = pd.DataFrame({
        'user_id': list(range(4)),
        'item_id': list(range(4)),
        'category_id': [1] * 4,
        'label': [1.0, 0.0, 1.0, 0.0],
    })
    ds = RecommendationDataset(interactions, sequences={}, max_seq_length=2)
    loader = RecommendationDataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    assert len(loader) == 2
    batches = list(loader)
    assert batches[0]['user_id'].shape[0] == 2


# ---------------------------------------------------------------------------
# PadLabelEncoder
# ---------------------------------------------------------------------------

def test_pad_label_encoder_pad_is_index_zero():
    enc = PadLabelEncoder()
    enc.fit(['a', 'b', 'c'])
    # Encoded ids should all be ≥ 1; class 0 is the PAD sentinel.
    encoded = enc.transform(['a', 'b', 'c'])
    assert (encoded >= 1).all()
    assert '<pad>' in enc.classes_
    assert enc.classes_[0] == '<pad>'


def test_pad_label_encoder_unknown_maps_to_pad():
    enc = PadLabelEncoder()
    enc.fit(['x', 'y'])
    encoded = enc.transform(['x', 'unseen', 'y'])
    # 'unseen' should map to 0 (PAD).
    assert encoded[1] == 0


def test_pad_label_encoder_raises_before_fit():
    enc = PadLabelEncoder()
    with pytest.raises(RuntimeError, match='fit'):
        enc.transform(['a'])
    with pytest.raises(RuntimeError, match='fit'):
        enc.inverse_transform([0])


def test_pad_label_encoder_inverse_roundtrip():
    enc = PadLabelEncoder()
    enc.fit(['x', 'y', 'z'])
    encoded = enc.transform(['z', 'x'])
    decoded = enc.inverse_transform(encoded)
    assert list(decoded) == ['z', 'x']


def test_pad_label_encoder_inverse_handles_out_of_range():
    enc = PadLabelEncoder()
    enc.fit(['a'])
    # Out-of-range indices fall back to PAD via the `where` guard.
    out = enc.inverse_transform([99, 0, 1])
    assert out[0] == '<pad>'   # 99 is out of range
    assert out[1] == '<pad>'   # explicit PAD
    assert out[2] == 'a'


# ---------------------------------------------------------------------------
# DataProcessor edge cases
# ---------------------------------------------------------------------------

def test_dataprocessor_rejects_bad_config_type():
    with pytest.raises(TypeError, match='path or a dict'):
        DataProcessor(12345)


def test_dataprocessor_loads_config_from_path(tmp_path, config):
    """The str/Path branch of __init__."""
    cfg_path = tmp_path / 'config.yaml'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)
    proc = DataProcessor(str(cfg_path))
    assert proc.config['training']['train_end_date'] == config['training']['train_end_date']


def test_dataprocessor_rejects_invalid_neg_strategy(config):
    cfg = copy.deepcopy(config)
    cfg['model']['training']['negative_sampling_strategy'] = 'gibberish'
    with pytest.raises(ValueError, match='negative_sampling_strategy'):
        DataProcessor(cfg)


def test_build_inference_candidates_without_preprocess_raises(config):
    proc = DataProcessor(config)
    with pytest.raises(RuntimeError, match='preprocess_data must be called'):
        proc.build_inference_candidates(
            pd.DataFrame({'time': [], 'user_id_encoded': [], 'item_id_encoded': []}),
            cutoff=pd.Timestamp('2014-12-18'),
            history_days=7,
        )


def test_sample_interactions_falls_back_when_no_purchases(config):
    """When a target date has zero behavior_type==4 actions, the
    sampler falls back to all interactions as labels (with a warning)."""
    cfg = copy.deepcopy(config)
    proc = DataProcessor(cfg)
    user_data, item_data = proc.load_data()
    user_data, item_data = proc.preprocess_data(user_data, item_data)

    # Force a target date where no purchases happened — use the very
    # first hour of the dataset, where the seeded synthetic data is
    # unlikely to contain a behavior_type==4 row by chance.
    no_purchase_day = pd.to_datetime(user_data['time'].min().date())
    # Filter all data on that day to non-purchases, then call sample.
    drained = user_data.copy()
    drained.loc[drained['time'].dt.date == no_purchase_day.date(), 'behavior_type'] = 1
    proc._all_item_ids = np.asarray(item_data['item_id_encoded'].unique())
    proc._item_to_category = dict(zip(item_data['item_id_encoded'].astype(int),
                                      item_data['category_encoded'].astype(int)))
    # Smoothed sampling probs — needed for the popularity branch.
    item_counts = (drained['item_id_encoded']
                   .value_counts()
                   .reindex(proc._all_item_ids, fill_value=0)
                   .to_numpy(dtype=np.float64))
    smoothed = (item_counts + 1.0) ** proc.neg_alpha
    proc._item_sampling_probs = smoothed / smoothed.sum()

    interactions = proc.sample_interactions(drained, no_purchase_day)
    assert len(interactions) > 0    # fallback produced something


# ---------------------------------------------------------------------------
# FeatureEngineer edge cases
# ---------------------------------------------------------------------------

def test_feature_engineer_accepts_no_config():
    fe = FeatureEngineer()  # config=None → empty dict
    assert fe.config == {}


def test_feature_engineer_rejects_bad_config_type():
    with pytest.raises(TypeError, match='path, dict, or None'):
        FeatureEngineer(12345)


def test_feature_engineer_loads_config_from_path(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    yaml.safe_dump({'foo': 'bar'}, open(cfg_path, 'w'))
    fe = FeatureEngineer(str(cfg_path))
    assert fe.config['foo'] == 'bar'


def test_feature_engineer_fit_rejects_zero_vocab():
    fe = FeatureEngineer()
    df = pd.DataFrame({'time': pd.to_datetime(['2014-12-18']),
                       'user_id_encoded': [1], 'item_id_encoded': [1],
                       'behavior_type': [1]})
    with pytest.raises(ValueError, match='vocab_sizes'):
        fe.fit(df, vocab_sizes={'user_id': 0, 'item_id': 5})


def test_feature_engineer_handles_empty_dataframe():
    fe = FeatureEngineer()
    empty = pd.DataFrame({'time': pd.Series([], dtype='datetime64[ns]'),
                          'user_id_encoded': pd.Series([], dtype='int64'),
                          'item_id_encoded': pd.Series([], dtype='int64'),
                          'category_encoded': pd.Series([], dtype='int64'),
                          'behavior_type': pd.Series([], dtype='int64'),
                          'hour': pd.Series([], dtype='int64'),
                          'weekday': pd.Series([], dtype='int64')})
    fe.fit(empty, vocab_sizes={'user_id': 3, 'item_id': 3})
    # All-zero feature tables of the expected shape.
    assert fe.user_feature_table.shape == (3, fe.user_feature_dim)
    assert fe.item_feature_table.shape == (3, fe.item_feature_dim)
    assert np.allclose(fe.user_feature_table, 0)
    assert np.allclose(fe.item_feature_table, 0)


def test_feature_engineer_tensor_accessors_before_fit():
    fe = FeatureEngineer()
    with pytest.raises(RuntimeError, match='fit'):
        fe.user_features_tensor()
    with pytest.raises(RuntimeError, match='fit'):
        fe.item_features_tensor()


# ---------------------------------------------------------------------------
# Trainer: TensorBoard → CSVLogger fallback
# ---------------------------------------------------------------------------

def test_trainer_falls_back_to_csvlogger_when_tensorboard_unavailable(config, monkeypatch):
    """Simulate `_HAS_TENSORBOARD = False` and verify training still works."""
    import src.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, '_HAS_TENSORBOARD', False)

    trainer = ModelTrainer(config)
    metrics = trainer.train()
    assert 'train_loss' in metrics or 'train_loss_epoch' in metrics


def test_trainer_loads_config_from_path(tmp_path, config):
    cfg_path = tmp_path / 'config.yaml'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f)
    trainer = ModelTrainer(str(cfg_path))
    assert trainer.config['training']['train_end_date'] == config['training']['train_end_date']


# ---------------------------------------------------------------------------
# Model: validation AUROC single-class branch
# ---------------------------------------------------------------------------

def test_validation_step_handles_single_class_auroc(config):
    """If a val batch happens to contain only one class, AUROC.compute()
    raises — the model is supposed to swallow that exception and continue."""
    from src.models.deep_recommender import DeepRecommender
    model = DeepRecommender(config, vocab_sizes={'user_id': 5, 'item_id': 5, 'category': 3})
    # Update val_auroc with a single-class label set, then compute() inside
    # on_validation_epoch_end. We mimic that path directly.
    probs = torch.tensor([0.7, 0.8, 0.9])
    labels = torch.tensor([1, 1, 1])
    model.val_auroc.update(probs, labels)
    # Should not raise.
    model.on_validation_epoch_end()


# ---------------------------------------------------------------------------
# Tail-end coverage gaps in data_processing
# ---------------------------------------------------------------------------

def test_create_submission_length_mismatch_raises(config):
    """`predictions` length must align with `interactions`."""
    proc = DataProcessor(config)
    user_data, item_data = proc.load_data()
    proc.preprocess_data(user_data, item_data)
    interactions = pd.DataFrame({'user_id': [1, 2], 'item_id': [3, 4]})
    with pytest.raises(ValueError, match='same length'):
        proc.create_submission(np.array([0.1]), interactions)


def test_calculate_sequence_stats_returns_dict(config):
    proc = DataProcessor(config)
    user_data, _ = proc.load_data()
    stats = proc.calculate_sequence_stats(user_data)
    assert set(stats.keys()) == {'avg_length', 'max_length', 'min_length'}
    assert stats['avg_length'] > 0


def test_prepare_inference_data_default_cutoff(config):
    """Default cutoff is `train_end_date`; default history is from config."""
    proc = DataProcessor(config)
    ds, candidates = proc.prepare_inference_data()
    assert len(candidates) > 0
    assert (candidates['label'] == 0).all()


def test_preprocess_skips_categorical_feature_absent_from_both(config, monkeypatch):
    """When a configured categorical is missing from both user and item
    data, preprocess_data should warn and continue rather than crash."""
    cfg = copy.deepcopy(config)
    cfg['data']['features']['categorical'] = ['user_id', 'item_id', 'ghost_column']
    proc = DataProcessor(cfg)
    user_data, item_data = proc.load_data()
    # Must not raise even though 'ghost_column' isn't present anywhere.
    proc.preprocess_data(user_data, item_data)


# ---------------------------------------------------------------------------
# Feature engineering: cover behavior_ratio backfill + missing-column branches
# ---------------------------------------------------------------------------

def test_feature_engineer_fills_missing_behavior_columns():
    """If only behaviors {1, 2} appear, ratios for {3, 4} are backfilled to 0."""
    from src.feature_engineering import FeatureEngineer
    df = pd.DataFrame({
        'time': pd.to_datetime(['2014-12-10', '2014-12-11', '2014-12-12']),
        'user_id_encoded': [1, 1, 2],
        'item_id_encoded': [1, 2, 1],
        'behavior_type': [1, 2, 1],   # only 1 and 2 — no 3 or 4
        'hour': [10, 11, 12],
        'weekday': [2, 3, 4],
        'category_encoded': [1, 1, 1],
    })
    fe = FeatureEngineer()
    fe.fit(df, vocab_sizes={'user_id': 5, 'item_id': 5})
    # Should have no NaN values — the missing-behavior backfill kicked in.
    assert not np.isnan(fe.user_feature_table).any()
    assert not np.isnan(fe.item_feature_table).any()


def test_feature_engineer_runs_without_category_or_weekday_columns():
    """Both `category_encoded` and `weekday` may be absent in test data —
    each has an else branch that fills zeros."""
    from src.feature_engineering import FeatureEngineer
    df = pd.DataFrame({
        'time': pd.to_datetime(['2014-12-10', '2014-12-11']),
        'user_id_encoded': [1, 2],
        'item_id_encoded': [1, 1],
        'behavior_type': [1, 4],
        'hour': [10, 11],
    })
    fe = FeatureEngineer()
    fe.fit(df, vocab_sizes={'user_id': 5, 'item_id': 5})
    assert fe.user_feature_table.shape[0] == 5
