"""Submission-format compliance tests against the Tianchi spec."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_processing import DataProcessor


def test_submission_columns_are_strings_and_deduped(config):
    proc = DataProcessor(config)
    train_ds, _ = proc.prepare_train_val_data()

    # Synthesize predictions for the training set, then ask
    # create_submission to format them.
    interactions = pd.DataFrame({
        'user_id': train_ds.user_ids.numpy(),
        'item_id': train_ds.item_ids.numpy(),
    })
    # Force several duplicate (user, item) pairs so we can verify dedup.
    interactions = pd.concat([interactions, interactions.head(5)], ignore_index=True)
    scores = np.linspace(0.0, 1.0, num=len(interactions), dtype=np.float32)

    submission = proc.create_submission(scores, interactions)

    # Columns: only user_id, item_id, both string-typed. Accept either
    # the legacy `object` dtype or pandas's newer StringDtype.
    assert list(submission.columns) == ['user_id', 'item_id']
    assert pd.api.types.is_string_dtype(submission['user_id'])
    assert pd.api.types.is_string_dtype(submission['item_id'])
    assert all(isinstance(v, str) for v in submission['user_id'].head())
    assert all(isinstance(v, str) for v in submission['item_id'].head())

    # No duplicate (user, item) pairs.
    assert not submission.duplicated().any()


def test_submission_respects_top_k(config):
    cfg = dict(config)
    cfg['training'] = dict(cfg['training'], top_k=3)
    proc = DataProcessor(cfg)
    train_ds, _ = proc.prepare_train_val_data()

    interactions = pd.DataFrame({
        'user_id': train_ds.user_ids.numpy(),
        'item_id': train_ds.item_ids.numpy(),
    })
    scores = np.random.default_rng(0).random(len(interactions)).astype(np.float32)
    submission = proc.create_submission(scores, interactions)

    per_user = submission.groupby('user_id').size()
    assert (per_user <= 3).all(), f'top_k=3 exceeded: {per_user.max()}'
