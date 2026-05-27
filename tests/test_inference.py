"""Leak-free inference candidate-set tests."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from src.data_processing import DataProcessor


def test_inference_candidates_dont_use_post_cutoff_data(config):
    """Candidates must come only from history ≤ cutoff. Items the user
    first interacts with on the prediction day itself must not appear."""
    proc = DataProcessor(config)
    user_data, item_data = proc.load_data()
    user_data, item_data = proc.preprocess_data(user_data, item_data)

    cutoff = pd.to_datetime(config['training']['train_end_date'])

    # Inject a post-cutoff sentinel interaction. If candidate building
    # leaks, this (user, item) pair will show up in the output.
    post_cutoff_user = int(user_data['user_id_encoded'].iloc[0])
    post_cutoff_item = int(user_data['item_id_encoded'].iloc[-1])
    sentinel = user_data.iloc[[0]].copy()
    sentinel['user_id_encoded'] = post_cutoff_user
    sentinel['item_id_encoded'] = post_cutoff_item
    sentinel['time'] = cutoff + pd.Timedelta(days=1, hours=5)
    leaked = pd.concat([user_data, sentinel], ignore_index=True)

    candidates = proc.build_inference_candidates(leaked, cutoff=cutoff, history_days=7)
    sentinel_pair = (post_cutoff_user, post_cutoff_item)
    candidate_pairs = set(
        (int(u), int(i)) for u, i in zip(candidates['user_id'], candidates['item_id'])
    )
    # The pair MIGHT appear if the same user interacted with that item
    # in earlier history too — but the leaked row alone must not put it
    # there. To check, ensure post-cutoff rows are excluded entirely:
    # filter `leaked` to the pre-cutoff window and rebuild — the pair
    # set should be identical to what we already got.
    pre_cutoff = leaked[leaked['time'] <= cutoff]
    expected = proc.build_inference_candidates(pre_cutoff, cutoff=cutoff, history_days=7)
    expected_pairs = set(
        (int(u), int(i)) for u, i in zip(expected['user_id'], expected['item_id'])
    )
    assert candidate_pairs == expected_pairs


def test_inference_candidates_in_item_subset_P(config):
    proc = DataProcessor(config)
    user_data, item_data = proc.load_data()
    user_data, item_data = proc.preprocess_data(user_data, item_data)
    cutoff = pd.to_datetime(config['training']['train_end_date'])

    items_in_p = set(int(x) for x in proc._all_item_ids)
    candidates = proc.build_inference_candidates(user_data, cutoff=cutoff, history_days=14)
    candidate_items = set(int(x) for x in candidates['item_id'])
    assert candidate_items.issubset(items_in_p), \
        f'{len(candidate_items - items_in_p)} candidates outside P'


def test_inference_candidates_have_no_labels(config):
    proc = DataProcessor(config)
    ds, candidates = proc.prepare_inference_data()
    # All placeholder labels are zero — there's no peeking at pred_date.
    assert (candidates['label'] == 0).all()
    # Dataset emits a label tensor regardless (model just ignores it).
    sample = ds[0]
    assert sample['label'].item() == 0.0


def test_actual_purchases_returns_string_pairs(config):
    cfg = copy.deepcopy(config)
    proc = DataProcessor(cfg)
    # `actual_purchases_on` internally loads and preprocesses, so it's
    # safe to call without an explicit warm-up.
    purchases = proc.actual_purchases_on(cfg['training']['pred_date'])
    assert {'user_id', 'item_id'} <= set(purchases.columns)
    if not purchases.empty:
        assert isinstance(purchases['user_id'].iloc[0], str)
        assert isinstance(purchases['item_id'].iloc[0], str)
    # No dupes.
    assert not purchases.duplicated().any()
