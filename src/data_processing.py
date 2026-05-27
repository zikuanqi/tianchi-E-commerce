"""Data processing pipeline for the Tianchi mobile-recommendation dataset.

The pipeline is intentionally linear:

    raw csv -> load_data -> preprocess_data -> build_sequences ->
    sample_interactions -> RecommendationDataset

`prepare_train_val_data` and `prepare_test_data` are the two public entry
points used by the CLI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from src.data.dataset import RecommendationDataset
from src.feature_engineering import FeatureEngineer
from src.utils import setup_logging

logger = logging.getLogger(__name__)


PURCHASE_BEHAVIOR = 4
PAD_ID = 0
PAD_TOKEN = '<pad>'


class PadLabelEncoder:
    """LabelEncoder variant that pins a PAD token at index 0.

    scikit-learn's :class:`LabelEncoder` sorts its classes
    alphabetically, so a sentinel like ``'<pad>'`` ends up at an index
    that depends on the rest of the vocabulary. We want PAD at a known
    index (0) so the model's ``nn.Embedding(padding_idx=0)`` and the
    FeatureEngineer's PAD-is-zero row stay in sync.

    Unknown values at transform time map to PAD (0) — this is the right
    behavior for inference on a user/item we never saw during training.
    """

    def __init__(self) -> None:
        self.classes_: Optional[np.ndarray] = None
        self._lookup: Dict[str, int] = {}

    def fit(self, values) -> 'PadLabelEncoder':
        unique = sorted({str(v) for v in values if str(v) != PAD_TOKEN})
        self.classes_ = np.array([PAD_TOKEN] + unique, dtype=object)
        self._lookup = {v: i for i, v in enumerate(self.classes_)}
        return self

    def transform(self, values) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError('PadLabelEncoder.fit was not called')
        return np.fromiter(
            (self._lookup.get(str(v), PAD_ID) for v in values),
            dtype=np.int64, count=len(values),
        )

    def inverse_transform(self, indices) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError('PadLabelEncoder.fit was not called')
        idx = np.asarray(indices, dtype=np.int64)
        in_range = (idx >= 0) & (idx < len(self.classes_))
        out = np.where(in_range, idx, 0)
        return self.classes_[out]


class DataProcessor:
    """End-to-end data loader/featurizer for the Tianchi dataset."""

    def __init__(self, config: Union[str, Path, dict]):
        if isinstance(config, (str, Path)):
            with open(config, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError('config must be a path or a dict')

        setup_logging(self.config.get('logging', {}))

        self.categorical_features: List[str] = self.config['data']['features']['categorical']
        self.encoders: Dict[str, PadLabelEncoder] = {f: PadLabelEncoder() for f in self.categorical_features}
        self.max_seq_length: int = self.config['data']['features']['sequence']['max_length']

        train_cfg = self.config['model']['training']
        self.neg_ratio: int = int(train_cfg.get('negative_sampling_ratio', 4))
        self.neg_strategy: str = str(train_cfg.get('negative_sampling_strategy', 'uniform')).lower()
        # word2vec-style smoothing exponent on item popularity. Setting
        # alpha=0 ⇒ uniform; alpha=1 ⇒ pure popularity; 0.75 is the value
        # that has held up well across recsys/word2vec/SGNS literature.
        self.neg_alpha: float = float(train_cfg.get('negative_sampling_alpha', 0.75))
        if self.neg_strategy not in {'uniform', 'popularity'}:
            raise ValueError(
                f"negative_sampling_strategy must be 'uniform' or 'popularity', got {self.neg_strategy!r}"
            )

        self.rng = np.random.default_rng(self.config.get('system', {}).get('seed', 42))

        # Numerical features (per-user / per-item dense tables) are
        # computed from training-window history only; fitting happens
        # inside preprocess_data.
        self.feature_engineer = FeatureEngineer(self.config)

        # Populated during preprocess_data
        self._all_item_ids: Optional[np.ndarray] = None
        self._item_to_category: Optional[Dict[int, int]] = None
        self._item_sampling_probs: Optional[np.ndarray] = None
        self._user_features_tensor: Optional[torch.Tensor] = None
        self._item_features_tensor: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Raw load + preprocessing
    # ------------------------------------------------------------------
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        paths = self.config['data']['paths']
        logger.info('Loading raw data from %s and %s', paths['raw_user_data'], paths['raw_item_data'])

        user_data = pd.read_csv(paths['raw_user_data'])
        item_data = pd.read_csv(paths['raw_item_data'])

        if 'item_category' in user_data.columns:
            user_data = user_data.rename(columns={'item_category': 'category'})
        if 'item_category' in item_data.columns:
            item_data = item_data.rename(columns={'item_category': 'category'})

        user_data['time'] = pd.to_datetime(user_data['time'], format='%Y-%m-%d %H')
        logger.info('Loaded %d user actions and %d item rows', len(user_data), len(item_data))
        return user_data, item_data

    def preprocess_data(self,
                        user_data: pd.DataFrame,
                        item_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Label-encode categorical columns; add time features.

        Encoders are fit on the union of values appearing in `user_data` and
        `item_data` so we never get an unseen-label error at inference time.
        """
        logger.info('Encoding categorical features %s', self.categorical_features)

        for feat in self.categorical_features:
            values: List[str] = []
            if feat in user_data.columns:
                values.append(user_data[feat].astype(str))
            if feat in item_data.columns:
                values.append(item_data[feat].astype(str))
            if not values:
                logger.warning('Categorical feature %s not present in user/item data', feat)
                continue
            combined = pd.concat(values, ignore_index=True)
            # PadLabelEncoder reserves index 0 for the PAD token; real
            # categorical values are encoded starting from 1.
            self.encoders[feat].fit(combined)

            if feat in user_data.columns:
                user_data[f'{feat}_encoded'] = self.encoders[feat].transform(user_data[feat].astype(str))
            if feat in item_data.columns:
                item_data[f'{feat}_encoded'] = self.encoders[feat].transform(item_data[feat].astype(str))

        user_data['hour'] = user_data['time'].dt.hour
        user_data['day'] = user_data['time'].dt.day
        user_data['weekday'] = user_data['time'].dt.weekday

        self._all_item_ids = np.asarray(item_data['item_id_encoded'].unique())
        if 'category_encoded' in item_data.columns:
            self._item_to_category = dict(
                zip(item_data['item_id_encoded'].astype(int),
                    item_data['category_encoded'].astype(int))
            )
        else:
            self._item_to_category = {}

        # Precompute the popularity sampling distribution. Items unseen
        # in `user_data` get a small base count so they retain a non-zero
        # probability — without this, the recommender can never produce
        # them as negatives and the model never learns to score them low.
        item_counts = (user_data['item_id_encoded']
                       .value_counts()
                       .reindex(self._all_item_ids, fill_value=0)
                       .to_numpy(dtype=np.float64))
        smoothed = (item_counts + 1.0) ** self.neg_alpha
        self._item_sampling_probs = smoothed / smoothed.sum()

        # Fit numerical feature tables on training-window history only.
        # `vocab_sizes` comes from the same encoders so the tables are
        # already aligned with the embedding-table row indices.
        train_end = pd.to_datetime(self.config['training']['train_end_date'])
        self.feature_engineer.fit(
            user_data,
            vocab_sizes=self.get_categorical_dims(),
            end_date=train_end,
        )
        self._user_features_tensor = self.feature_engineer.user_features_tensor()
        self._item_features_tensor = self.feature_engineer.item_features_tensor()
        return user_data, item_data

    # ------------------------------------------------------------------
    # Sequence building
    # ------------------------------------------------------------------
    def build_user_sequences(self,
                             user_data: pd.DataFrame,
                             end_date: pd.Timestamp) -> Dict[int, Dict[str, np.ndarray]]:
        """Per-user behavior sequence up to (and including) `end_date`."""
        max_len = self.max_seq_length
        filtered = user_data[user_data['time'] <= end_date]
        filtered = filtered.sort_values(['user_id_encoded', 'time'])

        sequences: Dict[int, Dict[str, np.ndarray]] = {}
        for user_id, group in filtered.groupby('user_id_encoded'):
            tail = group.tail(max_len)
            seq_len = len(tail)
            items = np.full(max_len, PAD_ID, dtype=np.int64)
            behaviors = np.full(max_len, PAD_ID, dtype=np.int64)
            # mask: True at positions to be IGNORED by attention (i.e. padding)
            mask = np.ones(max_len, dtype=bool)
            items[:seq_len] = tail['item_id_encoded'].values
            behaviors[:seq_len] = tail['behavior_type'].values
            mask[:seq_len] = False
            sequences[int(user_id)] = {
                'items': items,
                'behaviors': behaviors,
                'mask': mask,
                'length': np.int64(seq_len),
            }
        logger.info('Built sequences for %d users (max_len=%d)', len(sequences), max_len)
        return sequences

    # ------------------------------------------------------------------
    # Positive / negative interaction sampling
    # ------------------------------------------------------------------
    def sample_interactions(self,
                            user_data: pd.DataFrame,
                            target_date: pd.Timestamp) -> pd.DataFrame:
        """Build (user, item, label) rows for the given day.

        Positives are purchase actions (behavior_type==4) on `target_date`.
        Negatives are random items the user did not buy on that date.
        """
        day_actions = user_data[user_data['time'].dt.date == target_date.date()]
        positives = day_actions[day_actions['behavior_type'] == PURCHASE_BEHAVIOR][
            ['user_id_encoded', 'item_id_encoded']
        ].drop_duplicates()

        if positives.empty:
            logger.warning('No positives on %s — falling back to all interactions as labels', target_date.date())
            positives = day_actions[['user_id_encoded', 'item_id_encoded']].drop_duplicates()

        if self._all_item_ids is None or len(self._all_item_ids) == 0:
            raise RuntimeError('preprocess_data must be called before sample_interactions')

        rows: List[Tuple[int, int, int]] = []
        user_pos: Dict[int, set] = {}
        for u, i in zip(positives['user_id_encoded'].values, positives['item_id_encoded'].values):
            user_pos.setdefault(int(u), set()).add(int(i))
            rows.append((int(u), int(i), 1))

        all_items = self._all_item_ids
        probs = self._item_sampling_probs if self.neg_strategy == 'popularity' else None
        for user_id, pos_items in user_pos.items():
            needed = self.neg_ratio * len(pos_items)
            # Oversample to account for collisions with positives.
            sampled = self.rng.choice(
                all_items,
                size=needed + len(pos_items),
                replace=True,
                p=probs,
            )
            kept = 0
            for item_id in sampled:
                if kept >= needed:
                    break
                item_id_int = int(item_id)
                if item_id_int in pos_items:
                    continue
                rows.append((user_id, item_id_int, 0))
                kept += 1

        df = pd.DataFrame(rows, columns=['user_id', 'item_id', 'label']).astype({
            'user_id': np.int64,
            'item_id': np.int64,
            'label': np.int64,
        })
        df['category_id'] = df['item_id'].map(self._item_to_category).fillna(PAD_ID).astype(np.int64)
        logger.info('Sampled %d interactions on %s (pos=%d, neg=%d)',
                    len(df), target_date.date(),
                    int(df['label'].sum()), int((df['label'] == 0).sum()))
        return df

    # ------------------------------------------------------------------
    # Categorical dimensions for embedding tables
    # ------------------------------------------------------------------
    def get_categorical_dims(self) -> Dict[str, int]:
        return {
            feat: len(enc.classes_)
            for feat, enc in self.encoders.items()
            if hasattr(enc, 'classes_')
        }

    def get_user_feature_dim(self) -> int:
        return self.feature_engineer.user_feature_dim

    def get_item_feature_dim(self) -> int:
        return self.feature_engineer.item_feature_dim

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def prepare_train_val_data(self) -> Tuple[RecommendationDataset, RecommendationDataset]:
        user_data, item_data = self.load_data()
        user_data, item_data = self.preprocess_data(user_data, item_data)

        train_end = pd.to_datetime(self.config['training']['train_end_date'])
        val_date = pd.to_datetime(self.config['training']['pred_date'])

        # Sequences are built only from history up to train_end so we don't
        # leak validation labels into the user's behavior representation.
        sequences = self.build_user_sequences(user_data, train_end)

        train_interactions = self.sample_interactions(user_data, train_end)
        val_interactions = self.sample_interactions(user_data, val_date)

        train_ds = RecommendationDataset(
            train_interactions, sequences, self.max_seq_length,
            user_features=self._user_features_tensor,
            item_features=self._item_features_tensor,
        )
        val_ds = RecommendationDataset(
            val_interactions, sequences, self.max_seq_length,
            user_features=self._user_features_tensor,
            item_features=self._item_features_tensor,
        )
        return train_ds, val_ds

    def prepare_test_data(self, test_date: Optional[str] = None) -> RecommendationDataset:
        user_data, item_data = self.load_data()
        user_data, item_data = self.preprocess_data(user_data, item_data)

        if test_date is None:
            test_date = self.config['training']['pred_date']
        target = pd.to_datetime(test_date)

        sequences = self.build_user_sequences(user_data, target)
        interactions = self.sample_interactions(user_data, target)
        return RecommendationDataset(
            interactions, sequences, self.max_seq_length,
            user_features=self._user_features_tensor,
            item_features=self._item_features_tensor,
        )

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def create_submission(self,
                          predictions: np.ndarray,
                          interactions: pd.DataFrame) -> pd.DataFrame:
        """Build a top-k submission DataFrame from raw scores.

        Conforms to the Tianchi spec: two string columns ``user_id`` and
        ``item_id``, deduplicated, UTF-8 (encoding is the caller's job
        at ``to_csv`` time).

        `predictions` must align row-wise with `interactions`.
        """
        if len(predictions) != len(interactions):
            raise ValueError('predictions and interactions must have the same length')

        df = interactions[['user_id', 'item_id']].copy()
        df['score'] = predictions
        # Decode back to original IDs.
        df['user_id'] = self.encoders['user_id'].inverse_transform(df['user_id'].astype(int))
        df['item_id'] = self.encoders['item_id'].inverse_transform(df['item_id'].astype(int))

        # If multiple rows in the candidate set share the same (user,
        # item) pair (can happen when candidates are built from history
        # spanning several days), keep the row with the highest score.
        df = df.sort_values('score', ascending=False).drop_duplicates(['user_id', 'item_id'])

        top_k = int(self.config['training'].get('top_k', 20))
        df = (df.sort_values(['user_id', 'score'], ascending=[True, False])
                .groupby('user_id', sort=False).head(top_k)
                .reset_index(drop=True))

        # Spec says both columns must be string-typed.
        df['user_id'] = df['user_id'].astype(str)
        df['item_id'] = df['item_id'].astype(str)
        return df[['user_id', 'item_id']]

    # ------------------------------------------------------------------
    # Stats (used by `analyze-data` CLI)
    # ------------------------------------------------------------------
    def calculate_sequence_stats(self, user_data: pd.DataFrame) -> Dict[str, float]:
        lengths = user_data.groupby('user_id').size()
        return {
            'avg_length': float(lengths.mean()),
            'max_length': int(lengths.max()),
            'min_length': int(lengths.min()),
        }


if __name__ == '__main__':
    processor = DataProcessor('config/config.yaml')
    train_ds, val_ds = processor.prepare_train_val_data()
    logger.info('train=%d val=%d', len(train_ds), len(val_ds))
