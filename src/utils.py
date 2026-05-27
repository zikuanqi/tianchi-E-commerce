"""Utility helpers: logging, timing, memory, IO, and small data tools."""

from __future__ import annotations

import functools
import gc
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def timer(func: Callable) -> Callable:
    """Log how long a function takes to run."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            logger.info('%s took %.2fs', func.__name__, time.time() - start)

    return wrapper


def memory_usage(func: Callable) -> Callable:
    """Log RSS memory delta around a function call."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        before = process.memory_info().rss / 1024 ** 2
        try:
            return func(*args, **kwargs)
        finally:
            after = process.memory_info().rss / 1024 ** 2
            logger.info('%s memory: %.1fMB -> %.1fMB (Δ %+.1fMB)',
                        func.__name__, before, after, after - before)

    return wrapper


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(config: Optional[Dict] = None) -> logging.Logger:
    """Configure root logging to file + stderr.

    Safe to call multiple times — existing handlers are cleared first so we
    don't end up with duplicated log lines.
    """
    config = config or {}
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"recommender_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    level = getattr(logging, str(config.get('level', 'INFO')).upper(), logging.INFO)
    fmt = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.FileHandler(log_file, encoding='utf-8'),
                  logging.StreamHandler()],
    )

    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger.info('Logging configured. File: %s', log_file)
    return root


# ---------------------------------------------------------------------------
# Reproducibility & devices
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def report_memory(stage: str = '') -> Dict[str, float]:
    """Return (and log) a memory snapshot in GB."""
    process = psutil.Process(os.getpid())
    info = process.memory_info()
    snapshot = {
        'rss_gb': info.rss / 1024 ** 3,
        'vms_gb': info.vms / 1024 ** 3,
    }
    if torch.cuda.is_available():
        snapshot['cuda_allocated_gb'] = torch.cuda.memory_allocated() / 1024 ** 3
        snapshot['cuda_reserved_gb'] = torch.cuda.memory_reserved() / 1024 ** 3
    logger.info('Memory %s: %s', stage,
                ', '.join(f'{k}={v:.2f}' for k, v in snapshot.items()))
    return snapshot


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Downcast integer and float columns to their smallest safe dtype."""
    start_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
    if verbose:
        end_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
        logger.info('reduce_memory_usage: %.1fMB -> %.1fMB (%.1f%%)',
                    start_mb, end_mb, 100 * (start_mb - end_mb) / max(start_mb, 1e-9))
    return df


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------

def save_dict_to_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def load_dict_from_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Submission helpers
# ---------------------------------------------------------------------------

def create_submission_file(predictions: pd.DataFrame, output_path: Union[str, Path],
                           top_k: Optional[int] = None) -> Path:
    """Write a Tianchi-format submission file from a predictions DataFrame.

    `predictions` must contain at least columns: user_id, item_id, score.
    If `top_k` is given, keep only top-k items per user (by score).
    """
    required = {'user_id', 'item_id', 'score'}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f'predictions is missing columns: {missing}')

    df = predictions.copy()
    if top_k is not None:
        df = (df.sort_values(['user_id', 'score'], ascending=[True, False])
                .groupby('user_id')
                .head(top_k))
    df = df[['user_id', 'item_id']].drop_duplicates()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info('Wrote %d rows to %s', len(df), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def check_data_quality(df: pd.DataFrame, required_cols: Iterable[str] = ()) -> Dict[str, Any]:
    """Return a summary dict describing missing values, dtypes, and shape."""
    required_cols = list(required_cols)
    report: Dict[str, Any] = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {c: str(t) for c, t in df.dtypes.items()},
        'null_counts': {c: int(df[c].isna().sum()) for c in df.columns},
        'duplicate_rows': int(df.duplicated().sum()),
    }
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        report['missing_required_columns'] = missing
    return report


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

class DataFrameSerializer:
    """Save/load pandas DataFrames in parquet (preferred) or pickle format."""

    @staticmethod
    def save(df: pd.DataFrame, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in {'.parquet', '.pq'}:
            df.to_parquet(path, index=False)
        else:
            df.to_pickle(path)

    @staticmethod
    def load(path: Union[str, Path]) -> pd.DataFrame:
        path = Path(path)
        if path.suffix in {'.parquet', '.pq'}:
            return pd.read_parquet(path)
        return pd.read_pickle(path)


class TensorBatchGenerator:
    """In-memory tensor batch iterator, useful when DataLoader is overkill."""

    def __init__(self,
                 features: Dict[str, torch.Tensor],
                 labels: Optional[torch.Tensor] = None,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 device: Optional[torch.device] = None):
        if not features:
            raise ValueError('features must be a non-empty dict of tensors')
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device or get_device()
        self.n_samples = next(iter(features.values())).shape[0]

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n_samples)
        else:
            indices = torch.arange(self.n_samples)
        for start in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch = {k: v[batch_idx].to(self.device) for k, v in self.features.items()}
            if self.labels is not None:
                batch['labels'] = self.labels[batch_idx].to(self.device)
            yield batch


class MetricTracker:
    """Simple running-metric tracker keyed by metric name."""

    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for name, value in metrics.items():
            self.metrics.setdefault(name, []).append(float(value))

    def get_latest(self) -> Dict[str, float]:
        return {name: values[-1] for name, values in self.metrics.items() if values}

    def get_average(self) -> Dict[str, float]:
        return {name: float(np.mean(values)) for name, values in self.metrics.items() if values}

    def save_history(self, path: Union[str, Path]) -> None:
        save_dict_to_json(self.metrics, path)


def calculate_model_size(model: nn.Module) -> float:
    """Return model size in MB (parameters + buffers)."""
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / 1024 ** 2


def convert_to_tensor(data: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
                      dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        tensor = torch.from_numpy(data.values)
    else:
        raise TypeError(f'Unsupported data type: {type(data)}')
    return tensor.to(dtype) if dtype is not None else tensor
