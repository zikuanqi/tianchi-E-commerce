"""Training/inference orchestrator built on PyTorch Lightning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

try:
    from pytorch_lightning.loggers import TensorBoardLogger
    _HAS_TENSORBOARD = True
except ImportError:  # pragma: no cover - tensorboard is optional
    TensorBoardLogger = None  # type: ignore[assignment]
    _HAS_TENSORBOARD = False

from src.data.dataloader import build_dataloader
from src.data.dataset import RecommendationDataset
from src.data_processing import DataProcessor
from src.models.deep_recommender import DeepRecommender
from src.utils import report_memory, set_seeds, setup_logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Wraps DataProcessor + DeepRecommender + pl.Trainer.

    The trainer is CPU-friendly: it picks up whatever device the config
    requests but never raises if a GPU isn't present. It also defers
    model construction until it knows the categorical vocab sizes —
    those come from `DataProcessor.get_categorical_dims()` after the
    first call to `prepare_train_val_data`.
    """

    def __init__(self, config: Union[str, Path, dict]):
        if isinstance(config, (str, Path)):
            with open(config, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError('config must be a path or a dict')

        setup_logging(self.config.get('logging', {}))
        set_seeds(int(self.config.get('system', {}).get('seed', 42)))

        self.data_processor = DataProcessor(self.config)
        self.model: Optional[DeepRecommender] = None
        self.pl_trainer: Optional[pl.Trainer] = None
        self._resolve_device()

    # ------------------------------------------------------------------
    # Device & trainer setup
    # ------------------------------------------------------------------
    def _resolve_device(self) -> None:
        """Honor the config's accelerator request but downgrade silently."""
        requested = str(self.config.get('device', {}).get('accelerator', 'auto')).lower()
        cuda_ok = torch.cuda.is_available()
        if requested == 'gpu' and not cuda_ok:
            logger.warning('Config requested GPU but CUDA is unavailable — falling back to CPU')
            self.config['device']['accelerator'] = 'cpu'
            self.config['device']['precision'] = 32
        elif requested == 'auto':
            self.config['device']['accelerator'] = 'gpu' if cuda_ok else 'cpu'
        logger.info('Using accelerator: %s', self.config['device']['accelerator'])

    def _build_pl_trainer(self) -> pl.Trainer:
        device_cfg = self.config['device']
        train_cfg = self.config['model']['training']
        paths = self.config['data']['paths']

        Path(paths['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(paths['log_dir']).mkdir(parents=True, exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                dirpath=paths['checkpoint_dir'],
                filename='model-{epoch:02d}-{val_loss:.4f}',
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=int(train_cfg['early_stopping']['patience']),
                min_delta=float(train_cfg['early_stopping']['min_delta']),
                mode='min',
            ),
        ]

        opt_cfg = train_cfg.get('optimization', {})
        if _HAS_TENSORBOARD:
            try:
                pl_logger = TensorBoardLogger(save_dir=paths['log_dir'], name='lightning_logs')
            except ModuleNotFoundError:
                # tensorboardX/tensorboard not actually importable at runtime
                pl_logger = CSVLogger(save_dir=paths['log_dir'], name='csv_logs')
        else:
            pl_logger = CSVLogger(save_dir=paths['log_dir'], name='csv_logs')

        return pl.Trainer(
            accelerator=device_cfg['accelerator'],
            devices=device_cfg.get('devices', 'auto'),
            precision=device_cfg.get('precision', 32),
            max_epochs=int(train_cfg['num_epochs']),
            callbacks=callbacks,
            logger=pl_logger,
            gradient_clip_val=float(opt_cfg.get('gradient_clip_val', 1.0)),
            accumulate_grad_batches=int(opt_cfg.get('accumulate_grad_batches', 1)),
            log_every_n_steps=10,
            enable_progress_bar=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare_data(self) -> Tuple[RecommendationDataset, RecommendationDataset]:
        report_memory('before prepare_train_val_data')
        train_ds, val_ds = self.data_processor.prepare_train_val_data()
        report_memory('after prepare_train_val_data')
        logger.info('Datasets: train=%d, val=%d', len(train_ds), len(val_ds))
        return train_ds, val_ds

    def build_model(self) -> DeepRecommender:
        vocab_sizes = self.data_processor.get_categorical_dims()
        if not vocab_sizes:
            raise RuntimeError('No vocab sizes available — call prepare_data first')
        logger.info('Building model with vocab sizes: %s', vocab_sizes)
        self.model = DeepRecommender(self.config, vocab_sizes=vocab_sizes)
        return self.model

    def train(self,
              train_dataset: Optional[RecommendationDataset] = None,
              val_dataset: Optional[RecommendationDataset] = None) -> Dict[str, float]:
        """Run a full training loop. Datasets are optional; built if missing."""
        if train_dataset is None or val_dataset is None:
            train_dataset, val_dataset = self.prepare_data()

        if self.model is None:
            self.build_model()

        sys_cfg = self.config.get('system', {})
        train_cfg = self.config['model']['training']
        batch_size = int(train_cfg['batch_size'])
        # Multi-process loading on Windows requires `if __name__ == "__main__"`
        # guards everywhere, which we can't enforce; default to 0 workers.
        num_workers = 0 if torch.multiprocessing.get_start_method(allow_none=True) == 'spawn' \
            else int(sys_cfg.get('num_workers', 0))
        pin_memory = bool(sys_cfg.get('pin_memory', False)) and torch.cuda.is_available()

        train_loader = build_dataloader(train_dataset, batch_size, shuffle=True,
                                        num_workers=num_workers, pin_memory=pin_memory)
        val_loader = build_dataloader(val_dataset, batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=pin_memory)

        self.pl_trainer = self._build_pl_trainer()
        self.pl_trainer.fit(self.model, train_loader, val_loader)

        # Collect final metrics from the callback metrics dict.
        metrics = {k: float(v) for k, v in self.pl_trainer.callback_metrics.items()}
        logger.info('Training complete. Metrics: %s', metrics)
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def load_checkpoint(self, ckpt_path: Union[str, Path]) -> DeepRecommender:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        logger.info('Loading checkpoint %s', ckpt_path)
        self.model = DeepRecommender.load_from_checkpoint(str(ckpt_path))
        # Hyperparameters live inside the checkpoint; sync our config so
        # downstream consumers (DataProcessor) see the same training cfg.
        ckpt_cfg = self.model.hparams.get('config') if hasattr(self.model, 'hparams') else None
        if isinstance(ckpt_cfg, dict):
            self.config = ckpt_cfg
            self.data_processor = DataProcessor(self.config)
        return self.model

    def predict(self, dataset: RecommendationDataset) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('predict called before model was built or loaded')
        train_cfg = self.config['model']['training']
        batch_size = int(train_cfg['batch_size'])

        loader = build_dataloader(dataset, batch_size, shuffle=False,
                                  num_workers=0, pin_memory=False)
        if self.pl_trainer is None:
            self.pl_trainer = self._build_pl_trainer()

        outputs = self.pl_trainer.predict(self.model, loader)
        return torch.cat(outputs, dim=0).detach().cpu().numpy()
