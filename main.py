"""CLI entry point for training, prediction, data analysis, and config validation."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import click
import pytorch_lightning as pl
import torch
import yaml

from src.data_processing import DataProcessor
from src.trainer import ModelTrainer
from src.utils import setup_logging, timer

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@click.group()
def cli() -> None:
    """Tianchi mobile-recommendation algorithm CLI."""


@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
@click.option('--debug', is_flag=True, help='Enable DEBUG logging and seed everything')
@timer
def train(config: str, debug: bool) -> None:
    """Train the recommender model."""
    cfg = load_config(config)
    if debug:
        cfg.setdefault('logging', {})['level'] = 'DEBUG'
    setup_logging(cfg.get('logging', {}))
    pl.seed_everything(int(cfg.get('system', {}).get('seed', 42)))
    logger.info('Starting training...')

    try:
        trainer = ModelTrainer(cfg)
        train_ds, val_ds = trainer.prepare_data()
        trainer.build_model()
        metrics = trainer.train(train_ds, val_ds)

        output_dir = Path(cfg['data']['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'training_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info('Training complete. Metrics written to %s', output_dir / 'training_metrics.json')
    except Exception as exc:
        logger.exception('Training failed')
        raise click.ClickException(str(exc))


@cli.command()
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
@click.option('--test-date', '-d', default=None, help='Test date (YYYY-MM-DD)')
@click.option('--checkpoint', default=None, help='Checkpoint path; defaults to last.ckpt')
@timer
def predict(config: str, test_date: str, checkpoint: str) -> None:
    """Generate predictions from a trained checkpoint."""
    cfg = load_config(config)
    setup_logging(cfg.get('logging', {}))
    logger.info('Starting prediction...')

    try:
        trainer = ModelTrainer(cfg)
        ckpt = checkpoint or str(Path(cfg['data']['paths']['checkpoint_dir']) / 'last.ckpt')
        trainer.load_checkpoint(ckpt)

        test_ds = trainer.data_processor.prepare_test_data(test_date)
        scores = trainer.predict(test_ds)

        import pandas as pd
        interactions_df = pd.DataFrame({
            'user_id': test_ds.user_ids.numpy(),
            'item_id': test_ds.item_ids.numpy(),
        })
        submission = trainer.data_processor.create_submission(scores, interactions_df)

        output_dir = Path(cfg['data']['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / 'submission.csv'
        submission.to_csv(out_path, index=False)
        logger.info('Wrote %d rows to %s', len(submission), out_path)
    except Exception as exc:
        logger.exception('Prediction failed')
        raise click.ClickException(str(exc))


@cli.command('analyze-data')
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
@timer
def analyze_data(config: str) -> None:
    """Print and persist basic stats about the raw dataset."""
    cfg = load_config(config)
    setup_logging(cfg.get('logging', {}))
    logger.info('Starting data analysis...')

    try:
        processor = DataProcessor(cfg)
        user_data, item_data = processor.load_data()

        analysis = {
            'user_stats': {
                'total_users': int(user_data['user_id'].nunique()),
                'total_interactions': int(len(user_data)),
                'behavior_counts': {int(k): int(v) for k, v in
                                    user_data['behavior_type'].value_counts().items()},
                'date_range': {
                    'start': str(user_data['time'].min()),
                    'end': str(user_data['time'].max()),
                    'total_days': int((user_data['time'].max() - user_data['time'].min()).days + 1),
                },
            },
            'item_stats': {
                'total_items': int(item_data['item_id'].nunique()),
                'total_categories': int(item_data['category'].nunique()) if 'category' in item_data.columns else None,
            },
            'sequence_stats': processor.calculate_sequence_stats(user_data),
        }

        output_dir = Path(cfg['data']['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'data_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str, ensure_ascii=False)
        logger.info('Data analysis written to %s', output_dir / 'data_analysis.json')
    except Exception as exc:
        logger.exception('Data analysis failed')
        raise click.ClickException(str(exc))


@cli.command('run-all')
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
@click.option('--debug', is_flag=True, help='Enable DEBUG logging')
@timer
def run_all(config: str, debug: bool) -> None:
    """Run analyze-data → train → predict back-to-back."""
    ctx = click.get_current_context()
    ctx.invoke(analyze_data, config=config)
    ctx.invoke(train, config=config, debug=debug)
    ctx.invoke(predict, config=config, test_date=None, checkpoint=None)


@cli.command('validate-config')
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
def validate_config(config: str) -> None:
    """Sanity-check the config file."""
    try:
        cfg = load_config(config)
        required = ['data', 'model', 'training', 'device', 'logging']
        for section in required:
            if section not in cfg:
                raise click.ClickException(f'Missing required section: {section}')

        raw_user = Path(cfg['data']['paths']['raw_user_data'])
        if not raw_user.parent.exists():
            click.echo(f'WARNING: data directory does not exist: {raw_user.parent}')

        if str(cfg['device'].get('accelerator', 'auto')).lower() == 'gpu' and not torch.cuda.is_available():
            click.echo('WARNING: config requests GPU but CUDA is unavailable; trainer will fall back to CPU')

        click.echo('Configuration is valid.')
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc))


if __name__ == '__main__':
    cli()
