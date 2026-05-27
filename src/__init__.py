"""Tianchi mobile e-commerce recommendation system."""

from src.data.dataset import RecommendationDataset
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.models.deep_recommender import DeepRecommender
from src.trainer import ModelTrainer
from src.utils import (
    DataFrameSerializer,
    TensorBatchGenerator,
    check_data_quality,
    create_submission_file,
    load_dict_from_json,
    reduce_memory_usage,
    save_dict_to_json,
    setup_logging,
    timer,
)

__version__ = '0.3.0'
__author__ = '綦子宽'

__all__ = [
    'DataProcessor',
    'FeatureEngineer',
    'RecommendationDataset',
    'DeepRecommender',
    'ModelTrainer',
    'setup_logging',
    'timer',
    'DataFrameSerializer',
    'reduce_memory_usage',
    'save_dict_to_json',
    'load_dict_from_json',
    'create_submission_file',
    'check_data_quality',
    'TensorBatchGenerator',
]
