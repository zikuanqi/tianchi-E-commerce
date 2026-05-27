"""
移动电商推荐系统
~~~~~~~~~~~~~~~

基于阿里巴巴移动电商平台的用户行为数据构建的深度学习推荐系统。

主要模块:
- data_processing: 数据预处理模块
- feature_engineering: 特征工程模块
- models: 深度学习模型模块
- trainer: 训练管理模块
- utils: 工具函数模块
- data: 数据集和数据加载器

示例:
    >>> from src.trainer import DeepModelTrainer
    >>> trainer = DeepModelTrainer('config/config.yaml')
    >>> metrics = trainer.train()
"""

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl

from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.models.deep_recommender import DeepRecommender
from src.trainer import DeepModelTrainer
from src.data.dataset import RecommendationDataset
from src.utils import (
    setup_logging,
    timer,
    memory_usage,
    DataFrameSerializer,
    reduce_memory_usage,
    save_dict_to_json,
    load_dict_from_json,
    create_submission_file,
    check_data_quality,
    TensorBatchGenerator
)

__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

# 版本信息
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'final'
}

# 导出的主要类和函数
__all__ = [
    # 数据处理
    'DataProcessor',
    'FeatureEngineer',
    'RecommendationDataset',
    'TensorBatchGenerator',
    
    # 模型和训练
    'DeepRecommender',
    'DeepModelTrainer',
    
    # 工具函数
    'setup_logging',
    'timer',
    'memory_usage',
    'DataFrameSerializer',
    'reduce_memory_usage',
    'save_dict_to_json',
    'load_dict_from_json',
    'create_submission_file',
    'check_data_quality'
]

# 默认配置
DEFAULT_CONFIG = {
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'tensorboard': True
    },
    'data': {
        'random_seed': 42,
        'validation_split': 0.2,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 2
    },
    'model': {
        'type': 'deep',
        'architecture': {
            'embedding_dim': 64,
            'hidden_dims': [256, 128, 64],
            'dropout': 0.3
        },
        'training': {
            'batch_size': 1024,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'scheduler': {
                'type': 'cosine',
                'T_max': 100,
                'eta_min': 1e-6
            }
        }
    },
    'device': {
        'accelerator': 'auto',
        'strategy': 'ddp_find_unused_parameters_true',
        'devices': 'auto',
        'precision': '16-mixed'
    }
}

def get_version():
    """获取版本信息"""
    version_str = '{major}.{minor}.{patch}'.format(**VERSION_INFO)
    if VERSION_INFO['release'] != 'final':
        version_str += '-' + VERSION_INFO['release']
    return version_str

def setup_environment(config: dict = None):
    """
    初始化运行环境
    
    Args:
        config: 配置字典，如果为None则使用默认配置
    """
    if config is None:
        config = DEFAULT_CONFIG
        
    # 设置随机种子
    seed = config['data']['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # 设置CUDA
    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.deterministic = True
        
    # 设置PyTorch Lightning
    pl.seed_everything(seed)
        
    # 设置日志
    setup_logging(config['logging'])
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 返回设备信息
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_gpus': torch.cuda.device_count(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    }

def init():
    """包初始化函数"""
    env_info = setup_environment()
    return env_info

# 自动运行初始化配置
ENV_INFO = init()
