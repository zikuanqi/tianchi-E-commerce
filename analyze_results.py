import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import auc, confusion_matrix, roc_curve

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResultAnalyzer:
    def __init__(self, pred_file: str = 'data/output/tianchi_mobile_recommendation_predict.csv'):
        """初始化结果分析器"""
        self.pred_file = pred_file
        self.output_dir = Path('data/output/analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_predictions(self):
        """分析预测结果"""
        try:
            # 读取预测结果
            logger.info(f"Reading predictions from {self.pred_file}")
            self.pred_df = pd.read_csv(self.pred_file)
            
            # 运行所有分析
            self.basic_analysis()
            self.distribution_analysis()
            self.visualization_analysis()
            
            # 保存结果
            self.save_results()
            
        except Exception as e:
            logger.error(f"Error analyzing predictions: {str(e)}")
            raise

    def basic_analysis(self):
        """基本统计分析"""
        self.basic_stats = {
            'total_predictions': len(self.pred_df),
            'unique_users': self.pred_df['user_id'].nunique(),
            'unique_items': self.pred_df['item_id'].nunique(),
            'avg_items_per_user': len(self.pred_df) / self.pred_df['user_id'].nunique(),
            'memory_usage_mb': self.pred_df.memory_usage(deep=True).sum() / 1024**2
        }
        
        logger.info("\nBasic Statistics:")
        for key, value in self.basic_stats.items():
            logger.info(f"{key}: {value:,.2f}")

    def distribution_analysis(self):
        """分布统计分析"""
        # 用户推荐分布
        self.user_rec_counts = self.pred_df['user_id'].value_counts()
        self.item_rec_counts = self.pred_df['item_id'].value_counts()
        
        self.distribution_stats = {
            'user_recommendations': {
                'min': int(self.user_rec_counts.min()),
                'max': int(self.user_rec_counts.max()),
                'mean': float(self.user_rec_counts.mean()),
                'median': float(self.user_rec_counts.median()),
                'std': float(self.user_rec_counts.std())
            },
            'item_recommendations': {
                'min': int(self.item_rec_counts.min()),
                'max': int(self.item_rec_counts.max()),
                'mean': float(self.item_rec_counts.mean()),
                'median': float(self.item_rec_counts.median()),
                'std': float(self.item_rec_counts.std())
            }
        }
        
        logger.info("\nDistribution Analysis:")
        logger.info(f"User recommendation distribution: {self.distribution_stats['user_recommendations']}")
        logger.info(f"Item recommendation distribution: {self.distribution_stats['item_recommendations']}")

    def visualization_analysis(self):
        """可视化分析"""
        # 创建图形
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('default')
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 用户推荐分布
        plt.subplot(221)
        sns.histplot(self.user_rec_counts, bins=30)
        plt.title('User Recommendation Distribution')
        plt.xlabel('Number of Recommendations')
        plt.ylabel('Number of Users')
        
        # 2. 商品推荐分布
        plt.subplot(222)
        sns.histplot(self.item_rec_counts, bins=30)
        plt.title('Item Recommendation Distribution')
        plt.xlabel('Number of Times Recommended')
        plt.ylabel('Number of Items')
        
        # 3. 推荐密度图
        plt.subplot(223)
        sns.kdeplot(self.user_rec_counts)
        plt.title('User Recommendation Density')
        plt.xlabel('Number of Recommendations')
        
        # 4. 累积分布
        plt.subplot(224)
        plt.plot(np.sort(self.user_rec_counts), 
                np.linspace(0, 1, len(self.user_rec_counts)))
        plt.title('Cumulative Distribution')
        plt.xlabel('Number of Recommendations')
        plt.ylabel('Cumulative Proportion')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'recommendation_analysis.png')
        plt.close()

    def analyze_model_outputs(self, model_outputs: Dict[str, torch.Tensor]):
        """分析模型输出"""
        # 提取模型输出
        embeddings = model_outputs['embeddings'].cpu().numpy()
        predictions = model_outputs['predictions'].cpu().numpy()
        labels = model_outputs['labels'].cpu().numpy()
        
        # 1. t-SNE可视化
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Embeddings')
        plt.savefig(self.output_dir / 'embeddings_tsne.png')
        plt.close()
        
        # 2. ROC曲线
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig(self.output_dir / 'roc_curve.png')
        plt.close()
        
        # 3. 混淆矩阵
        conf_matrix = confusion_matrix(labels, predictions > 0.5)
        plt.figure(figsize=(8, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()

    def analyze_training_history(self, history_file: str):
        """分析训练历史"""
        with open(history_file) as f:
            history = json.load(f)
            
        # 1. 损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig(self.output_dir / 'training_history.png')
        plt.close()
        
        # 2. 指标趋势
        metrics = ['precision', 'recall', 'f1']
        plt.figure(figsize=(12, 4))
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, i)
            plt.plot(history[f'train_{metric}'], label=f'Train {metric}')
            plt.plot(history[f'val_{metric}'], label=f'Val {metric}')
            plt.title(f'{metric.capitalize()} History')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_history.png')
        plt.close()

    def save_results(self):
        """保存分析结果"""
        results = {
            'basic_statistics': self.basic_stats,
            'distribution_statistics': self.distribution_stats,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.output_dir / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Analysis results saved to {self.output_dir}")

if __name__ == "__main__":
    analyzer = ResultAnalyzer()
    analyzer.analyze_predictions()
    
    # 如果有训练历史文件
    if Path('data/output/training_history.json').exists():
        analyzer.analyze_training_history('data/output/training_history.json')
