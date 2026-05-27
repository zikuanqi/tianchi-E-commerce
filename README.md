## 移动电商深度推荐系统

基于阿里巴巴移动电商平台的用户行为数据构建的深度学习推荐系统，针对天池竞赛"移动推荐算法"设计。采用深度学习模型对用户行为序列建模，实现个性化推荐。

## 项目特点

- 深度学习模型架构（多塔模型 + Transformer序列编码）
- GPU加速训练支持
- 完整的数据处理和特征工程流程
- PyTorch Lightning框架实现
- 丰富的评估和可视化工具

## 项目结构
```
mobile_recommendation/
│
├── config/
│   └── config.yaml           # 配置文件
│
├── data/
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后的数据
│   ├── checkpoints/          # 模型检查点
│   └── output/               # 输出结果
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # 数据集类
│   │   └── dataloader.py     # 数据加载器
│   ├── models/
│   │   ├── __init__.py
│   │   ├── deep_recommender.py  # 深度推荐模型
│   │   └── layer.py            # 模型层定义
│   ├── data_processing.py    # 数据处理
│   ├── feature_engineering.py # 特征工程
│   ├── trainer.py            # 训练管理
│   └── utils.py             # 工具函数
│
├── requirements.txt          # 项目依赖
└── main.py                  # 主程序
```

## 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU训练)
- 依赖包：
  ```
  torch>=1.9.0
  pytorch-lightning>=1.5.0
  numpy>=1.20.0
  pandas>=1.3.0
  scikit-learn>=0.24.0
  ```

## 快速开始

### 1. 环境配置

```bash
# 创建Conda环境
conda create -n tianchi-deep python=3.9
conda activate tianchi-deep

# 安装PyTorch (GPU版本)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 数据准备

将数据文件放入 `data/raw/` 目录：
- tianchi_fresh_comp_train_user_2w.csv
- tianchi_fresh_comp_train_item_2w.csv

### 3. 运行项目

```bash
# 验证配置
python3 main.py validate-config

# 数据分析
python3 main.py analyze-data

# 训练模型（使用GPU）
python3 main.py train --gpu

# 生成预测
python3 main.py predict

# 运行完整流程
python3 main.py run-all --gpu
```

## 模型架构

### 多塔结构
1. 用户塔
   - 用户基础特征编码
   - 行为序列编码
   - 多层感知机

2. 商品塔
   - 商品特征编码
   - 类别嵌入
   - 多层感知机

3. 序列编码器
   - Transformer编码器
   - 自注意力机制
   - 位置编码

### 特征处理
- 类别特征嵌入
- 数值特征标准化
- 序列特征处理
- 时间特征编码

### 训练策略
- AdamW优化器
- 余弦退火学习率
- 混合精度训练
- 梯度裁剪

## 性能优化

1. 计算优化
   - GPU加速
   - 混合精度训练
   - 数据预取

2. 内存优化
   - 渐进式加载
   - 特征缓存
   - 内存监控

3. 训练优化
   - 分布式训练
   - 梯度累积
   - 检查点保存

## 评估指标

- 精确率(Precision)
- 召回率(Recall)
- F1分数

## 可视化分析

- TensorBoard支持
- 训练过程可视化
- 嵌入空间可视化
- 注意力权重可视化

## 开发指南

1. 代码风格
   - 遵循PEP 8
   - 类型注解
   - 文档字符串

2. 测试
   ```bash
   # 运行单元测试
   pytest tests/
   
   # 运行特定测试
   pytest tests/test_model.py -v
   ```

## 常见问题

1. CUDA内存不足
   - 减小batch_size
   - 启用混合精度训练
   - 使用梯度累积

2. 训练不稳定
   - 调整学习率
   - 使用梯度裁剪
   - 检查数据预处理

## 作者

綦子宽

## 更新日志

### v0.2.1 (2024-11-12)
- 迁移到深度学习架构
- 添加GPU训练支持
- 优化特征处理流程
- 增加可视化功能

### v0.1.0 (2024-11-10)
- 初始版本发布
- 基本功能实现

## 参考

- [天池移动推荐算法](https://tianchi.aliyun.com/competition/entrance/231522/information)
- [PyTorch文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch Lightning文档](https://pytorch-lightning.readthedocs.io/)

## 许可证

MIT License
```
