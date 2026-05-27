## 移动电商深度推荐系统

基于阿里巴巴移动电商平台用户行为数据的深度推荐系统，针对天池竞赛 *移动推荐算法* 设计。采用两塔 + Transformer 序列模型对用户行为序列建模，实现个性化推荐。

[![smoke tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)

## 项目特点

- 两塔结构 + Transformer 序列编码器
- 基于 PyTorch Lightning，支持 GPU/CPU 自动切换
- 完整的数据处理与负采样流水线
- 端到端冒烟测试（CPU 上 ~12s 跑完）
- TensorBoard 可选；缺失时自动回退到 CSV Logger

## 项目结构

```
tianchi-E-commerce/
├── config/
│   └── config.yaml              # 配置文件
├── data/
│   ├── raw/                     # 原始 CSV
│   ├── processed/               # 中间产物
│   ├── checkpoints/             # 模型检查点
│   └── output/                  # 提交与分析结果
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # RecommendationDataset
│   │   └── dataloader.py        # build_dataloader + collate_fn
│   ├── models/
│   │   ├── __init__.py
│   │   ├── deep_recommender.py  # DeepRecommender (LightningModule)
│   │   └── layer.py             # PositionalEncoding, build_mlp
│   ├── data_processing.py       # DataProcessor: 数据加载 + 编码 + 负采样
│   ├── feature_engineering.py   # （可选）独立特征工程模块
│   ├── trainer.py               # ModelTrainer: 训练/推理编排
│   └── utils.py                 # 日志/计时/IO/张量工具
├── tests/
│   ├── conftest.py              # 合成数据 fixture
│   └── test_smoke.py            # 端到端冒烟测试
├── analyze_results.py           # 预测结果可视化
├── main.py                      # CLI 入口
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.9 +
- （可选）CUDA 11.x，否则自动使用 CPU
- 主要依赖：`torch>=1.9`, `pytorch-lightning>=1.5`, `pandas`, `scikit-learn`, `click`, `pyyaml`

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 跑冒烟测试（不需要任何原始数据，~12s）
python -m pytest tests/ -v

# 3. 验证配置
python main.py validate-config

# 4. 数据分析（需要 data/raw/ 下的原始 CSV）
python main.py analyze-data

# 5. 训练 / 预测 / 全流程
python main.py train
python main.py predict
python main.py run-all
```

### 数据准备

将下列文件放入 `data/raw/`：

- `tianchi_fresh_comp_train_user_2w.csv`
- `tianchi_fresh_comp_train_item_2w.csv`

## 模型架构

### 两塔 + 序列

```
user_id  ──► User Embedding ──► MLP ────┐
                                         │
item_id  ──► Item Embedding             │
category ──► Cat  Embedding ──concat──► MLP ──┐
                                                ├──► concat ──► MLP ──► logit
seq_items ──► Item Embedding                   │
seq_behav ──► Behavior Embedding ──concat──► Linear ──► PosEnc
                            │
                            ▼
                Transformer Encoder ──► Masked Mean Pool ──► MLP ──┘
```

- 用户塔：`user_id` → 嵌入 → MLP
- 商品塔：`item_id` 与 `category` 嵌入拼接 → MLP
- 序列塔：`(item, behavior)` 序列 → 位置编码 → Transformer → 掩码均值池化
- 融合层：三路表示拼接 → MLP → **logit**（配合 `BCEWithLogitsLoss`，数值更稳）

### 训练策略

- AdamW + Cosine Annealing
- BCEWithLogitsLoss（避免 sigmoid + BCE 数值问题）
- 早停（监控 `val_loss`，可配置 `patience`）
- 梯度裁剪 + （GPU 上）混合精度

### 负采样

- 正样本：目标日 `behavior_type == 4`（购买）
- 负样本：该用户当日未购买的随机商品（比例由 `negative_sampling_ratio` 控制）

## 评估指标

- `train_loss` / `val_loss`（BCE）
- `val_acc`（二分类阈值 0.5）
- 后续可通过 `analyze_results.py` 计算精确率/召回率/F1/ROC

## 常见问题

1. **GPU 不可用 / CUDA 缺失**：把 `device.accelerator` 设为 `cpu`，或保留 `auto` 让 Trainer 自动检测。
2. **`tensorboard` 没装**：训练自动回退到 `CSVLogger`，日志写入 `logs/csv_logs/`。
3. **Windows 多进程报错**：默认 `system.num_workers: 0`；如确需多进程加载，请保证在 `if __name__ == '__main__'` 守卫下运行。

## 开发

```bash
# 运行测试
python -m pytest tests/ -v
```

## 作者

綦子宽

## 更新日志

### v0.3.0 (2026-05-28)
- 全面重写数据/模型/训练流水线，使各层接口一致
- 新增端到端冒烟测试
- 默认 CPU 友好；GPU 不可用时静默回退
- `BCEWithLogitsLoss` 取代 `Sigmoid + BCE`
- 删除重复的 `src/model.py`，整理 `__init__.py`

### v0.2.1 (2024-11-12)
- 迁移到深度学习架构
- 添加 GPU 训练支持

### v0.1.0 (2024-11-10)
- 初始版本

## 参考

- [天池移动推荐算法](https://tianchi.aliyun.com/competition/entrance/231522/information)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)

## 许可证

MIT
