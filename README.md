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

# 5. 训练 / 预测 / 评估 / 全流程
python main.py train
python main.py predict              # 严格不偷看 pred_date，输出 tianchi_mobile_recommendation_predict.csv
python main.py evaluate \
    --target-date 2014-12-19 \
    --cutoff-date 2014-12-18         # 在留出日上计算竞赛集合 F1
python main.py run-all
```

### 数据准备

将官方数据放入 `data/raw/`：

- `tianchi_fresh_comp_train_user.csv`（用户行为全集 D，2014-11-18 ~ 2014-12-18）
- `tianchi_fresh_comp_train_item.csv`（商品子集 P，推荐结果必须落在 P 内）

若使用第三方分发的 `_2w` 子集，请修改 `config.yaml` 的 `data.paths.raw_user_data` / `raw_item_data` 指向实际文件。

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
- 采样分布：`negative_sampling_strategy: uniform | popularity`，热门加权时使用 word2vec 风格平滑：`p(item) ∝ (count + 1) ^ alpha`，默认 `alpha = 0.75`
- 可选：in-batch 对比损失（`use_in_batch_negatives: true`），将同一 batch 中其他正样本作为负样本，按温度归一化的余弦相似度做 softmax 交叉熵；通过 `infonce_weight` 与 BCE 主损失加权

### 数值特征

`FeatureEngineer` 从训练窗口（≤ `train_end_date`）的行为中统计每个用户/商品的稠密数值特征：

- 用户：动作总数、独立商品/类别数、四种 behavior 占比、平均小时、周末占比、活跃天数、平均行为间隔
- 商品：动作总数、独立用户数、四种 behavior 占比、用户多样性

特征 StandardScaler 标准化，并截断到 ±5 σ。零向量用于训练时未见过的冷启动用户/商品。这些数值特征与对应的嵌入向量拼接后进入两塔 MLP。

## 竞赛规范对齐

| 规范 | 实现 |
|---|---|
| 数据集 D：`user_id, item_id, behavior_type, user_geohash, item_category, time` | `DataProcessor.load_data` 按此 schema 加载，`item_category` 重命名为 `category` |
| 商品子集 P：`item_id, item_geohash, item_category` | 同上；`_all_item_ids` 从 P 提取，所有候选/推荐都限定在 P |
| `behavior_type` ∈ {1 浏览, 2 收藏, 3 加购, 4 购买} | `PURCHASE_BEHAVIOR = 4`，序列嵌入 `num_behaviors=5`（0=PAD） |
| 训练数据 11.18 ~ 12.18，预测 12.19 | `training.train_end_date: 2014-12-18` / `pred_date: 2014-12-19` |
| 推荐必须在 P 内 | `build_inference_candidates` 强制过滤；测试覆盖 |
| 推理不能"偷看"预测日 | `prepare_inference_data` 只用截止日及之前的历史；有专门的反泄漏测试 |
| 输出文件名 `tianchi_mobile_recommendation_predict.csv` | `main.py predict` 写入该文件名，UTF-8 |
| 列：`user_id, item_id`，string 类型，去重 | `create_submission` 强制 `astype(str)` + `drop_duplicates` |
| 评测：集合 P/R/F1 | `src.evaluation.set_precision_recall_f1`；`main.py evaluate` 报告 |

## 评估指标

通过 `torchmetrics` 在每个验证 epoch 计算并记录：

- `val_loss`（BCE）
- `val_acc` 准确率
- `val_precision`、`val_recall`、`val_f1`
- `val_auroc`（验证集需同时包含正负样本，否则跳过）
- 训练侧记录 `train_bce` 与 `train_infonce`（启用 InfoNCE 时）

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

### v0.5.0 (2026-05-28)
- 与 Tianchi 移动推荐算法竞赛规范完全对齐
- 提交文件名改为 `tianchi_mobile_recommendation_predict.csv`，UTF-8，强制 string + 去重
- 拆分推理路径：`prepare_inference_data` 严格不偷看预测日；`prepare_eval_data` 保留给训练验证
- 候选集限定在商品子集 P + 用户近期历史；新增 `evaluate` CLI 子命令报告竞赛集合 F1
- `set_precision_recall_f1` 工具函数与 7 项单元测试
- 默认数据文件名改为官方名称（去掉 `_2w` 后缀）
- 测试套件扩展到 35 个

### v0.4.0 (2026-05-28)
- 验证集新增 P/R/F1/AUROC 指标
- 负采样支持 `popularity` 策略，带 word2vec 风格平滑
- 集成 `FeatureEngineer`：用户/商品数值特征自动进入两塔
- 引入 `PadLabelEncoder` 把 PAD 固定到索引 0，消除嵌入与特征表的索引漂移
- 可选 InfoNCE in-batch 对比损失
- 测试套件扩展到 15 个（CPU 上 ~13 秒）

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
