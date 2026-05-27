<div align="center">

# 移动电商深度推荐系统

**基于阿里巴巴移动电商用户行为数据的深度推荐系统 · 针对天池竞赛"移动推荐算法"设计**

[![CI](https://github.com/zikuanqi/tianchi-E-commerce/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/zikuanqi/tianchi-E-commerce/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/zikuanqi/tianchi-E-commerce/branch/main/graph/badge.svg)](https://codecov.io/gh/zikuanqi/tianchi-E-commerce)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-1.5%2B-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#贡献)
[![Last Commit](https://img.shields.io/github/last-commit/zikuanqi/tianchi-E-commerce/main)](https://github.com/zikuanqi/tianchi-E-commerce/commits/main)
[![Tianchi 231522](https://img.shields.io/badge/Tianchi-231522-FF6A00)](https://tianchi.aliyun.com/competition/entrance/231522/information)

</div>

---

## 目录

- [项目特点](#项目特点)
- [模型架构](#模型架构)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [训练与推理流程](#训练与推理流程)
- [竞赛规范对齐](#竞赛规范对齐)
- [评估指标](#评估指标)
- [负采样策略](#负采样策略)
- [数值特征](#数值特征)
- [测试](#测试)
- [开发](#开发)
- [常见问题](#常见问题)
- [更新日志](#更新日志)
- [贡献](#贡献)
- [作者与许可证](#作者与许可证)

---

## 项目特点

- **两塔 + 序列**架构：用户塔 + 商品塔 + Transformer 行为序列编码器
- **CPU/GPU 自适应**：默认 `accelerator: auto`，无 CUDA 时静默回退到 CPU
- **零泄漏推理**：`prepare_inference_data` 严格不偷看预测日
- **竞赛规范对齐**：提交文件名、列类型、去重、商品子集 P 限定、集合 F1 评测全部符合规范
- **稠密数值特征**：`FeatureEngineer` 自动产出 11 维用户 + 7 维商品特征，喂入两塔
- **多种负采样**：`uniform` / `popularity`（word2vec 平滑）+ 可选 InfoNCE in-batch 负样本
- **完整评估**：训练时记录 P/R/F1/AUROC（per-row）；`evaluate` 子命令计算竞赛集合 F1
- **测试套件**：97 个测试，CPU 上 ~18s 全跑通；**99% 行+分支覆盖率**
- **Lightning 风格**：检查点自描述（`save_hyperparameters`），断点续训和迁移推理零摩擦

---

## 模型架构

```
                    ┌────────────────────────────────────────┐
                    │              融合层 MLP                 │
                    │           ↓ sigmoid → logit            │
                    │     (BCEWithLogitsLoss / InfoNCE)       │
                    └─────────────┬─────────────┬─────────────┘
                                  │ concat
       ┌────────────────────┬─────┴─────┬────────────────────┐
       │                    │           │                    │
   用户塔 MLP            商品塔 MLP        序列塔 MLP
       ▲                    ▲                    ▲
       │                    │                    │
  user_emb ⊕            item_emb ⊕         masked-mean-pool
  user_numerical       cat_emb ⊕                   ▲
       ▲              item_numerical               │
       │                    ▲                Transformer Encoder
   user_id                  │                ( + Positional Enc.)
                       item_id, category            ▲
                                            seq_item_emb ⊕ seq_behavior_emb
```

### 设计要点

1. **嵌入层**：`user_id` / `item_id` / `category` / `behavior` 各自独立嵌入；PAD 固定在索引 0（通过 `PadLabelEncoder`），`nn.Embedding(padding_idx=0)` 不参与梯度。
2. **数值特征**：与对应嵌入向量在塔输入处拼接，StandardScaler 标准化并截断到 ±5σ。
3. **序列编码**：Transformer Encoder + sinusoidal 位置编码；`batch_first=True`；通过 `src_key_padding_mask` 忽略填充位置；冷启动用户（全 PAD）安全回退到零向量。
4. **损失函数**：默认 `BCEWithLogitsLoss`（数值稳定）；可选 InfoNCE 辅助损失，按 `infonce_weight` 加权。

---

## 快速开始

### 安装

```bash
git clone https://github.com/zikuanqi/tianchi-E-commerce.git
cd tianchi-E-commerce

# 创建虚拟环境（可选）
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 跑测试（不需要原始数据）

```bash
pytest tests/ -v
# 28 passed in ~12s
```

### 准备数据

将官方数据放入 `data/raw/`：

| 文件 | 说明 |
|---|---|
| `tianchi_fresh_comp_train_user.csv` | 用户行为全集 D（2014-11-18 ~ 2014-12-18） |
| `tianchi_fresh_comp_train_item.csv` | 商品子集 P，推荐结果必须落在 P 内 |

> 使用第三方分发的 `_2w` 子集时，请在 `config/config.yaml` 中修改 `data.paths.raw_user_data` / `raw_item_data` 指向实际文件。

### 完整流水线

```bash
# 验证配置
python main.py validate-config

# 数据概览
python main.py analyze-data

# 训练
python main.py train

# 推理（写出符合竞赛规范的提交文件）
python main.py predict
#  → data/output/tianchi_mobile_recommendation_predict.csv

# 评估（在留出日上计算竞赛集合 F1）
python main.py evaluate \
    --target-date 2014-12-19 \
    --cutoff-date 2014-12-18 \
    --write-submission
#  → 控制台: F1=... P=... R=... |pred|=... |truth|=... TP=...
#  → data/output/evaluation_2014-12-19.json
```

---

## 项目结构

```
tianchi-E-commerce/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions：pytest + coverage 上传 Codecov
├── config/
│   └── config.yaml             # 全部超参数（路径、模型、训练、采样、推理）
├── data/                       # （gitignore）原始 CSV / 检查点 / 输出
│   ├── raw/
│   ├── processed/
│   ├── checkpoints/
│   └── output/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # RecommendationDataset
│   │   └── dataloader.py       # build_dataloader + collate_fn
│   ├── models/
│   │   ├── __init__.py
│   │   ├── deep_recommender.py # DeepRecommender (LightningModule)
│   │   └── layer.py            # PositionalEncoding, build_mlp
│   ├── data_processing.py      # DataProcessor + PadLabelEncoder
│   ├── feature_engineering.py  # FeatureEngineer
│   ├── trainer.py              # ModelTrainer
│   ├── evaluation.py           # set_precision_recall_f1
│   └── utils.py                # 日志/计时/IO/张量工具
├── tests/
│   ├── conftest.py             # 合成数据 fixture
│   ├── test_smoke.py
│   ├── test_negative_sampling.py
│   ├── test_feature_engineering.py
│   ├── test_infonce.py
│   ├── test_submission.py
│   ├── test_evaluation.py
│   └── test_inference.py
├── analyze_results.py          # 预测结果可视化
├── main.py                     # CLI 入口
├── pyproject.toml              # pytest / coverage / ruff 配置
├── requirements.txt
└── README.md
```

---

## 配置说明

`config/config.yaml` 关键段落（节选）：

```yaml
data:
  paths:
    raw_user_data: 'data/raw/tianchi_fresh_comp_train_user.csv'
    raw_item_data: 'data/raw/tianchi_fresh_comp_train_item.csv'
  features:
    categorical: [user_id, item_id, category]
    sequence:
      max_length: 50              # 用户行为序列截断长度
      num_behaviors: 5            # 0=PAD, 1..4=四种行为
      behavior_dim: 8
    user:  { embedding_dim: 32 }
    item:  { embedding_dim: 32 }

device:
  accelerator: 'auto'             # 'gpu' | 'cpu' | 'auto'
  precision: 32                   # GPU 上可设 16 启用混合精度

model:
  architecture:
    embedding_dim: 32
    hidden_dims: [128, 64, 32]
    dropout: 0.2
    num_attention_heads: 4
    num_transformer_layers: 2
  training:
    batch_size: 512
    num_epochs: 5
    learning_rate: 0.001
    weight_decay: 0.0001
    negative_sampling_ratio: 4
    negative_sampling_strategy: 'popularity'   # uniform | popularity
    negative_sampling_alpha: 0.75
    use_in_batch_negatives: false              # InfoNCE 辅助损失
    infonce_temperature: 0.1
    infonce_weight: 0.1
    inference_history_days: 7                  # 推理候选集回看窗口

training:
  train_end_date: '2014-12-18'
  pred_date: '2014-12-19'
  top_k: 20
```

---

## 训练与推理流程

### 数据流（线性）

```
raw csv
   ↓ load_data
preprocess_data                ← PadLabelEncoder + 时间特征 + FeatureEngineer.fit
   ↓
build_user_sequences           ← ≤ train_end_date 才进入用户表示
   ↓
sample_interactions            ← 正样本=该日购买；负样本=按策略采
   ↓
RecommendationDataset
   ↓
build_dataloader → Lightning Trainer.fit
```

### 推理流（零泄漏）

```
preprocess_data
   ↓
build_inference_candidates     ← (cutoff - history_days, cutoff] 内 ∩ P
   ↓
DeepRecommender.predict        ← 输出 [B] sigmoid 分数
   ↓
create_submission              ← 去重 + top-k + 强制 str 类型
   ↓
tianchi_mobile_recommendation_predict.csv
```

### 评估流（带真实标签）

```
prepare_inference_data(cutoff = target_date - 1 day)
   ↓
predict → submission (top-k per user)
   ↓
actual_purchases_on(target_date)   ← behavior_type==4 ∩ P
   ↓
set_precision_recall_f1(submission, ground_truth)
```

---

## 竞赛规范对齐

依据 [Tianchi 移动推荐算法 (231522)](https://tianchi.aliyun.com/competition/entrance/231522/information)：

| 规范要求 | 实现位置 |
|---|---|
| 用户行为表 D：`user_id, item_id, behavior_type, user_geohash, item_category, time` | `DataProcessor.load_data` |
| 商品子集表 P：`item_id, item_geohash, item_category` | 同上；`item_category → category` 重命名 |
| `behavior_type` ∈ {1 浏览, 2 收藏, 3 加购, 4 购买} | `PURCHASE_BEHAVIOR = 4`；序列嵌入 `num_behaviors = 5`（0=PAD） |
| 训练数据 11.18 ~ 12.18，预测 12.19 | `training.train_end_date` / `pred_date` |
| 推荐必须落在商品子集 P 内 | `build_inference_candidates` 强制 `item_id ∈ _all_item_ids`；测试覆盖 |
| 推理不能"偷看"预测日 | `prepare_inference_data` 只用 `time ≤ cutoff` 的历史；反泄漏测试覆盖 |
| 输出文件名 `tianchi_mobile_recommendation_predict.csv` | `main.py predict` 写入该名称，UTF-8 |
| 输出列 `user_id, item_id`（string、去重） | `create_submission` 强制 `astype(str) + drop_duplicates` |
| 评测：集合 P/R/F1 | `src.evaluation.set_precision_recall_f1` + `main.py evaluate` 子命令 |

---

## 评估指标

### 训练时（per-row 二分类）

通过 `torchmetrics` 在每个验证 epoch 计算：

- `val_loss`（BCE）
- `val_acc` — 阈值 0.5 的准确率
- `val_precision`、`val_recall`、`val_f1`
- `val_auroc`（验证集需含正负样本，否则跳过）

训练侧记录 `train_bce`；启用 InfoNCE 时额外记录 `train_infonce`。

### 竞赛 F1（set-based）

> 这是 Tianchi 提交的**唯一评测指标**，与上面的 per-row F1 含义不同。

```
PredictionSet = {(user_id, item_id) | 模型推荐}
ReferenceSet  = {(user_id, item_id) | 目标日真实购买 ∩ P}

Precision = |PredictionSet ∩ ReferenceSet| / |PredictionSet|
Recall    = |PredictionSet ∩ ReferenceSet| / |ReferenceSet|
F1        = 2 · P · R / (P + R)
```

通过 `python main.py evaluate` 直接报告。

---

## 负采样策略

- **正样本**：目标日 `behavior_type == 4`（购买）
- **负样本**：该用户当日未购买的随机商品
- **采样比例**：`negative_sampling_ratio`（每个正样本对应几个负样本）
- **采样分布**：
  - `uniform`：所有商品等概率
  - `popularity`：按全局行为计数加权，使用 word2vec 风格平滑

    $$p(\text{item}) \propto (\text{count} + 1)^{\alpha}$$

    `alpha = 0` 等价 uniform；`alpha = 1` 纯热门；默认 `alpha = 0.75`
- **可选 InfoNCE**：开启 `use_in_batch_negatives` 后，将同一 batch 中其他正样本作为隐式负样本，按温度归一化的余弦相似度做 softmax 交叉熵；与 BCE 主损失按 `infonce_weight` 加权融合

---

## 数值特征

`FeatureEngineer` 从训练窗口（≤ `train_end_date`）的行为中统计每个用户/商品的稠密特征。所有特征经 `StandardScaler` 标准化，并截断到 ±5σ。

### 用户特征（11 维）

| 特征 | 说明 |
|---|---|
| `total_actions` | 行为总数 |
| `unique_items` | 涉及的不同商品数 |
| `unique_categories` | 涉及的不同类目数 |
| `behavior_1_ratio` ~ `behavior_4_ratio` | 四种 behavior 各自占比 |
| `avg_hour` | 行为平均发生小时 |
| `weekend_ratio` | 周末行为占比 |
| `action_days` | 活跃天数 |
| `avg_time_diff_hours` | 相邻行为平均间隔（小时） |

### 商品特征（7 维）

| 特征 | 说明 |
|---|---|
| `total_actions` | 被交互总次数 |
| `unique_users` | 涉及的不同用户数 |
| `behavior_1_ratio` ~ `behavior_4_ratio` | 四种 behavior 各自占比（含转化率） |
| `user_diversity` | `unique_users / total_actions` |

冷启动用户/商品的特征向量为零（标准化后即均值），不污染推理。

---

## 测试

```bash
# 全部测试
pytest tests/ -v

# 单类别
pytest tests/test_inference.py -v       # 反泄漏推理
pytest tests/test_evaluation.py -v      # 集合 F1
pytest tests/test_submission.py -v      # 提交格式

# 覆盖率
pytest tests/ --cov=src --cov-report=term-missing
```

| 测试套件 | 覆盖范围 |
|---|---|
| `test_smoke.py` | 端到端：数据 → 模型 → 训练一个 epoch → 检查点 → 推理 |
| `test_negative_sampling.py` | 正负样本不冲突；popularity 加权倾向热门；非法策略报错 |
| `test_feature_engineering.py` | 维度稳定、表形状、Dataset 输出、时间窗口生效 |
| `test_infonce.py` | 默认关闭、损失正且有限、端到端启用 |
| `test_submission.py` | 列名、string 类型、去重、top-k |
| `test_evaluation.py` | 完美匹配/无重叠/部分重叠、DataFrame、dtype 强制、空集、缺列报错 |
| `test_inference.py` | 注入截止日之后 sentinel 行验证零泄漏；候选集 ⊆ P；占位标签 |

**当前状态**：97 通过，CPU 上 ~18s，**99.1% 行+分支覆盖率**。

---

## 开发

```bash
# 代码风格检查
pip install ruff
ruff check src/ tests/
ruff format src/ tests/         # 自动格式化

# 启动 TensorBoard（如已安装）
tensorboard --logdir logs/
```

### CI

GitHub Actions 在每次推送和 PR 上运行 pytest（Python 3.10/3.11/3.12 矩阵 + CPU torch 轮子），并将 Python 3.11 这一格的覆盖率上传到 Codecov。

要激活 Codecov 徽章：在 [codecov.io](https://codecov.io) 启用该仓库后，把 `CODECOV_TOKEN` 加到 GitHub Actions secrets（公开仓库通常可免 token）。

---

## 常见问题

**Q：GPU 不可用 / CUDA 缺失会怎样？**
A：把 `device.accelerator` 设为 `cpu`，或保留默认 `auto` 让 Trainer 自动检测。`gpu` 模式在 CPU 主机上会自动降级，并把 `precision` 调回 32。

**Q：没装 `tensorboard` 会报错吗？**
A：不会。训练自动回退到 `CSVLogger`，日志写入 `logs/csv_logs/`。

**Q：Windows 下 DataLoader 多进程崩溃？**
A：默认 `system.num_workers = 0`。如需开多进程加载，请确保在 `if __name__ == '__main__':` 守卫下运行。

**Q：训练时 OOM？**
A：调小 `model.training.batch_size`；GPU 上把 `device.precision` 设为 16 启用混合精度；`model.training.optimization.accumulate_grad_batches` 大于 1 可做梯度累积。

**Q：评估时 F1 一直是 0？**
A：检查 `actual_purchases_on(target_date)` 是否返回非空。当 `target_date` 上没有任何在 P 内的购买行为时，集合 F1 为 0 是正确的。

---

## 更新日志

### v0.6.0 (2026-05-28)
- 覆盖率 77% → **99.1%**（97 个测试，CPU 上 ~18s）
- 新增 `test_utils.py` / `test_trainer.py` / `test_edge_cases.py`
- 修复 `get_categorical_dims` 在 PadLabelEncoder 未 fit 时的 `len(None)` bug
- 合成数据 fixture 改为更接近官方 schema（保留 `item_category` 列）

### v0.5.0 (2026-05-28)
- 与 Tianchi 移动推荐算法竞赛规范完全对齐
- 提交文件名改为 `tianchi_mobile_recommendation_predict.csv`，UTF-8，强制 string + 去重
- 拆分推理路径：`prepare_inference_data` 严格不偷看预测日
- 新增 `evaluate` CLI 子命令报告竞赛集合 F1
- 新增 `src/evaluation.set_precision_recall_f1` 工具
- GitHub Actions CI + Codecov 集成

### v0.4.0
- 验证集新增 P/R/F1/AUROC 指标
- 负采样支持 `popularity` 策略
- 集成 `FeatureEngineer`：用户/商品数值特征自动进入两塔
- `PadLabelEncoder` 把 PAD 固定到索引 0
- 可选 InfoNCE in-batch 对比损失

### v0.3.0
- 全面重写数据/模型/训练流水线，使各层接口一致
- 新增端到端冒烟测试
- 默认 CPU 友好；GPU 不可用时静默回退
- `BCEWithLogitsLoss` 取代 `Sigmoid + BCE`

### v0.2.1
- 迁移到深度学习架构
- 添加 GPU 训练支持

### v0.1.0
- 初始版本

---

## 贡献

欢迎 Issue 和 PR。提交前请确保：

1. `pytest tests/` 全部通过
2. `ruff check src/ tests/` 无警告
3. 涉及行为变化的修改附带新测试
4. commit message 用 [Conventional Commits](https://www.conventionalcommits.org/) 格式（`feat:` / `fix:` / `refactor:` / `docs:` / `test:` / `chore:`）

---

## 作者与许可证

**作者**：[綦子宽](https://github.com/zikuanqi)

**许可证**：[MIT License](LICENSE)

## 参考

- [天池 移动推荐算法 竞赛主页 (231522)](https://tianchi.aliyun.com/competition/entrance/231522/information)
- [PyTorch 文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch Lightning 文档](https://lightning.ai/docs/pytorch/stable/)
- [torchmetrics 文档](https://lightning.ai/docs/torchmetrics/stable/)
- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (NIPS 2013) — word2vec 的 negative sampling 平滑指数 `alpha = 0.75`
- van den Oord et al., *Representation Learning with Contrastive Predictive Coding* (2018) — InfoNCE 损失
