"""Two-tower + sequence recommender."""

from __future__ import annotations

from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from src.models.layer import PositionalEncoding, build_mlp


class DeepRecommender(pl.LightningModule):
    """Two-tower + sequence model.

    - User tower: embedding(user_id) → MLP
    - Item tower: embedding(item_id) ⊕ embedding(category_id) → MLP
    - Sequence tower: embedding(seq_items) ⊕ embedding(seq_behaviors) →
      positional encoding → Transformer encoder → masked mean-pool
    - Concatenate the three representations and project to a logit.

    The model emits a logit (not a probability) and uses
    `BCEWithLogitsLoss`, which is numerically more stable than
    Sigmoid + BCE.
    """

    def __init__(self, config: dict, vocab_sizes: Dict[str, int]):
        super().__init__()
        # Lightning saves these so checkpoints are self-describing.
        self.save_hyperparameters({'config': config, 'vocab_sizes': vocab_sizes})
        self.config = config
        self.vocab_sizes = vocab_sizes

        arch = config['model']['architecture']
        d_emb = int(arch['embedding_dim'])
        hidden_dims = list(arch['hidden_dims'])
        dropout = float(arch['dropout'])
        n_heads = int(arch['num_attention_heads'])
        n_layers = int(arch['num_transformer_layers'])
        ff_dim = int(arch.get('transformer_ff_dim', d_emb * 4))

        seq_cfg = config['data']['features']['sequence']
        behavior_vocab = int(seq_cfg['num_behaviors'])
        behavior_dim = int(seq_cfg['behavior_dim'])

        # +1 on every embedding size: id 0 is the explicit PAD sentinel.
        self.user_embedding = nn.Embedding(vocab_sizes['user_id'] + 1, d_emb, padding_idx=0)
        self.item_embedding = nn.Embedding(vocab_sizes['item_id'] + 1, d_emb, padding_idx=0)
        self.category_embedding = nn.Embedding(vocab_sizes.get('category', 1) + 1, d_emb, padding_idx=0)
        self.behavior_embedding = nn.Embedding(behavior_vocab, behavior_dim, padding_idx=0)

        # User tower: just user_id → hidden
        self.user_tower = build_mlp(d_emb, hidden_dims, dropout=dropout)

        # Item tower: item ⊕ category → hidden
        self.item_tower = build_mlp(d_emb * 2, hidden_dims, dropout=dropout)

        # Sequence tower: project item⊕behavior into d_model, then attend
        d_model = d_emb + behavior_dim
        self.seq_input_proj = nn.Linear(d_model, d_emb)
        self.positional_encoding = PositionalEncoding(d_emb, max_len=seq_cfg['max_length'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.sequence_tower = build_mlp(d_emb, hidden_dims, dropout=dropout)

        # Fusion head
        fusion_in = hidden_dims[-1] * 3
        self.head = build_mlp(fusion_in, [fusion_in // 2], dropout=dropout, output_dim=1)

        self.loss_fn = nn.BCEWithLogitsLoss()

        # Validation metrics. torchmetrics handles aggregation across the
        # epoch so we don't have to accumulate batches ourselves.
        self.val_acc = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_auroc = BinaryAUROC()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def encode_sequence(self,
                        seq_items: torch.Tensor,
                        seq_behaviors: torch.Tensor,
                        seq_mask: torch.Tensor) -> torch.Tensor:
        """[B, L] ints + bool mask → [B, d_emb] pooled representation."""
        item_emb = self.item_embedding(seq_items)              # [B, L, d_emb]
        beh_emb = self.behavior_embedding(seq_behaviors)        # [B, L, behavior_dim]
        seq = torch.cat([item_emb, beh_emb], dim=-1)            # [B, L, d_emb+behavior_dim]
        seq = self.seq_input_proj(seq)                          # [B, L, d_emb]
        seq = self.positional_encoding(seq)

        # If every position is padding (cold-start user with empty seq),
        # the transformer would NaN out — fall back to zeros.
        all_pad = seq_mask.all(dim=1, keepdim=True)             # [B, 1]
        safe_mask = seq_mask.clone()
        safe_mask[all_pad.squeeze(1)] = False  # let attention run, then we'll zero it out

        encoded = self.sequence_encoder(seq, src_key_padding_mask=safe_mask)  # [B, L, d_emb]

        # Masked mean over non-pad positions
        keep = (~seq_mask).float().unsqueeze(-1)                # [B, L, 1]
        denom = keep.sum(dim=1).clamp(min=1.0)                  # [B, 1]
        pooled = (encoded * keep).sum(dim=1) / denom            # [B, d_emb]
        pooled = pooled.masked_fill(all_pad, 0.0)               # zero out cold-start users
        return pooled

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_repr = self.user_tower(self.user_embedding(batch['user_id']))

        item_emb = self.item_embedding(batch['item_id'])
        cat_emb = self.category_embedding(batch['category_id'])
        item_repr = self.item_tower(torch.cat([item_emb, cat_emb], dim=-1))

        seq = batch['sequence']
        seq_repr_raw = self.encode_sequence(seq['items'], seq['behaviors'], seq['mask'])
        seq_repr = self.sequence_tower(seq_repr_raw)

        fused = torch.cat([user_repr, item_repr, seq_repr], dim=-1)
        return self.head(fused).squeeze(-1)                     # [B] logits

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch['label'])
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch['label']
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
        labels_int = labels.int()

        self.val_acc.update(probs, labels_int)
        self.val_precision.update(probs, labels_int)
        self.val_recall.update(probs, labels_int)
        self.val_f1.update(probs, labels_int)
        self.val_auroc.update(probs, labels_int)

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        # Compute epoch-level metrics (torchmetrics aggregates internally).
        # AUROC requires both classes in the epoch; skip cleanly otherwise.
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_precision', self.val_precision.compute(), prog_bar=True)
        self.log('val_recall', self.val_recall.compute(), prog_bar=True)
        self.log('val_f1', self.val_f1.compute(), prog_bar=True)
        try:
            self.log('val_auroc', self.val_auroc.compute(), prog_bar=True)
        except (ValueError, RuntimeError):
            pass
        for m in (self.val_acc, self.val_precision, self.val_recall,
                  self.val_f1, self.val_auroc):
            m.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return torch.sigmoid(self(batch))

    def configure_optimizers(self):
        train_cfg = self.config['model']['training']
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(train_cfg['learning_rate']),
            weight_decay=float(train_cfg.get('weight_decay', 0.0)),
        )
        sched_cfg = train_cfg.get('scheduler', {})
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get('T_max', train_cfg.get('num_epochs', 10))),
            eta_min=float(sched_cfg.get('eta_min', 1e-6)),
        )
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
