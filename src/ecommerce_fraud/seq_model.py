"""Transformer encoder over a transaction and its 31 earlier same-uid transactions.

Batch dict (all tensors, T = L + 1 tokens, the current transaction is the LAST token):
    cat  long  [B, T, K]   categorical codes (0 = missing / masked)
    num  float [B, T, F]   normalized numerics (NaN already 0)
    miss float [B, T, F]   1 = missing or masked
    gap  float [B, T]      log1p(hours before the current transaction), 0 for the current token
    pad  bool  [B, T]      True = padding token (hist == -1)
No label is ever part of a batch.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SeqEncoder(nn.Module):
    def __init__(self, num_features: int, cat_cardinalities, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        cards = np.maximum(np.asarray(cat_cardinalities, dtype=np.int64), 2)
        self.num_features = int(num_features)
        self.n_cat = len(cards)
        self.d_model = d_model
        offsets = np.concatenate([[0], np.cumsum(cards)[:-1]])
        self.register_buffer("cat_offsets", torch.as_tensor(offsets, dtype=torch.long), persistent=False)
        # one table holding the K per-field embeddings (each of dim d_model); rows addressed with offsets
        self.cat_emb = nn.Embedding(int(cards.sum()), d_model)
        self.num_proj = nn.Linear(2 * self.num_features, d_model)
        self.time_proj = nn.Linear(1, d_model)
        self.cur_emb = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)

    def forward_tokens(self, batch) -> torch.Tensor:
        """Hidden states of all tokens, [B, T, d_model]."""
        cat, num, miss, gap, pad = (batch[k] for k in ("cat", "num", "miss", "gap", "pad"))
        x = self.cat_emb(cat + self.cat_offsets).sum(dim=2)
        x = x + self.num_proj(torch.cat([num, miss], dim=-1))
        x = x + self.time_proj(gap.unsqueeze(-1))
        is_cur = torch.zeros(pad.shape, dtype=torch.long, device=pad.device)
        is_cur[:, -1] = 1
        x = x + self.cur_emb(is_cur)
        h = self.transformer(x, src_key_padding_mask=pad)
        return self.out_norm(h)

    def forward(self, batch) -> torch.Tensor:
        """Final hidden state of the current token, [B, d_model]."""
        return self.forward_tokens(batch)[:, -1]

    def embed(self, batch) -> torch.Tensor:
        return self.forward(batch)


class FraudHead(nn.Module):
    def __init__(self, d_model: int = 64, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


class MaskedFieldHead(nn.Module):
    """Per categorical field a Linear to its cardinality; per numeric field a scalar regression."""

    def __init__(self, num_features: int, cat_cardinalities, d_model: int = 64):
        super().__init__()
        cards = np.maximum(np.asarray(cat_cardinalities, dtype=np.int64), 2)
        self.cat_heads = nn.ModuleList([nn.Linear(d_model, int(c)) for c in cards])
        self.num_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_features)])

    def cat_logits(self, k: int, h):
        return self.cat_heads[k](h)

    def num_pred(self, h):
        """All numeric field predictions, [..., F]."""
        return torch.cat([m(h) for m in self.num_heads], dim=-1)


class SeqClassifier(nn.Module):
    def __init__(self, encoder: SeqEncoder, head: FraudHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, batch):
        return self.head(self.encoder(batch))

    def embed(self, batch):
        return self.encoder.embed(batch)
