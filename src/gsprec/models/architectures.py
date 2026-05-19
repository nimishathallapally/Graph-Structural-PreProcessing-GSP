"""
GNN model architectures for GSP-based recommendation.

Implements:
  - LightGCN   – simplified linear propagation (He et al. 2020)
  - GraphSAGE  – inductive mean/max aggregation
  - GAT        – attention-based, multi-head

All models expose:
    forward(edge_index) -> z  (num_nodes × out_dim)

Factory
-------
    get_model(name, config) -> nn.Module

where name ∈ {"lightgcn", "graphsage", "gat"}
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from torch.utils.checkpoint import checkpoint as _grad_checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, SAGEConv
    from torch_geometric.nn.conv import MessagePassing
    from torch_geometric.utils import degree
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "torch_geometric is required. Install with:\n"
        "  pip install torch-geometric\n"
        f"Original error: {exc}"
    ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_embedding(emb: nn.Embedding, std: float = 0.01) -> None:
    """Xavier-like normal initialisation for the embedding table."""
    nn.init.normal_(emb.weight, mean=0.0, std=std)


# ─────────────────────────────────────────────────────────────────────────────
# LightGCN
# ─────────────────────────────────────────────────────────────────────────────

class LightGCNConv(MessagePassing):
    """
    A single LightGCN propagation layer.

    z_new(v) = Σ_{u∈N(v)}  e_u / sqrt(deg(v) * deg(u))

    No learnable weight matrix, no activation.
    Per the original paper (He et al. 2020), self-loops are NOT added.
    """
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        # No self-loops: LightGCN propagates only through neighbours (He et al. 2020)
        row, col = edge_index
        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.unsqueeze(-1) * x_j


class LightGCNRecommender(nn.Module):
    """
    LightGCN with K stacked propagation layers.

    Final embedding = mean of layer-0 … layer-K representations
    (as in the original paper).
    """
    def __init__(
        self,
        num_nodes: int,
        emb_dim: int = 64,
        num_layers: int = 3,
        **kwargs,  # absorb unused hidden_dim / out_dim
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_nodes, emb_dim)
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])
        _init_embedding(self.embedding)

    @property
    def out_dim(self) -> int:
        return self.embedding.embedding_dim

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        e0 = self.embedding.weight
        embeddings = [e0]
        x = e0
        for conv in self.convs:
            x = conv(x, edge_index)
            embeddings.append(x)
        # Layer-aggregated embedding (mean pooling over all layers)
        z = torch.stack(embeddings, dim=0).mean(dim=0)
        return z


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGERecommender(nn.Module):
    """
    Two-layer GraphSAGE with mean aggregation.

    Supports mini-batch inference: pass a subset of edge_index per call.
    """
    def __init__(
        self,
        num_nodes: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.embedding = nn.Embedding(num_nodes, emb_dim)
        self.conv1 = SAGEConv(emb_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        _init_embedding(self.embedding)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.embedding.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# GAT (multi-head)
# ─────────────────────────────────────────────────────────────────────────────

class GATRecommender(nn.Module):
    """
    Two-layer Graph Attention Network with multi-head attention.

    Layer 1: 4 heads concatenated → hidden_dim
    Layer 2: 1 head averaged    → out_dim

    For large graphs (>100 K nodes by default), gradient checkpointing is
    auto-enabled during training to avoid OOM from storing edge-level message
    tensors (E × heads × head_dim) for the backward pass.
    """
    def __init__(
        self,
        num_nodes: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
        use_gradient_checkpointing: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        # Auto-enable gradient checkpointing for very large graphs (>100 K nodes).
        # Recomputes intermediate activations during backward instead of storing
        # them, trading ~2× compute for a ~4× reduction in peak activation memory.
        if use_gradient_checkpointing is None:
            self.use_checkpoint = num_nodes > 100_000
        else:
            self.use_checkpoint = use_gradient_checkpointing
        self.embedding = nn.Embedding(num_nodes, emb_dim)
        # Layer 1: concat mode → output is heads * (hidden_dim // heads)
        head_dim = max(hidden_dim // heads, 1)
        self.conv1 = GATConv(emb_dim, head_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(head_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
        _init_embedding(self.embedding)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.embedding.weight
        if self.use_checkpoint and self.training:
            x = _grad_checkpoint(self.conv1, x, edge_index, use_reentrant=False)
        else:
            x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.use_checkpoint and self.training:
            x = _grad_checkpoint(self.conv2, x, edge_index, use_reentrant=False)
        else:
            x = self.conv2(x, edge_index)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass for model construction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    emb_dim: int = 64
    hidden_dim: int = 128
    out_dim: int = 64
    num_layers: int = 3          # used by LightGCN
    heads: int = 4               # used by GAT
    dropout: float = 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_REGISTRY = {
    "lightgcn": LightGCNRecommender,
    "graphsage": GraphSAGERecommender,
    "gat": GATRecommender,
    # legacy aliases kept for backward compat with existing runner
    "gcn": GraphSAGERecommender,
}


def get_model(name: str, num_nodes: int, config: ModelConfig) -> nn.Module:
    """
    Factory function: build a GNN recommender model by name.

    Parameters
    ----------
    name
        One of ``"lightgcn"``, ``"graphsage"``, ``"gat"``.
    num_nodes
        Total number of nodes in the bipartite graph (U + I).
    config
        :class:`ModelConfig` with architecture hyper-parameters.

    Returns
    -------
    nn.Module
        The requested model with shared embedding initialisation.
    """
    key = name.lower().strip()
    cls = _MODEL_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Choose from: {sorted(_MODEL_REGISTRY)}"
        )
    return cls(
        num_nodes=num_nodes,
        emb_dim=config.emb_dim,
        hidden_dim=config.hidden_dim,
        out_dim=config.out_dim,
        num_layers=config.num_layers,
        heads=config.heads,
        dropout=config.dropout,
    )


def init_item_embeddings_from_features(
    model: nn.Module,
    feat_matrix: "np.ndarray",
    item_offset: int,
    emb_dim: int,
    blend_alpha: float = 0.8,
) -> None:
    """Initialise the item slice of a model's embedding table from semantic features.

    Projects ``feat_matrix`` (shape ``num_items × feat_dim``) down to
    ``emb_dim`` via TruncatedSVD, L2-normalises each row, scales to the
    same magnitude as the random initialisation, and blends with the
    existing random weights:

        new_weight = blend_alpha * semantic + (1 - blend_alpha) * random

    Parameters
    ----------
    model
        A GNN model with an ``embedding`` (``nn.Embedding``) attribute.
    feat_matrix
        ``(num_items, feat_dim)`` float32 array of semantic features.
    item_offset
        Index into the embedding table where item embeddings start
        (typically ``num_users`` for baseline, ``num_super`` for GSP).
    emb_dim
        Embedding dimension (must match ``model.embedding.embedding_dim``).
    blend_alpha
        Weight of semantic component vs. random initialisation.
        0.0 = pure random, 1.0 = pure semantic.  Default 0.8.
    """
    import numpy as np

    n_items, feat_dim = feat_matrix.shape

    # Project to emb_dim
    if feat_dim >= emb_dim:
        try:
            from sklearn.decomposition import TruncatedSVD  # type: ignore
            from sklearn.preprocessing import normalize     # type: ignore
            max_comp = min(n_items - 1, feat_dim - 1, emb_dim)
            actual = max(max_comp, 1)
            svd = TruncatedSVD(n_components=actual, random_state=42)
            projected = svd.fit_transform(feat_matrix.astype(np.float64)).astype(np.float32)
            projected = normalize(projected, norm="l2", axis=1)
            if actual < emb_dim:
                pad = np.zeros((n_items, emb_dim - actual), dtype=np.float32)
                projected = np.concatenate([projected, pad], axis=1)
            else:
                projected = projected[:, :emb_dim]
        except Exception:
            # fallback: zero-pad or truncate
            projected = np.zeros((n_items, emb_dim), dtype=np.float32)
            projected[:, :min(feat_dim, emb_dim)] = feat_matrix[:, :min(feat_dim, emb_dim)]
    else:
        # feat_dim < emb_dim: zero-pad on the right
        projected = np.zeros((n_items, emb_dim), dtype=np.float32)
        projected[:, :feat_dim] = feat_matrix

    # Scale to match the random init magnitude (std ≈ 0.01)
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected = projected / np.maximum(norms, 1e-8) * 0.01

    semantic_tensor = torch.from_numpy(projected)
    with torch.no_grad():
        existing = model.embedding.weight.data[item_offset: item_offset + n_items]
        model.embedding.weight.data[item_offset: item_offset + n_items] = (
            blend_alpha * semantic_tensor + (1.0 - blend_alpha) * existing
        )
    print(
        f"[SemanticInit] Initialised {n_items:,} item embeddings from "
        f"{feat_dim}-dim features → {emb_dim}-dim (alpha={blend_alpha})"
    )
