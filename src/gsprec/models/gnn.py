from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "torch_geometric is required for GAT/GraphSAGE/GCN. "
        "Make sure your environment has a compatible PyTorch + PyG install."
    ) from e


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_jsonl(path: str, record: Dict) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8", buffering=1) as f:
        f.write(json.dumps(record) + "\n")


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return rmse, mae


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    relevance = relevance[:k].astype(np.float64)
    if relevance.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevance.size + 2))
    dcg = float(np.sum(relevance * discounts))
    ideal = np.sort(relevance)[::-1]
    idcg = float(np.sum(ideal * discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg


def precision_recall_at_k(hits: np.ndarray, k: int, num_pos: int) -> Tuple[float, float]:
    hits_k = int(np.sum(hits[:k]))
    precision = float(hits_k / max(k, 1))
    recall = float(hits_k / max(num_pos, 1))
    return precision, recall


@dataclass
class RankingEvalConfig:
    k: int = 10
    num_negatives: int = 99
    seed: int = 42


def evaluate_ranking_from_embeddings(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    test_positives: Dict[int, List[int]],
    seen_positives: Optional[Dict[int, set]] = None,
    config: RankingEvalConfig = RankingEvalConfig(),
) -> Dict[str, float]:
    rng = np.random.default_rng(config.seed)
    num_items = item_emb.shape[0]

    ndcgs: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    hit_rates: List[float] = []

    for user_id, pos_items in test_positives.items():
        if not pos_items:
            continue
        seen = seen_positives.get(user_id, set()) if seen_positives is not None else set()
        seen = set(seen)
        pos_items_unique = list(dict.fromkeys(pos_items))
        seen.update(pos_items_unique)

        negatives: List[int] = []
        attempts = 0
        while len(negatives) < config.num_negatives and attempts < config.num_negatives * 20:
            cand = int(rng.integers(0, num_items))
            if cand in seen:
                attempts += 1
                continue
            negatives.append(cand)
            seen.add(cand)
            attempts += 1

        candidates = pos_items_unique + negatives
        labels = np.zeros(len(candidates), dtype=np.int64)
        labels[: len(pos_items_unique)] = 1

        u = user_emb[user_id]
        scores = item_emb[candidates] @ u
        order = np.argsort(scores)[::-1]
        ranked_labels = labels[order]

        ndcgs.append(ndcg_at_k(ranked_labels, config.k))
        p, r = precision_recall_at_k(ranked_labels, config.k, num_pos=len(pos_items_unique))
        precisions.append(p)
        recalls.append(r)
        hit_rates.append(1.0 if int(np.sum(ranked_labels[:config.k])) >= 1 else 0.0)

    return {
        f"NDCG@{config.k}": float(np.mean(ndcgs) if ndcgs else 0.0),
        f"Precision@{config.k}": float(np.mean(precisions) if precisions else 0.0),
        f"Recall@{config.k}": float(np.mean(recalls) if recalls else 0.0),
        f"HitRate@{config.k}": float(np.mean(hit_rates) if hit_rates else 0.0),
        "UsersEvaluated": float(len(ndcgs)),
    }


class GATRecommender(torch.nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int = 64, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, emb_dim)
        self.conv1 = GATConv(emb_dim, hidden_dim, heads=2, concat=False)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=2, concat=False)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.embedding.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class SAGERecommender(torch.nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int = 64, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, emb_dim)
        self.conv1 = SAGEConv(emb_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.embedding.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GCNRecommender(torch.nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int = 64, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, emb_dim)
        self.conv1 = GCNConv(emb_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.embedding.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def predict_scores(z: torch.Tensor, user_nodes: torch.Tensor, item_nodes: torch.Tensor) -> torch.Tensor:
    return (z[user_nodes] * z[item_nodes]).sum(dim=-1)


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 131072
    neg_ratio: int = 2
    rating_loss_weight: float = 0.2
    emb_l2_weight: float = 1e-6
    log_jsonl_path: str = "outputs/training_metrics.jsonl"
    checkpoint_dir: str = "outputs/checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    train_user_nodes: torch.Tensor,
    train_item_nodes: torch.Tensor,
    train_ratings: torch.Tensor,
    config: TrainConfig,
    run_name: str,
    eval_callback=None,
) -> Dict[str, float]:
    device = torch.device(config.device)
    model = model.to(device)
    edge_index = edge_index.to(device)
    train_user_nodes = train_user_nodes.to(device)
    train_item_nodes = train_item_nodes.to(device)
    train_ratings = train_ratings.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
        eta_min=1e-5,
    )

    ckpt_root = os.path.join(config.checkpoint_dir, run_name)
    _ensure_dir(ckpt_root)

    best_loss = float("inf")
    best_epoch = 0
    best_record: Dict[str, float] = {}
    epoch_times: List[float] = []
    peak_mems_mb: List[float] = []

    for epoch in range(1, config.epochs + 1):
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        z = model(edge_index)

        # Train on a shuffled subset each epoch to avoid overfitting to static full-batch updates.
        num_pos_total = int(train_user_nodes.numel())
        if num_pos_total == 0:
            raise RuntimeError("No training edges found after preprocessing.")
        sample_size = min(num_pos_total, int(config.batch_size))
        perm = torch.randperm(num_pos_total, device=device)[:sample_size]

        pos_u = train_user_nodes[perm]
        pos_i = train_item_nodes[perm]
        pos_scores = predict_scores(z, pos_u, pos_i)

        # Negative sampling over item-node id range.
        item_min = int(torch.min(train_item_nodes).item())
        item_max = int(torch.max(train_item_nodes).item())
        neg_count = sample_size * max(int(config.neg_ratio), 1)

        neg_u = pos_u.repeat_interleave(max(int(config.neg_ratio), 1))
        neg_i = torch.randint(
            low=item_min,
            high=item_max + 1,
            size=(neg_count,),
            device=device,
            dtype=pos_i.dtype,
        )
        neg_scores = predict_scores(z, neg_u, neg_i)

        # Ranking-aware BCE objective.
        logits = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(pos_scores, device=device),
                torch.zeros_like(neg_scores, device=device),
            ],
            dim=0,
        )
        ranking_loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Auxiliary rating regression on positives; predictions constrained to [1,5].
        y_true = torch.clamp(train_ratings[perm], 1.0, 5.0)
        y_true_norm = (y_true - 1.0) / 4.0
        y_pred_norm = torch.sigmoid(pos_scores)
        rating_loss = F.mse_loss(y_pred_norm, y_true_norm)

        emb_reg = (z.pow(2).mean())
        loss = ranking_loss + config.rating_loss_weight * rating_loss + config.emb_l2_weight * emb_reg
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_time_s = float(time.perf_counter() - t0)
        epoch_times.append(epoch_time_s)
        gpu_mem_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if (torch.cuda.is_available() and device.type == "cuda")
            else 0
        )
        peak_mems_mb.append(float(gpu_mem_bytes / 1e6))

        record = {
            "run": run_name,
            "epoch": epoch,
            "loss_mse": float(loss.item()),
            "ranking_loss": float(ranking_loss.item()),
            "rating_loss": float(rating_loss.item()),
            "time_s": epoch_time_s,
            "gpu_mem_mb": float(gpu_mem_bytes / 1e6),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        if eval_callback is not None:
            try:
                eval_metrics = eval_callback(model)
                record.update(eval_metrics)
            except Exception as e:
                record["eval_error"] = str(e)

        _append_jsonl(config.log_jsonl_path, record)
        print(
            f"[{run_name}] epoch={epoch} loss={record['loss_mse']:.4f} "
            f"time={record['time_s']:.2f}s mem={record['gpu_mem_mb']:.1f}MB"
        )

        ckpt_path = os.path.join(ckpt_root, f"epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": record,
            },
            ckpt_path,
        )

        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": record},
                os.path.join(ckpt_root, "best.pt"),
            )
            best_record = record

    avg_time = float(np.mean(epoch_times) if epoch_times else 0.0)
    max_mem_mb = float(np.max(peak_mems_mb) if peak_mems_mb else 0.0)

    summary: Dict[str, float] = {
        "best_epoch": float(best_epoch),
        "best_loss_mse": float(best_loss),
        "avg_epoch_time_s": avg_time,
        "max_gpu_mem_mb": max_mem_mb,
    }
    for k, v in best_record.items():
        if k in {"run", "epoch", "loss_mse", "time_s", "gpu_mem_mb"}:
            continue
        summary[k] = v

    _append_jsonl(config.log_jsonl_path, {"summary": True, "run": run_name, **summary})
    return summary
