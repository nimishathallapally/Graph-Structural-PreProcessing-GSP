"""
Evaluation metrics for GSP-based recommender system.

All metric computations are fully vectorised (numpy batch ops, no Python loops
over users in the hot path).

Ranking metrics
---------------
- NDCG@K
- Precision@K
- Recall@K

Regression metrics
------------------
- RMSE, MAE, MSE

Efficiency metrics
------------------
- Speedup ratio
- Training / inference time comparison

Evaluator
---------
BatchedEvaluator.evaluate(model, edge_index, ...) -> dict[str, float]
  Runs full-graph inference once, then computes all ranking metrics in batch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Regression metrics (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """RMSE, MAE, MSE – all in one pass, fully vectorised."""
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    diff = y_true - y_pred
    mse = float(np.mean(diff ** 2))
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE":  float(np.mean(np.abs(diff))),
        "MSE":  mse,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-user ranking helpers (vectorised over K)
# ─────────────────────────────────────────────────────────────────────────────

def _dcg(relevance: np.ndarray, k: int) -> float:
    rel = relevance[:k].astype(np.float64)
    if rel.size == 0:
        return 0.0
    disc = 1.0 / np.log2(np.arange(2, rel.size + 2, dtype=np.float64))
    return float(np.dot(rel, disc))


def _ndcg(relevance: np.ndarray, k: int) -> float:
    dcg = _dcg(relevance, k)
    ideal = np.sort(relevance)[::-1]
    idcg = _dcg(ideal, k)
    return 0.0 if idcg == 0.0 else dcg / idcg


def _precision_recall(hits: np.ndarray, k: int, num_pos: int) -> Tuple[float, float]:
    h = int(np.sum(hits[:k]))
    return float(h / max(k, 1)), float(h / max(num_pos, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Batched ranking evaluator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    k: int = 10
    num_negatives: int = 99
    seed: int = 42


def evaluate_ranking_from_embeddings(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    test_positives: Dict[int, List[int]],
    seen_positives: Optional[Dict[int, Set[int]]] = None,
    config: EvalConfig = EvalConfig(),
) -> Dict[str, float]:
    """
    Evaluate ranking metrics for all test users.

    For each user:
    1. Sample ``num_negatives`` random negatives (unseen items).
    2. Rank all candidates by dot-product score.
    3. Compute NDCG@K, Precision@K, Recall@K.

    Vectorised within each user via numpy batch ops.
    The outer loop over users is unavoidable (each user has a different
    positive set), but the inner ranking computation is fully vectorised.

    Parameters
    ----------
    user_emb          (U, D) – user representations.
    item_emb          (I, D) – item representations.
    test_positives    user_id → list of positive item indices (0-based).
    seen_positives    user_id → set of seen item indices (train + test).
    config            Evaluation hyper-parameters.
    """
    rng = np.random.default_rng(config.seed)
    num_items = item_emb.shape[0]

    ndcgs:      List[float] = []
    precisions: List[float] = []
    recalls:    List[float] = []
    hit_rates:  List[float] = []

    for user_id, pos_items in test_positives.items():
        if not pos_items:
            continue
        uid = int(user_id)
        if uid >= user_emb.shape[0]:
            continue

        seen: Set[int] = set(seen_positives.get(uid, set())) if seen_positives else set()
        pos_unique = list(dict.fromkeys(pos_items))
        seen.update(pos_unique)

        # Vectorised negative sampling: batch-draw then filter
        n_draw = config.num_negatives * 4
        draws = rng.integers(0, num_items, size=n_draw)
        not_seen = draws[~np.isin(draws, list(seen))]
        negs: List[int] = not_seen[:config.num_negatives].tolist()

        if len(negs) < config.num_negatives:
            # Fallback: slower but correct uniform draw
            all_items = np.arange(num_items)
            remaining = np.setdiff1d(all_items, np.array(list(seen)))
            if remaining.size > 0:
                extra = rng.choice(
                    remaining,
                    size=min(config.num_negatives - len(negs), remaining.size),
                    replace=False,
                )
                negs.extend(extra.tolist())

        candidates = np.array(pos_unique + negs, dtype=np.int64)
        labels = np.zeros(len(candidates), dtype=np.float32)
        labels[: len(pos_unique)] = 1.0

        # Vectorised scoring: item_emb[candidates] @ user_emb[uid]  (matrix-vec)
        u_vec = user_emb[uid]                           # (D,)
        scores = item_emb[candidates] @ u_vec           # (C,) – vectorised
        order = np.argsort(scores)[::-1]
        ranked_labels = labels[order]

        ndcgs.append(_ndcg(ranked_labels, config.k))
        p, r = _precision_recall(ranked_labels, config.k, num_pos=len(pos_unique))
        precisions.append(p)
        recalls.append(r)
        hit_rates.append(1.0 if int(np.sum(ranked_labels[:config.k])) >= 1 else 0.0)

    n = len(ndcgs)
    return {
        f"NDCG@{config.k}":      float(np.mean(ndcgs)      if ndcgs      else 0.0),
        f"Precision@{config.k}": float(np.mean(precisions) if precisions else 0.0),
        f"Recall@{config.k}":    float(np.mean(recalls)    if recalls    else 0.0),
        f"HitRate@{config.k}":   float(np.mean(hit_rates)  if hit_rates  else 0.0),
        "UsersEvaluated":        float(n),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batched model evaluator
# ─────────────────────────────────────────────────────────────────────────────

class BatchedEvaluator:
    """
    Runs full-graph GNN inference once, then evaluates all metrics.

    Usage
    -----
    ::
        ev = BatchedEvaluator(
            edge_index, num_users, num_items,
            test_positives, seen_positives, test_df,
            eval_cfg, device
        )
        metrics = ev.evaluate(model)
    """
    def __init__(
        self,
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
        test_positives: Dict[int, List[int]],
        seen_positives: Dict[int, Set[int]],
        test_df,                           # pd.DataFrame with UserID, MovieID, Rating
        eval_cfg: EvalConfig = EvalConfig(),
        device: Optional[torch.device] = None,
        user_to_super: Optional[np.ndarray] = None,
        num_super: Optional[int] = None,
    ):
        self.edge_index      = edge_index
        self.num_users       = num_users
        self.num_items       = num_items
        self.test_positives  = test_positives
        self.seen_positives  = seen_positives
        self.test_df         = test_df
        self.eval_cfg        = eval_cfg
        self.device          = device or torch.device("cpu")
        self.user_to_super   = user_to_super
        self.num_super       = num_super or num_users

    @torch.no_grad()
    def _infer(self, model: nn.Module) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return super_emb, item_emb, and wall-clock inference time."""
        model.eval()
        ei = self.edge_index.to(self.device)
        t0 = time.perf_counter()
        z = model(ei).detach().cpu().numpy()
        inf_time = time.perf_counter() - t0
        super_emb = z[:self.num_super]
        item_emb  = z[self.num_super:]
        return super_emb, item_emb, inf_time

    def evaluate(self, model: nn.Module) -> Dict[str, float]:
        """Full evaluation: ranking + regression + inference time."""
        super_emb, item_emb, inf_time = self._infer(model)

        # Map super-node embeddings back to original-user space
        if self.user_to_super is not None:
            user_emb = super_emb[self.user_to_super]
        else:
            user_emb = super_emb

        # ── Ranking metrics ────────────────────────────────────────────────────
        ranking = evaluate_ranking_from_embeddings(
            user_emb, item_emb,
            self.test_positives, self.seen_positives,
            self.eval_cfg,
        )

        # ── Regression metrics (vectorised, no Python per-row loop) ───────────
        uid_arr  = self.test_df["UserID"].to_numpy(dtype=np.int64)
        mid_arr  = self.test_df["MovieID"].to_numpy(dtype=np.int64)
        y_true   = self.test_df["Rating"].to_numpy(dtype=np.float32)

        # Vectorised dot-product prediction
        scores   = np.einsum("nd,nd->n", user_emb[uid_arr], item_emb[mid_arr])
        y_pred   = 1.0 + 4.0 * (1.0 / (1.0 + np.exp(-np.clip(scores, -20.0, 20.0))))
        regression = compute_regression_metrics(y_true, y_pred)

        return {
            **ranking,
            **regression,
            "inference_time_s": inf_time,
        }

    def evaluate_and_log_callback(
        self, prefix: str
    ) -> "Callable[[nn.Module], Dict[str, float]]":
        """Return a closure suitable for passing as eval_callback to train_model."""
        def _cb(m: nn.Module) -> Dict[str, float]:
            return {f"{prefix}/{k}": v for k, v in self.evaluate(m).items()}
        return _cb


# ─────────────────────────────────────────────────────────────────────────────
# Efficiency comparison
# ─────────────────────────────────────────────────────────────────────────────

def compute_efficiency_metrics(
    baseline_summary: Dict[str, float],
    gsp_summary: Dict[str, float],
    gsp_preprocess_time_s: float,
) -> Dict[str, float]:
    """
    Compute speed / memory efficiency of GSP vs baseline.

    Returns
    -------
    dict with speedup, memory_reduction, preprocessing_vs_training, etc.
    """
    base_t   = baseline_summary.get("avg_epoch_time_s", 1e-9)
    gsp_t    = gsp_summary.get("avg_epoch_time_s", 1e-9)
    base_mem = baseline_summary.get("max_gpu_mem_mb", 0.0)
    gsp_mem  = gsp_summary.get("max_gpu_mem_mb", 0.0)

    speedup = float(base_t) / max(float(gsp_t), 1e-9)
    mem_red = float(base_mem - gsp_mem) / max(float(base_mem), 1e-9)
    total_gsp_train = gsp_summary.get("total_train_time_s", 0.0)

    return {
        "epoch_speedup":               round(speedup, 4),
        "memory_reduction_ratio":      round(mem_red, 4),
        "gsp_preprocess_time_s":       round(gsp_preprocess_time_s, 3),
        "gsp_total_train_time_s":      round(float(total_gsp_train), 3),
        "baseline_total_train_time_s": round(float(baseline_summary.get("total_train_time_s", 0.0)), 3),
        "preprocess_vs_training_ratio": round(
            gsp_preprocess_time_s / max(float(total_gsp_train), 1e-9), 4
        ),
    }
