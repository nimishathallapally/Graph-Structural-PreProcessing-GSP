from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

from gsprec.config import ProjectConfig
from gsprec.data import load_and_build_graph
from gsprec.graph import gsp_preprocess
from gsprec.models import (
    GATRecommender,
    GCNRecommender,
    LightGCNRecommender,
    RankingEvalConfig,
    SAGERecommender,
    TrainConfig,
    evaluate_ranking_from_embeddings,
    rmse_mae,
    train_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize each row."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


def _append_jsonl(path: str, record: Dict) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8", buffering=1) as f:
        f.write(json.dumps(record) + "\n")


def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _rss_mb() -> float:
    """Current process RSS memory in MB (Linux /proc/self/status)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Dataset statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_dataset_stats(ratings_df, train_df, test_df, num_users: int, num_items: int) -> Dict:
    """Compute and return dataset-level statistics."""
    n_interactions = len(ratings_df)
    sparsity = 1.0 - n_interactions / max(num_users * num_items, 1)
    n_train = len(train_df)
    n_test = len(test_df)
    return {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": n_interactions,
        "sparsity": round(sparsity, 6),
        "train_interactions": n_train,
        "test_interactions": n_test,
        "train_ratio": round(n_train / max(n_interactions, 1), 4),
        "test_ratio": round(n_test / max(n_interactions, 1), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bipartite graph statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_bipartite_graph_stats(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    num_users: int,
    num_items: int,
) -> Dict:
    """
    Compute statistics of the bipartite user-item graph.

    Parameters
    ----------
    user_ids, item_ids   parallel arrays of 0-based user and item indices
                         representing unique (user, item) edges.
    """
    N = num_users + num_items
    E = len(user_ids)  # unique undirected edges

    if E == 0:
        return {
            "num_nodes": N,
            "num_edges": 0,
            "avg_degree": 0.0,
            "max_degree": 0,
            "min_degree": 0,
            "density": 0.0,
            "num_components": N,
        }

    # Degree sequence: user degree + item degree (offset items by num_users)
    # Build scipy sparse for connected_components
    row = user_ids.astype(np.int32)
    col = (item_ids + num_users).astype(np.int32)
    data = np.ones(E, dtype=np.int8)
    adj = sp.csr_matrix((data, (row, col)), shape=(N, N))
    adj = adj + adj.T  # make symmetric (undirected)

    deg = np.array(adj.sum(axis=1)).flatten()  # degree for every node

    num_comp, _ = connected_components(adj, directed=False, return_labels=True)
    avg_deg = float(2 * E / N)
    density = float(2 * E / (N * (N - 1))) if N > 1 else 0.0

    return {
        "num_nodes": int(N),
        "num_edges": int(E),
        "avg_degree": round(avg_deg, 4),
        "max_degree": int(deg.max()),
        "min_degree": int(deg.min()),
        "density": round(density, 8),
        "num_components": int(num_comp),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full-item ranking evaluation (no negative sampling)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_full_ranking(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    test_positives: Dict[int, List[int]],
    seen_positives: Dict[int, Set[int]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Rank ALL items for each test user (full-item evaluation).
    Excludes items seen in training.  Returns Precision, Recall, NDCG, HitRate at k.
    """
    num_items = item_emb.shape[0]
    precisions, recalls, ndcgs, hit_rates = [], [], [], []

    for user_id, pos_items in test_positives.items():
        if not pos_items:
            continue
        uid = int(user_id)
        if uid >= user_emb.shape[0]:
            continue

        seen: Set[int] = set(seen_positives.get(uid, set()))
        pos_set = set(pos_items)

        # Score all items
        scores = item_emb @ user_emb[uid]  # (I,)

        # Mask out seen train items (exclude positives from training)
        train_seen = seen - pos_set
        if train_seen:
            mask_idx = np.array(list(train_seen), dtype=np.int64)
            valid_mask = mask_idx[mask_idx < num_items]
            scores[valid_mask] = -1e9

        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

        hits = np.array([1.0 if idx in pos_set else 0.0 for idx in top_k_idx])
        num_pos = len(pos_set)

        # Precision@k
        p = float(hits.sum()) / k
        # Recall@k
        r = float(hits.sum()) / max(num_pos, 1)
        # NDCG@k
        discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
        dcg = float(np.dot(hits, discounts))
        ideal_hits = np.ones(min(num_pos, k), dtype=np.float64)
        ideal_discounts = discounts[:len(ideal_hits)]
        idcg = float(np.dot(ideal_hits, ideal_discounts))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        # HitRate@k
        hr = 1.0 if hits.sum() >= 1 else 0.0

        precisions.append(p)
        recalls.append(r)
        ndcgs.append(ndcg)
        hit_rates.append(hr)

    n = len(ndcgs)
    return {
        f"Precision@{k}":  round(float(np.mean(precisions)) if precisions else 0.0, 6),
        f"Recall@{k}":     round(float(np.mean(recalls))    if recalls    else 0.0, 6),
        f"NDCG@{k}":       round(float(np.mean(ndcgs))      if ndcgs      else 0.0, 6),
        f"HitRate@{k}":    round(float(np.mean(hit_rates))  if hit_rates  else 0.0, 6),
        "UsersEvaluated":  n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_by_user_leave_one_out(ratings_df, seed: int = 42, test_ratio: float = 0.2) -> Tuple:
    """Per-user ratio split: ``test_ratio`` fraction of each user's interactions
    go to test (minimum 1), the rest stay in train. Expects 0-based UserID."""
    rng = np.random.default_rng(seed)
    df = ratings_df.copy()
    test_indices = []
    for _, group in df.groupby("UserID"):
        idxs = group.index.to_numpy()
        if idxs.size == 0:
            continue
        n_test = max(1, int(round(len(idxs) * test_ratio)))
        # Keep at least 1 interaction in train
        n_test = min(n_test, len(idxs) - 1)
        chosen = rng.choice(idxs, size=n_test, replace=False)
        test_indices.extend(chosen.tolist())

    test_mask = df.index.isin(test_indices)
    train_df = df.loc[~test_mask, ["UserID", "MovieID", "Rating", "Timestamp"]].reset_index(drop=True)
    test_df = df.loc[test_mask, ["UserID", "MovieID", "Rating", "Timestamp"]].reset_index(drop=True)
    return train_df, test_df


def build_seen_sets(df) -> Dict[int, Set[int]]:
    """Build seen-item sets per user. Expects 0-based UserID / MovieID."""
    seen: Dict[int, Set[int]] = {}
    for row in df.itertuples(index=False):
        u = int(row.UserID)      # already 0-based
        i = int(row.MovieID)     # already 0-based
        seen.setdefault(u, set()).add(i)
    return seen


def build_test_positives(test_df, threshold: float = 4.0) -> Dict[int, List[int]]:
    """Build test-positive lists per user. Expects 0-based UserID / MovieID."""
    pos: Dict[int, List[int]] = {}
    for row in test_df.itertuples(index=False):
        if float(row.Rating) < threshold:
            continue
        u = int(row.UserID)      # already 0-based
        i = int(row.MovieID)     # already 0-based
        pos.setdefault(u, []).append(i)
    return pos


def _build_model(name: str, num_nodes: int, emb_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
    n = name.lower()
    if n == "gat":
        return GATRecommender(num_nodes=num_nodes, emb_dim=emb_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    if n == "graphsage":
        return SAGERecommender(num_nodes=num_nodes, emb_dim=emb_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    if n == "gcn":
        return GCNRecommender(num_nodes=num_nodes, emb_dim=emb_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    if n == "lightgcn":
        return LightGCNRecommender(num_nodes=num_nodes, emb_dim=emb_dim, num_layers=num_layers)
    raise ValueError(f"Unsupported model: {name}")


def run_pipeline(config: ProjectConfig) -> None:
    t_pipeline0 = time.perf_counter()
    outputs_path = config.output_dir
    _ensure_dir(outputs_path)

    # ── Data loading (with caching and optional debug mode) ───────────────────
    print(f"[Runner] Loading dataset: {config.data.dataset_name} …")
    data = load_and_build_graph(
        debug_mode=config.data.debug_mode,
        max_debug_users=config.data.max_debug_users,
        cache_dir=config.data.cache_dir,
        force_reload=config.data.force_reload,
        dataset_name=config.data.dataset_name,
        dataset_path=config.data.dataset_path,        min_interactions=config.data.min_interactions,    )
    ratings = data["ratings_df"]          # 0-based UserID, MovieID
    num_users: int = data["num_users"]
    num_items: int = data["num_items"]
    data_load_time: float = data["load_time_s"]
    print(f"[Runner] Interactions={len(ratings)}  users={num_users}  items={num_items}")

    train_df, test_df = split_by_user_leave_one_out(ratings, seed=config.data.test_seed)
    seen_train = build_seen_sets(train_df)
    test_pos = build_test_positives(test_df, threshold=config.data.implicit_threshold)
    train_mean_rating = float(train_df["Rating"].mean())

    # ── GSP preprocessing (Stage I + II) ─────────────────────────────────────
    gsp_out = gsp_preprocess(
        ratings_df=train_df,
        num_users=num_users,
        num_items=num_items,
        implicit_threshold=config.data.implicit_threshold,
        alpha=config.gsp.alpha,
        curvature_percentile=config.gsp.curvature_percentile,
        curvature_topk=config.gsp.curvature_topk or None,
        importance_percentile=config.gsp.importance_percentile,
        importance_topk=config.gsp.importance_topk or None,
        er_num_eigenvectors=config.gsp.er_num_eigenvectors,
        max_cluster_size=config.gsp.max_cluster_size or 0,
        min_shared_interactions=getattr(config.gsp, "min_shared_interactions", 2),
        curvature_mode=getattr(config.gsp, "curvature_mode", "cosine"),
        er_solver=getattr(config.gsp, "er_solver", "dwlv"),
        er_node_limit=getattr(config.gsp, "er_node_limit", 0),
        cache_dir=config.data.cache_dir,
        output_dir=outputs_path,
        data_load_time_s=data_load_time,
        device=None,   # auto-detect CUDA
    )
    user_to_super: np.ndarray = gsp_out["user_to_super"]
    num_super: int = gsp_out["num_super"]
    F = gsp_out["F_hc"]      # curvature of high-curvature edges (Stage I output)
    I_e = gsp_out["I_e"]
    u_all = gsp_out["u_hc"]  # high-curvature edge set (Stage I output)

    num_users_coarsened = num_users - num_super
    coarsen_ratio = num_users_coarsened / max(num_users, 1)
    _append_jsonl(
        os.path.join(outputs_path, "pipeline_metrics.jsonl"),
        {
            "stage": "coarsen",
            "time_s": gsp_out["timing"]["coarsening_time_s"],
            "num_users": num_users,
            "num_super": num_super,
            "num_users_coarsened": num_users_coarsened,
            "coarsen_ratio": coarsen_ratio,
            "user_user_edges_total": int(u_all.size),
        },
    )

    # ── Build coarsened bipartite graph ───────────────────────────────────────
    # UserID/MovieID are already 0-based; map users → super-nodes then aggregate.
    t0 = time.perf_counter()
    train_copy = train_df.copy()
    train_copy["super_idx"] = user_to_super[train_copy["UserID"].to_numpy(dtype=np.int64)]
    coarsened_train = (
        train_copy.groupby(["super_idx", "MovieID"], as_index=False)
        .agg(rating=("Rating", "mean"), count=("Rating", "size"))
        .rename(columns={"MovieID": "item_idx"})
        .astype({"super_idx": np.int64, "item_idx": np.int64})
    )

    # Build PyG edge_index for coarsened bipartite graph (undirected)
    su = coarsened_train["super_idx"].to_numpy(dtype=np.int64)
    it = coarsened_train["item_idx"].to_numpy(dtype=np.int64) + num_super
    edge_index_np = np.stack(
        [np.concatenate([su, it]), np.concatenate([it, su])], axis=0
    )
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    t_graph = float(time.perf_counter() - t0)

    original_unique_edges = int(train_df[["UserID", "MovieID"]].drop_duplicates().shape[0])
    coarsened_unique_edges = int(coarsened_train.shape[0])
    edge_reduction_ratio = (original_unique_edges - coarsened_unique_edges) / max(original_unique_edges, 1)
    _append_jsonl(
        os.path.join(outputs_path, "pipeline_metrics.jsonl"),
        {
            "stage": "build_graph",
            "time_s": t_graph,
            "num_edges": int(edge_index.shape[1]),
            "original_unique_user_item_edges": original_unique_edges,
            "coarsened_unique_user_item_edges": coarsened_unique_edges,
            "edge_reduction_ratio": float(edge_reduction_ratio),
            "edge_reduction_percent": float(edge_reduction_ratio * 100.0),
        },
    )

    train_super = torch.tensor(coarsened_train["super_idx"].to_numpy(dtype=np.int64), dtype=torch.long)
    train_item = torch.tensor((coarsened_train["item_idx"].to_numpy(dtype=np.int64) + num_super), dtype=torch.long)
    train_y = torch.tensor(coarsened_train["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = TrainConfig(
        epochs=config.train.epochs,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        batch_size=config.train.batch_size,
        neg_ratio=config.train.neg_ratio,
        rating_loss_weight=config.train.rating_loss_weight,
        emb_l2_weight=config.train.emb_l2_weight,
        log_jsonl_path=os.path.join(outputs_path, "training_metrics.jsonl"),
        checkpoint_dir=os.path.join(outputs_path, "checkpoints"),
        device=device,
    )
    rank_cfg = RankingEvalConfig(k=config.eval.k, num_negatives=config.eval.num_negatives, seed=config.eval.seed)

    num_nodes = num_super + num_items

    def eval_rmse_mae_for_model(model) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            z = model(edge_index.to(train_cfg.device)).detach().cpu().numpy()
        super_emb = z[:num_super]
        item_emb = z[num_super:]
        user_emb = super_emb[user_to_super]

        y_true_list: List[float] = []
        y_pred_list: List[float] = []
        for row in test_df.itertuples(index=False):
            u_local = int(row.UserID)   # already 0-based
            i_local = int(row.MovieID)  # already 0-based
            y_true_list.append(float(row.Rating))
            score = float(item_emb[i_local] @ user_emb[u_local])
            score = float(np.clip(score, -20.0, 20.0))
            y_pred_list.append(float(1.0 + 4.0 * (1.0 / (1.0 + np.exp(-score)))))
        y_true = np.array(y_true_list, dtype=np.float32)
        y_pred = np.array(y_pred_list, dtype=np.float32)
        # Global mean bias correction: model uses dot-product → sigmoid → [1,5]
        # which centres at 3.0, but actual ratings cluster near train_mean_rating.
        pred_bias = train_mean_rating - float(np.mean(y_pred))
        y_pred = np.clip(y_pred + pred_bias, 1.0, 5.0)
        rmse, mae = rmse_mae(y_true, y_pred)
        return {"RMSE": rmse, "MAE": mae}

    def make_eval_callback(local_edge_index: torch.Tensor, local_num_users: int, local_num_super: int, run_name: str):
        def _cb(m):
            m.eval()
            with torch.no_grad():
                z = m(local_edge_index.to(train_cfg.device)).detach().cpu().numpy()

            if local_num_super == local_num_users:
                user_emb = z[:local_num_users]
                item_emb = z[local_num_users:]
            else:
                super_emb = z[:local_num_super]
                item_emb = z[local_num_super:]
                user_emb = super_emb[user_to_super]

            metrics = evaluate_ranking_from_embeddings(
                user_emb=_l2_normalize(user_emb),
                item_emb=_l2_normalize(item_emb),
                test_positives=test_pos,
                seen_positives=seen_train,
                config=rank_cfg,
            )
            return {f"{run_name}_{k}": v for k, v in metrics.items() if k != "UsersEvaluated"}

        return _cb

    gsp_summaries: Dict[str, Dict[str, float]] = {}
    gsp_models = {}

    for model_name in config.models:
        model = _build_model(model_name, num_nodes, config.train.emb_dim, config.train.hidden_dim, config.train.out_dim, config.train.num_layers)
        summary = train_model(
            model,
            edge_index=edge_index,
            train_user_nodes=train_super,
            train_item_nodes=train_item,
            train_ratings=train_y,
            config=train_cfg,
            run_name=f"{model_name}_gsp",
            eval_callback=make_eval_callback(edge_index, num_users, num_super, f"{model_name}_gsp"),
        )
        gsp_summaries[model_name] = summary
        gsp_models[model_name] = model

    for model_name, model in gsp_models.items():
        model.eval()
        with torch.no_grad():
            z = model(edge_index.to(train_cfg.device)).detach().cpu().numpy()
        super_emb = z[:num_super]
        item_emb = z[num_super:]
        user_emb = super_emb[user_to_super]
        rank = evaluate_ranking_from_embeddings(_l2_normalize(user_emb), _l2_normalize(item_emb), test_pos, seen_train, rank_cfg)
        reg = eval_rmse_mae_for_model(model)
        _append_jsonl(os.path.join(outputs_path, "eval_metrics.jsonl"), {"model": f"{model_name}_gsp", **rank, **reg})

    baseline_summaries: Dict[str, Dict[str, float]] = {}

    if config.run_baseline:
        # UserID / MovieID are already 0-based; no offset arithmetic needed.
        base_interactions = train_df[["UserID", "MovieID", "Rating"]].copy()
        base_agg = (
            base_interactions.groupby(["UserID", "MovieID"], as_index=False)
            .agg(rating=("Rating", "mean"), count=("Rating", "size"))
            .rename(columns={"UserID": "super_idx", "MovieID": "item_idx"})
            .astype({"super_idx": np.int64, "item_idx": np.int64})
        )
        # Build PyG edge_index for baseline (no coarsening, full user set)
        base_su = base_agg["super_idx"].to_numpy(dtype=np.int64)
        base_it = base_agg["item_idx"].to_numpy(dtype=np.int64) + num_users
        base_edge_index = torch.tensor(
            np.stack(
                [np.concatenate([base_su, base_it]), np.concatenate([base_it, base_su])], axis=0
            ),
            dtype=torch.long,
        )
        _append_jsonl(
            os.path.join(outputs_path, "pipeline_metrics.jsonl"),
            {"stage": "baseline_build_graph", "num_edges": int(base_edge_index.shape[1])},
        )

        base_train_user = torch.tensor(base_agg["super_idx"].to_numpy(dtype=np.int64), dtype=torch.long)
        base_train_item = torch.tensor(base_agg["item_idx"].to_numpy(dtype=np.int64) + num_users, dtype=torch.long)
        base_train_y = torch.tensor(base_agg["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)
        base_nodes = num_users + num_items

        for model_name in config.models:
            base_model = _build_model(model_name, base_nodes, config.train.emb_dim, config.train.hidden_dim, config.train.out_dim, config.train.num_layers)
            summary = train_model(
                base_model,
                edge_index=base_edge_index,
                train_user_nodes=base_train_user,
                train_item_nodes=base_train_item,
                train_ratings=base_train_y,
                config=train_cfg,
                run_name=f"{model_name}_baseline",
                eval_callback=make_eval_callback(base_edge_index, num_users, num_users, f"{model_name}_baseline"),
            )
            baseline_summaries[model_name] = summary

            base_model.eval()
            with torch.no_grad():
                z = base_model(base_edge_index.to(train_cfg.device)).detach().cpu().numpy()
            user_emb = z[:num_users]
            item_emb = z[num_users:]
            rank = evaluate_ranking_from_embeddings(_l2_normalize(user_emb), _l2_normalize(item_emb), test_pos, seen_train, rank_cfg)

            y_true, y_pred = [], []
            for row in test_df.itertuples(index=False):
                u_local = int(row.UserID)   # already 0-based
                i_local = int(row.MovieID)  # already 0-based
                y_true.append(float(row.Rating))
                score = float(item_emb[i_local] @ user_emb[u_local])
                score = float(np.clip(score, -20.0, 20.0))
                y_pred.append(float(1.0 + 4.0 * (1.0 / (1.0 + np.exp(-score)))))
            y_true_arr = np.array(y_true, dtype=np.float32)
            y_pred_arr = np.array(y_pred, dtype=np.float32)
            # Global mean bias correction (same as GSP eval)
            pred_bias = train_mean_rating - float(np.mean(y_pred_arr))
            y_pred_arr = np.clip(y_pred_arr + pred_bias, 1.0, 5.0)
            rmse, mae = rmse_mae(y_true_arr, y_pred_arr)
            _append_jsonl(
                os.path.join(outputs_path, "eval_metrics.jsonl"),
                {"model": f"{model_name}_baseline", **rank, "RMSE": rmse, "MAE": mae},
            )

        speedup_record: Dict[str, float] = {"stage": "speedup"}
        for model_name in config.models:
            base_t = float(baseline_summaries[model_name].get("avg_epoch_time_s", 0.0))
            gsp_t = float(gsp_summaries[model_name].get("avg_epoch_time_s", 1e-9))
            speedup_record[f"{model_name}_speedup"] = float(base_t / max(gsp_t, 1e-9))
            speedup_record[f"{model_name}_baseline_avg_epoch_time_s"] = base_t
            speedup_record[f"{model_name}_gsp_avg_epoch_time_s"] = gsp_t
            speedup_record[f"{model_name}_baseline_max_gpu_mem_mb"] = float(
                baseline_summaries[model_name].get("max_gpu_mem_mb", 0.0)
            )
            speedup_record[f"{model_name}_gsp_max_gpu_mem_mb"] = float(gsp_summaries[model_name].get("max_gpu_mem_mb", 0.0))
        _append_jsonl(os.path.join(outputs_path, "pipeline_metrics.jsonl"), speedup_record)

    total_time = float(time.perf_counter() - t_pipeline0)
    _append_jsonl(os.path.join(outputs_path, "pipeline_metrics.jsonl"), {"stage": "total", "time_s": total_time})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Graph Structural Pre-conditioning recommender pipeline")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to JSON config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if config_path.exists():
        config = ProjectConfig.from_json(str(config_path))
    else:
        config = ProjectConfig()
    run_pipeline(config)


if __name__ == "__main__":
    main()
