#!/usr/bin/env python3
"""
run_ml25m.py  –  MovieLens-25M paper-results pipeline
======================================================

Mirrors run_ml1m.py exactly but uses the MovieLens-25M dataset with
large-dataset-safe defaults (chunk_size, max_item_degree, max_neighbors_per_user,
min_interactions).  One curvature mode per invocation – identical to
run_ml1m.py so results can be compared directly across the three datasets.

All model failures (OOM, CUDA errors) are caught and logged; the sweep
continues to the next model.

Output files (identical layout to run_ml1m.py):
    dataset_stats.json        hardware_info.json
    gsp_stats.json
    full_results.json
    results_table.csv         speedup_results.csv
    training_metrics_{model}_{baseline|gsp}.jsonl
    training_log_{model}_{baseline|gsp}.txt

Typical usage
-------------
    # Single mode, 50 epochs
    python scripts/run_ml25m.py --curvature_mode cosine --epochs 50

    # Quick debug on 5k users
    python scripts/run_ml25m.py --debug_mode --epochs 5 --curvature_mode cosine

    # Custom output dir
    python scripts/run_ml25m.py --output_dir outputs/ml25m_run1 --curvature_mode cosine
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
import torch

# ── make gsprec importable ────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gsprec.data.pipeline import load_and_build_graph
from gsprec.graph.gsp_ops import gsp_preprocess
from gsprec.graph.embedding_store import project_embeddings
from gsprec.models.architectures import get_model, ModelConfig
from gsprec.models.trainer import TrainConfig, train_model
from gsprec.models.gnn import RankingEvalConfig, evaluate_ranking_from_embeddings, rmse_mae
from gsprec.utils.hardware_info import (
    collect_hardware_info, rss_mb,
    gpu_max_memory_allocated_mb, reset_gpu_peak_memory,
)
from gsprec.analytics import run_analytics_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_KS: Tuple[int, ...] = (10, 20, 50)
_TAG = "ML25M"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MovieLens-25M paper-results pipeline (one curvature mode per run)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output_dir", default="outputs/ml25m",
                   help="Where to write results")
    p.add_argument("--cache_dir", default="outputs/cache",
                   help="Cache directory for preprocessed data")
    p.add_argument("--force_reload", action="store_true",
                   help="Bypass on-disk cache and reload from source")
    p.add_argument("--models", default="lightgcn,gat,graphsage,gcn",
                   help="Comma-separated model list: lightgcn,gat,graphsage,gcn")
    p.add_argument("--curvature_mode", default="cosine",
                   choices=["cosine", "forman_ricci"],
                   help="Curvature metric for UU edge scoring. "
                        "'cosine': shared/sqrt(deg_u*deg_v), always in (0,1]. "
                        "'forman_ricci': 4-deg_u-deg_v+shared (classic formula).")
    p.add_argument("--epochs", type=int, default=50,
                   help="Training epochs per model")
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=65536)
    p.add_argument("--neg_ratio", type=int, default=4)
    p.add_argument("--implicit_threshold", type=float, default=3.5,
                   help="Minimum rating to treat as a positive interaction")
    p.add_argument("--min_interactions", type=int, default=10,
                   help="k-core filter: drop users/items with fewer interactions")
    p.add_argument("--eval_negatives", type=int, default=99,
                   help="Negative items sampled per user at evaluation (LOO-99 protocol)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stopping_patience", type=int, default=10,
                   help="Stop after this many epochs with no loss improvement. 0 = disabled.")
    p.add_argument("--target_fraction", type=float, default=None,
                   help="Keep the top X%% most active users by interaction count "
                        "(e.g. 0.25 = top 25%% of users). Applied before LOO split. "
                        "Range: (0.0, 1.0].")
    p.add_argument("--curvature_percentile", type=float, default=50.0)
    p.add_argument("--er_eigvecs", type=int, default=32)
    p.add_argument("--er_node_limit", type=int, default=0,
                   help="Skip ER when num_users > this; 0 = always run ER")
    p.add_argument("--er_solver", default="dwlv",
                   choices=["arpack", "lobpcg", "jl", "dwlv"],
                   help="ER solver. 'dwlv' = O(nnz) closed-form, fastest.")
    p.add_argument("--er_sketches", type=int, default=32)
    p.add_argument("--max_cluster_size", type=int, default=50,
                   help="Maximum users per super-node")
    p.add_argument("--clustering_method", default="hem",
                   choices=["hem", "connected_components"],
                   help="Clustering algorithm: 'hem' (default) = Heavy-Edge Matching.")
    p.add_argument("--min_shared", type=int, default=5,
                   help="Minimum shared items required for a user-user edge")
    p.add_argument("--max_item_degree", type=int, default=1000,
                   help="Exclude items rated by more than this many users from UU similarity "
                        "(0 = no cap). Cap at 1000 keeps per-chunk nnz bounded for ML-25M.")
    p.add_argument("--max_neighbors_per_user", type=int, default=100,
                   help="Per-user top-K cap in UU graph. Bounds total UU edges to "
                        "num_users*K/2 regardless of graph density. 0 = no cap.")
    p.add_argument("--chunk_size", type=int, default=20,
                   help="Row-chunk size for A@A.T; 0 = auto (20 for large, full for small)")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable mixed-precision training")
    p.add_argument("--debug_mode", action="store_true",
                   help="Use a small user subset for quick iteration")
    p.add_argument("--max_debug_users", type=int, default=5000)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Any) -> None:
    _mkdir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        def _default(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
        json.dump(obj, fh, indent=2, default=_default)


def _section(title: str) -> None:
    bar = "═" * (len(title) + 4)
    print(f"\n[{_TAG}] {bar}")
    print(f"[{_TAG}] ║  {title}  ║")
    print(f"[{_TAG}] {bar}")


def _l2_norm(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


def _proc_cpu_time_s() -> float:
    try:
        with open("/proc/self/stat") as fh:
            fields = fh.read().split()
        ticks_per_s = os.sysconf("SC_CLK_TCK")
        return (int(fields[13]) + int(fields[14])) / ticks_per_s
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers  (vectorised – safe for 25M rows)
# ─────────────────────────────────────────────────────────────────────────────

def _split_leave_one_out(df, threshold: float, seed: int = 42) -> Tuple[Any, Any]:
    """Leave-One-Out: hold out exactly 1 positive per user (vectorised)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    pos_mask = df["Rating"].to_numpy(dtype=np.float32) >= threshold
    pos_df = df.loc[pos_mask].copy()

    # For each user, randomly pick one positive row to hold out
    pos_df["_rnd"] = rng.random(len(pos_df))
    # Keep only users with >= 2 positives
    counts = pos_df.groupby("UserID")["_rnd"].transform("count")
    eligible = pos_df.loc[counts >= 2].copy()
    # Within eligible, hold out the row with the smallest random value per user
    eligible["_rank"] = eligible.groupby("UserID")["_rnd"].rank(method="first")
    test_idx = eligible.loc[eligible["_rank"] == 1].index
    mask = df.index.isin(test_idx)
    return df.loc[~mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)


def _build_seen(df) -> Dict[int, Set[int]]:
    """Vectorised: build seen-item sets per user."""
    seen: Dict[int, Set[int]] = {}
    users = df["UserID"].to_numpy(dtype=np.int64)
    items = df["MovieID"].to_numpy(dtype=np.int64)
    for u, i in zip(users, items):
        seen.setdefault(int(u), set()).add(int(i))
    return seen


def _build_positives(df, threshold: float) -> Dict[int, List[int]]:
    """Vectorised: build positive-item lists per user above threshold."""
    mask = df["Rating"].to_numpy(dtype=np.float32) >= threshold
    pos_df = df.loc[mask]
    pos: Dict[int, List[int]] = {}
    users = pos_df["UserID"].to_numpy(dtype=np.int64)
    items = pos_df["MovieID"].to_numpy(dtype=np.int64)
    for u, i in zip(users, items):
        pos.setdefault(int(u), []).append(int(i))
    return pos


def _build_edge_index(
    user_arr: np.ndarray,
    item_arr: np.ndarray,
    item_offset: int,
) -> torch.Tensor:
    it_shifted = item_arr + item_offset
    src = np.concatenate([user_arr, it_shifted])
    dst = np.concatenate([it_shifted, user_arr])
    return torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval_multi_k(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    test_positives: Dict[int, List[int]],
    seen_positives: Dict[int, Set[int]],
    num_negatives: int,
    seed: int,
    ks: Tuple[int, ...] = _EVAL_KS,
) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    users_eval = 0.0
    for k in ks:
        cfg = RankingEvalConfig(k=k, num_negatives=num_negatives, seed=seed)
        m = evaluate_ranking_from_embeddings(user_emb, item_emb, test_positives, seen_positives, cfg)
        users_eval = m.pop("UsersEvaluated", users_eval)
        merged.update(m)
    merged["UsersEvaluated"] = users_eval
    return merged


def _compute_rmse_mae(
    test_df,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    train_mean: float,
) -> Dict[str, float]:
    y_true, y_pred = [], []
    for row in test_df.itertuples(index=False):
        u, i, r = int(row.UserID), int(row.MovieID), float(row.Rating)
        if u >= user_emb.shape[0] or i >= item_emb.shape[0]:
            continue
        score = float(np.clip(item_emb[i] @ user_emb[u], -20.0, 20.0))
        y_pred.append(float(1.0 + 4.0 / (1.0 + np.exp(-score))))
        y_true.append(r)
    if not y_true:
        return {"RMSE": float("nan"), "MAE": float("nan")}
    y_t = np.array(y_true, dtype=np.float32)
    y_p = np.array(y_pred, dtype=np.float32)
    y_p = np.clip(y_p + (train_mean - float(y_p.mean())), 1.0, 5.0)
    rmse_val, mae_val = rmse_mae(y_t, y_p)
    return {"RMSE": rmse_val, "MAE": mae_val}


def _infer(model: torch.nn.Module, edge_index: torch.Tensor, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(edge_index.to(device)).detach().cpu().float().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_paper_table(rows: List[Dict], ks: Tuple[int, ...] = _EVAL_KS) -> None:
    cols_rank = [f"{m}@{k}" for k in ks for m in ("Precision", "Recall", "NDCG", "HitRate")]
    cols_reg  = ["RMSE", "MAE"]
    header = (
        f"{'Model':<18}  {'Type':<16}  "
        + "  ".join(f"{c:>14}" for c in cols_rank + cols_reg)
        + f"  {'Train(s)':>10}"
    )
    print("\n" + "=" * len(header))
    print("PAPER RESULTS TABLE  —  MovieLens-25M")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = [f"{r.get(c, float('nan')):>14.4f}" for c in cols_rank + cols_reg]
        print(
            f"{r['model']:<18}  {r['run_type']:<16}  "
            + "  ".join(vals)
            + f"  {r.get('training_time_s', 0):>10.1f}"
        )
    print("=" * len(header))

    # LaTeX snippet
    n_cols = len(cols_rank) + len(cols_reg)
    print("\n--- LaTeX snippet ---")
    print(r"\begin{tabular}{ll" + "r" * n_cols + r"}")
    print(r"\toprule")
    rank_headers = " & ".join(cols_rank)
    print(f"Model & Type & {rank_headers} & RMSE & MAE \\\\")
    print(r"\midrule")
    for r in rows:
        model_tex = r["model"].replace("_", r"\_")
        type_tex  = r["run_type"].replace("_", r"\_")
        vals = " & ".join(
            f"{r.get(c, float('nan')):.4f}" for c in cols_rank + cols_reg
        )
        print(f"{model_tex} & {type_tex} & {vals} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def _save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    _mkdir(os.path.dirname(os.path.abspath(path)))
    fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline  (flat – one curvature mode per invocation, identical to ml1m)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    t_wall = time.perf_counter()

    out_dir = args.output_dir
    _mkdir(out_dir)
    _mkdir(os.path.join(out_dir, "checkpoints"))
    _mkdir(os.path.join(out_dir, "cache"))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_to_run = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    print(f"[{_TAG}] Device  : {device}")
    print(f"[{_TAG}] Models  : {models_to_run}")
    print(f"[{_TAG}] Epochs  : {args.epochs}")
    print(f"[{_TAG}] Mode    : {args.curvature_mode}")
    if args.target_fraction is not None:
        print(f"[{_TAG}] Target  : top {args.target_fraction*100:.1f}% most active users")
    else:
        print(f"[{_TAG}] Target  : full dataset (no user filtering)")
    print(f"[{_TAG}] Output  : {out_dir}")

    hw = collect_hardware_info()
    hw.update({"device": device, "seed": args.seed, "models": models_to_run,
               "curvature_mode": args.curvature_mode, "target_fraction": args.target_fraction})
    _write_json(os.path.join(out_dir, "hardware_info.json"), hw)

    # ── STAGE 0: Load MovieLens-25M ───────────────────────────────────────────
    _section("STAGE 0: Dataset Loading (MovieLens-25M)")
    t0 = time.perf_counter()
    data = load_and_build_graph(
        debug_mode=args.debug_mode,
        max_debug_users=args.max_debug_users,
        cache_dir=args.cache_dir,
        force_reload=args.force_reload,
        dataset_name="movielens25m",
        min_interactions=args.min_interactions,
    )
    load_time = time.perf_counter() - t0
    ratings_df = data["ratings_df"]
    num_users: int = data["num_users"]
    num_items: int = data["num_items"]
    n_interactions = len(ratings_df)
    sparsity = 1.0 - n_interactions / max(num_users * num_items, 1)

    print(
        f"[{_TAG}] {num_users:,} users | {num_items:,} items | "
        f"{n_interactions:,} interactions | sparsity={sparsity:.4%} | {load_time:.2f}s"
    )

    # ── Optional: filter to top-frequent users ────────────────────────────────
    _subset_method_label = "full_dataset"
    if args.target_fraction is not None:
        if not (0.0 < args.target_fraction <= 1.0):
            raise ValueError(
                f"--target_fraction must be in (0, 1], got {args.target_fraction}"
            )
        _n_users_total = ratings_df["UserID"].nunique()
        _n_to_select = max(1, int(math.ceil(args.target_fraction * _n_users_total)))
        _interaction_counts = ratings_df.groupby("UserID").size()
        _top_users = _interaction_counts.nlargest(_n_to_select).index.to_numpy()
        _selected_set = set(_top_users.tolist())
        ratings_df = ratings_df[ratings_df["UserID"].isin(_selected_set)].copy()
        _uniq_u = np.sort(ratings_df["UserID"].unique())
        _uniq_i = np.sort(ratings_df["MovieID"].unique())
        ratings_df["UserID"] = np.searchsorted(_uniq_u, ratings_df["UserID"].to_numpy())
        ratings_df["MovieID"] = np.searchsorted(_uniq_i, ratings_df["MovieID"].to_numpy())
        ratings_df = ratings_df.reset_index(drop=True)
        num_users = int(len(_uniq_u))
        num_items = int(len(_uniq_i))
        n_interactions = len(ratings_df)
        sparsity = 1.0 - n_interactions / max(num_users * num_items, 1)
        _subset_method_label = (
            f"top_frequent_users fraction={args.target_fraction:.4f} "
            f"\u2192 top {num_users:,}/{_n_users_total:,} users"
        )
        print(f"[{_TAG}] Subset  : {_subset_method_label}")
        print(
            f"[{_TAG}] After filter: {num_users:,} users | {num_items:,} items | "
            f"{n_interactions:,} interactions | sparsity={sparsity:.4%}"
        )

    dataset_summary: Dict[str, Any] = {
        "dataset": "movielens-25m",
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": n_interactions,
        "sparsity": round(sparsity, 6),
        "implicit_threshold": args.implicit_threshold,
        "min_interactions": args.min_interactions,
        "load_time_s": round(load_time, 3),
        "debug_mode": args.debug_mode,
        "target_fraction": args.target_fraction,
        "subset_method_label": _subset_method_label,
    }

    # ── Train/test split (Leave-One-Out) ──────────────────────────────────────
    _section("Train / Test Split  (Leave-One-Out)")
    train_df, test_df = _split_leave_one_out(
        ratings_df, threshold=args.implicit_threshold, seed=args.seed
    )
    seen_train = _build_seen(train_df)
    test_pos   = _build_positives(test_df, args.implicit_threshold)
    train_mean = float(train_df["Rating"].mean())

    dataset_summary.update({
        "train_interactions": len(train_df),
        "test_interactions": len(test_df),
        "test_users_with_positives": len(test_pos),
        "split_protocol": "leave-one-out",
    })
    print(
        f"[{_TAG}] LOO split | train={len(train_df):,} | test={len(test_df):,} | "
        f"test-positive users={len(test_pos):,}"
    )
    _write_json(os.path.join(out_dir, "dataset_stats.json"), dataset_summary)

    # ── STAGE I-II: GSP Preprocessing ────────────────────────────────────────
    _section("STAGE I-II: GSP Preprocessing")
    t_gsp = time.perf_counter()
    gsp_out = gsp_preprocess(
        ratings_df=train_df,
        num_users=num_users,
        num_items=num_items,
        implicit_threshold=args.implicit_threshold,
        alpha=0.5,
        curvature_percentile=args.curvature_percentile,
        curvature_mode=args.curvature_mode,
        curvature_topk=None,
        importance_percentile=50.0,
        importance_topk=None,
        er_num_eigenvectors=args.er_eigvecs,
        max_cluster_size=args.max_cluster_size,
        min_shared_interactions=args.min_shared,
        max_item_degree=args.max_item_degree,
        max_neighbors_per_user=args.max_neighbors_per_user,
        chunk_size=args.chunk_size,
        er_node_limit=args.er_node_limit,
        er_solver=args.er_solver,
        er_sketches=args.er_sketches,
        clustering_method=args.clustering_method,
        seed=args.seed,
        cache_dir=os.path.join(out_dir, "cache"),
        output_dir=out_dir,
        data_load_time_s=load_time,
    )
    gsp_elapsed = time.perf_counter() - t_gsp

    gsp_stats: Dict = gsp_out["stats"]
    user_to_super: np.ndarray = gsp_out["user_to_super"]
    num_super: int = gsp_out["num_super"]
    C: sp.csr_matrix = gsp_out["C"]

    print(
        f"[{_TAG}] GSP done in {gsp_elapsed:.1f}s | "
        f"compression={gsp_stats['compression_ratio']*100:.1f}% | "
        f"super-nodes={num_super:,} | "
        f"avg_cluster={gsp_stats['avg_cluster_size']:.2f} | "
        f"singleton={gsp_stats.get('singleton_ratio', gsp_stats.get('singleton_fraction', 0))*100:.1f}%"
    )

    gsp_paper_stats: Dict[str, Any] = {
        "curvature_mode":           args.curvature_mode,
        "num_users":                num_users,
        "num_super_nodes":          num_super,
        "compression_ratio_pct":    round(gsp_stats["compression_ratio"] * 100, 2),
        "avg_cluster_size":         round(gsp_stats["avg_cluster_size"], 3),
        "singleton_ratio_pct":      round(gsp_stats.get("singleton_ratio", gsp_stats.get("singleton_fraction", 0)) * 100, 2),
        "largest_cluster":          gsp_stats.get("largest_cluster_size", gsp_stats.get("max_cluster_size", "?")),
        "gsp_preprocessing_time_s": round(gsp_elapsed, 3),
    }

    # ── Build edge indices ─────────────────────────────────────────────────────
    _section("Building Graph Edge Indices")

    # Baseline (original users × items)
    base_agg = (
        train_df.groupby(["UserID", "MovieID"], as_index=False)
        .agg(rating=("Rating", "mean"))
        .astype({"UserID": np.int64, "MovieID": np.int64})
    )
    edge_index_base = _build_edge_index(
        base_agg["UserID"].to_numpy(),
        base_agg["MovieID"].to_numpy(),
        num_users,
    )
    base_train_u = torch.tensor(base_agg["UserID"].to_numpy(), dtype=torch.long)
    base_train_i = torch.tensor(base_agg["MovieID"].to_numpy() + num_users, dtype=torch.long)
    base_train_y = torch.tensor(base_agg["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)

    # GSP (super-nodes × items)
    train_copy = train_df.copy()
    train_copy["super_idx"] = user_to_super[train_copy["UserID"].to_numpy(dtype=np.int64)]
    coarsened = (
        train_copy.groupby(["super_idx", "MovieID"], as_index=False)
        .agg(rating=("Rating", "mean"))
        .astype({"super_idx": np.int64, "MovieID": np.int64})
    )
    del train_copy
    edge_index_gsp = _build_edge_index(
        coarsened["super_idx"].to_numpy(),
        coarsened["MovieID"].to_numpy(),
        num_super,
    )
    gsp_train_super = torch.tensor(coarsened["super_idx"].to_numpy(), dtype=torch.long)
    gsp_train_item  = torch.tensor(coarsened["MovieID"].to_numpy() + num_super, dtype=torch.long)
    gsp_train_y     = torch.tensor(coarsened["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)

    nodes_orig = num_users + num_items
    nodes_gsp  = num_super + num_items
    edges_orig = int(base_agg.shape[0])
    edges_gsp  = int(coarsened.shape[0])

    gsp_paper_stats.update({
        "bipartite_nodes_original":     nodes_orig,
        "bipartite_nodes_gsp":          nodes_gsp,
        "bipartite_edges_original":     edges_orig,
        "bipartite_edges_gsp":          edges_gsp,
        "bipartite_edge_reduction_pct": round((1 - edges_gsp / max(edges_orig, 1)) * 100, 2),
    })
    print(
        f"[{_TAG}] Baseline graph : {nodes_orig:,} nodes | {edge_index_base.shape[1]:,} edge-slots"
    )
    print(
        f"[{_TAG}] GSP graph      : {nodes_gsp:,} nodes | {edge_index_gsp.shape[1]:,} edge-slots"
        f"  ({gsp_paper_stats['bipartite_edge_reduction_pct']:.1f}% reduction)"
    )
    _write_json(os.path.join(out_dir, "gsp_stats.json"), gsp_paper_stats)

    # Shared model config
    model_cfg = ModelConfig(
        emb_dim=args.emb_dim,
        hidden_dim=args.emb_dim * 2,
        out_dim=args.emb_dim,
        num_layers=args.num_layers,
        heads=4,
        dropout=0.1,
    )

    metrics_rows: List[Dict] = []
    speedup_rows: List[Dict] = []

    # ── GAT-specific: head-reduction fallback ──────────────────────────────────
    # GATConv materializes O(E × heads) attention tensors.  On ML-25M with
    # large fractions the bipartite graph can have millions of edges, causing
    # OOM even with gradient checkpointing.  We retry with 4→2→1 heads before
    # giving up, so GAT always produces a result.  The actual heads used are
    # recorded in the output row for transparency.
    _GAT_HEAD_FALLBACKS = [4, 2, 1]

    def _build_gat_variants(num_nodes_: int) -> List["nn.Module"]:
        """Return a list of GAT models with decreasing heads; others return [model]."""
        variants = []
        for h in _GAT_HEAD_FALLBACKS:
            cfg_ = ModelConfig(
                emb_dim=model_cfg.emb_dim,
                hidden_dim=model_cfg.hidden_dim,
                out_dim=model_cfg.out_dim,
                num_layers=model_cfg.num_layers,
                heads=h,
                dropout=model_cfg.dropout,
            )
            try:
                variants.append((h, get_model("gat", num_nodes_, cfg_)))
            except Exception:
                pass
        return variants

    # ═══════════════════════════════════════════════════════════════════════════
    # Per-model loop
    # ═══════════════════════════════════════════════════════════════════════════
    for model_name in models_to_run:
        _section(f"MODEL: {model_name.upper()}")

        # ── Baseline ──────────────────────────────────────────────────────────
        print(f"[{_TAG}] {model_name} | Baseline  ({nodes_orig:,} nodes, {edges_orig:,} edges)")
        try:
            base_model = get_model(model_name, nodes_orig, model_cfg)
        except Exception as exc:
            print(f"[{_TAG}] WARNING: cannot build '{model_name}': {exc}  Skipping.")
            continue

        train_cfg_base = TrainConfig(
            epochs=args.epochs, lr=args.lr, weight_decay=1e-5,
            batch_size=args.batch_size, neg_ratio=args.neg_ratio,
            emb_l2_weight=1e-5, seed=args.seed, use_amp=(not args.no_amp),
            checkpoint_dir=os.path.join(out_dir, "checkpoints"),
            save_epoch_checkpoints=False,
            metrics_jsonl_path=os.path.join(out_dir, f"training_metrics_{model_name}_baseline.jsonl"),
            training_log_path=os.path.join(out_dir,  f"training_log_{model_name}_baseline.txt"),
            device=device,
            early_stopping_patience=args.early_stopping_patience,
        )

        reset_gpu_peak_memory()
        t_base_start = time.perf_counter()
        cpu_base_start = _proc_cpu_time_s()

        # For GAT: try head counts 4→2→1 until one succeeds.
        _gat_heads_used_base: int = model_cfg.heads
        _base_attempt_exc: Optional[Exception] = None
        if model_name == "gat":
            _gat_variants_base = _build_gat_variants(nodes_orig)
            _base_success = False
            for _h, _cand_model in _gat_variants_base:
                try:
                    print(f"[{_TAG}] gat baseline attempt: heads={_h}")
                    reset_gpu_peak_memory()
                    t_base_start = time.perf_counter()
                    cpu_base_start = _proc_cpu_time_s()
                    _ = train_model(
                        _cand_model,
                        edge_index=edge_index_base,
                        train_user_nodes=base_train_u,
                        train_item_nodes=base_train_i,
                        train_ratings=base_train_y,
                        config=train_cfg_base,
                        run_name=f"{model_name}_baseline",
                    )
                    base_model = _cand_model
                    _gat_heads_used_base = _h
                    _base_success = True
                    break
                except Exception as _exc:
                    print(f"[{_TAG}] gat baseline heads={_h} failed: {_exc}")
                    _base_attempt_exc = _exc
                    try:
                        del _cand_model
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
            if not _base_success:
                print(f"[{_TAG}] gat baseline failed at all head counts; skipping.")
                metrics_rows.append({"model": model_name, "run_type": "baseline",
                                     "Error": str(_base_attempt_exc)[:120]})
                continue
        else:
            try:
                _ = train_model(
                    base_model,
                    edge_index=edge_index_base,
                    train_user_nodes=base_train_u,
                    train_item_nodes=base_train_i,
                    train_ratings=base_train_y,
                    config=train_cfg_base,
                    run_name=f"{model_name}_baseline",
                )
            except Exception as exc:
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
                print(f"[{_TAG}] {'OOM' if is_oom else 'ERROR'} during {model_name} BASELINE: {exc}")
                try:
                    del base_model
                except Exception:
                    pass
                torch.cuda.empty_cache()
                metrics_rows.append({"model": model_name, "run_type": "baseline",
                                     "Error": ("OOM" if is_oom else str(exc)[:120])})
                continue

        t_base = time.perf_counter() - t_base_start
        cpu_base_s = _proc_cpu_time_s() - cpu_base_start
        gpu_base_mb = gpu_max_memory_allocated_mb()

        try:
            z_base = _infer(base_model, edge_index_base, device)
            ue_base = _l2_norm(z_base[:num_users])
            ie_base = _l2_norm(z_base[num_users:])
            rank_base = _eval_multi_k(ue_base, ie_base, test_pos, seen_train, args.eval_negatives, args.seed)
            reg_base  = _compute_rmse_mae(test_df, ue_base, ie_base, train_mean)
        except Exception as exc:
            print(f"[{_TAG}] ERROR during {model_name} BASELINE inference: {exc}")
            try:
                del base_model
            except Exception:
                pass
            torch.cuda.empty_cache()
            metrics_rows.append({"model": model_name, "run_type": "baseline",
                                  "Error": str(exc)[:120]})
            continue

        del base_model
        torch.cuda.empty_cache()

        _rank_str_base = "  ".join(f"NDCG@{k}={rank_base.get(f'NDCG@{k}', 0):.4f}" for k in _EVAL_KS)
        print(
            f"[{_TAG}] {model_name} BASELINE  "
            f"{_rank_str_base}  "
            f"RMSE={reg_base['RMSE']:.4f}  MAE={reg_base['MAE']:.4f}  "
            f"time={t_base:.1f}s  GPU={gpu_base_mb:.0f}MB"
        )
        row_base: Dict = {
            "model": model_name, "run_type": "baseline",
            **{k: v for k, v in rank_base.items() if k != "UsersEvaluated"},
            **reg_base,
            "training_time_s": round(t_base, 3),
            "gpu_peak_MB": round(gpu_base_mb, 1),
            "cpu_time_s": round(cpu_base_s, 3),
            "cpu_efficiency_pct": round(cpu_base_s / max(t_base, 1e-9) * 100, 1),
            "ram_rss_MB": round(rss_mb(), 1),
            **(  # record which head count succeeded for GAT
                {"gat_heads_used": _gat_heads_used_base} if model_name == "gat" else {}
            ),
        }
        metrics_rows.append(row_base)

        # ── GSP ───────────────────────────────────────────────────────────────
        print(f"\n[{_TAG}] {model_name} | GSP  ({nodes_gsp:,} nodes, {edges_gsp:,} edges)")
        try:
            gsp_model = get_model(model_name, nodes_gsp, model_cfg)
        except Exception as exc:
            print(f"[{_TAG}] WARNING: cannot build GSP '{model_name}': {exc}  Skipping GSP.")
            continue

        train_cfg_gsp = TrainConfig(
            epochs=args.epochs, lr=args.lr, weight_decay=1e-5,
            batch_size=args.batch_size, neg_ratio=args.neg_ratio,
            emb_l2_weight=1e-5, seed=args.seed, use_amp=(not args.no_amp),
            checkpoint_dir=os.path.join(out_dir, "checkpoints"),
            save_epoch_checkpoints=False,
            metrics_jsonl_path=os.path.join(out_dir, f"training_metrics_{model_name}_gsp.jsonl"),
            training_log_path=os.path.join(out_dir,  f"training_log_{model_name}_gsp.txt"),
            device=device,
            early_stopping_patience=args.early_stopping_patience,
        )

        reset_gpu_peak_memory()
        t_gsp_train_start = time.perf_counter()
        cpu_gsp_start = _proc_cpu_time_s()

        # For GAT: try head counts 4→2→1 until one succeeds (same fallback as baseline).
        _gat_heads_used_gsp: int = _gat_heads_used_base if model_name == "gat" else model_cfg.heads
        if model_name == "gat":
            _gat_variants_gsp = _build_gat_variants(nodes_gsp)
            _gsp_success = False
            _gsp_attempt_exc: Optional[Exception] = None
            for _h, _cand_gsp in _gat_variants_gsp:
                try:
                    print(f"[{_TAG}] gat gsp attempt: heads={_h}")
                    reset_gpu_peak_memory()
                    t_gsp_train_start = time.perf_counter()
                    cpu_gsp_start = _proc_cpu_time_s()
                    _ = train_model(
                        _cand_gsp,
                        edge_index=edge_index_gsp,
                        train_user_nodes=gsp_train_super,
                        train_item_nodes=gsp_train_item,
                        train_ratings=gsp_train_y,
                        config=train_cfg_gsp,
                        run_name=f"{model_name}_gsp",
                    )
                    gsp_model = _cand_gsp
                    _gat_heads_used_gsp = _h
                    _gsp_success = True
                    break
                except Exception as _exc:
                    print(f"[{_TAG}] gat gsp heads={_h} failed: {_exc}")
                    _gsp_attempt_exc = _exc
                    try:
                        del _cand_gsp
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
            if not _gsp_success:
                print(f"[{_TAG}] gat gsp failed at all head counts; skipping.")
                metrics_rows.append({"model": model_name, "run_type": "gsp_projected",
                                     "Error": str(_gsp_attempt_exc)[:120]})
                continue
        else:
            try:
                _ = train_model(
                    gsp_model,
                    edge_index=edge_index_gsp,
                    train_user_nodes=gsp_train_super,
                    train_item_nodes=gsp_train_item,
                    train_ratings=gsp_train_y,
                    config=train_cfg_gsp,
                    run_name=f"{model_name}_gsp",
                )
            except Exception as exc:
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
                print(f"[{_TAG}] {'OOM' if is_oom else 'ERROR'} during {model_name} GSP: {exc}")
                try:
                    del gsp_model
                except Exception:
                    pass
                torch.cuda.empty_cache()
                metrics_rows.append({"model": model_name, "run_type": "gsp_projected",
                                     "Error": ("OOM" if is_oom else str(exc)[:120])})
                continue

        t_gsp_train = time.perf_counter() - t_gsp_train_start
        cpu_gsp_s = _proc_cpu_time_s() - cpu_gsp_start
        gpu_gsp_mb = gpu_max_memory_allocated_mb()

        try:
            # Projection: super-node embeddings → per-user embeddings via C
            z_gsp = _infer(gsp_model, edge_index_gsp, device)
            H_super = z_gsp[:num_super].astype(np.float32)
            H_final, proj_t = project_embeddings(H_super, C)
        except Exception as exc:
            print(f"[{_TAG}] ERROR during {model_name} GSP inference/projection: {exc}")
            try:
                del gsp_model
            except Exception:
                pass
            torch.cuda.empty_cache()
            metrics_rows.append({"model": model_name, "run_type": "gsp_projected",
                                  "Error": str(exc)[:120]})
            continue

        del gsp_model
        torch.cuda.empty_cache()

        ie_gsp   = _l2_norm(z_gsp[num_super:])
        ue_proj  = _l2_norm(H_final)
        rank_gsp = _eval_multi_k(ue_proj, ie_gsp, test_pos, seen_train, args.eval_negatives, args.seed)
        reg_gsp  = _compute_rmse_mae(test_df, ue_proj, ie_gsp, train_mean)

        _rank_str_gsp = "  ".join(f"NDCG@{k}={rank_gsp.get(f'NDCG@{k}', 0):.4f}" for k in _EVAL_KS)
        print(
            f"[{_TAG}] {model_name} GSP+PROJ  "
            f"{_rank_str_gsp}  "
            f"RMSE={reg_gsp['RMSE']:.4f}  MAE={reg_gsp['MAE']:.4f}  "
            f"time={t_gsp_train:.1f}s  GPU={gpu_gsp_mb:.0f}MB  proj={proj_t:.3f}s"
        )
        row_gsp: Dict = {
            "model": model_name, "run_type": "gsp_projected",
            **{k: v for k, v in rank_gsp.items() if k != "UsersEvaluated"},
            **reg_gsp,
            "training_time_s": round(t_gsp_train, 3),
            "gpu_peak_MB": round(gpu_gsp_mb, 1),
            "cpu_time_s": round(cpu_gsp_s, 3),
            "cpu_efficiency_pct": round(cpu_gsp_s / max(t_gsp_train, 1e-9) * 100, 1),
            "ram_rss_MB": round(rss_mb(), 1),
            "projection_time_s": round(proj_t, 4),
            **(  # record which head count succeeded for GAT
                {"gat_heads_used": _gat_heads_used_gsp} if model_name == "gat" else {}
            ),
        }
        metrics_rows.append(row_gsp)

        # ── Speedup record ────────────────────────────────────────────────────
        speedup = t_base / max(t_gsp_train, 1e-9)
        speedup_rows.append({
            "model":                      model_name,
            "curvature_mode":             args.curvature_mode,
            "training_time_baseline_s":   round(t_base, 3),
            "training_time_gsp_s":        round(t_gsp_train, 3),
            "speedup_factor":             round(speedup, 4),
            "gsp_preprocessing_s":        round(gsp_elapsed, 3),
            "net_time_saved_s":           round(t_base - t_gsp_train - gsp_elapsed, 3),
            "gpu_baseline_MB":            round(gpu_base_mb, 1),
            "gpu_gsp_MB":                 round(gpu_gsp_mb, 1),
            "gpu_reduction_pct":          round((1 - gpu_gsp_mb / max(gpu_base_mb, 1)) * 100, 2),
            "cpu_time_baseline_s":        round(cpu_base_s, 3),
            "cpu_time_gsp_s":             round(cpu_gsp_s, 3),
            **{f"Precision@{k}_baseline": round(rank_base.get(f"Precision@{k}", 0), 4) for k in _EVAL_KS},
            **{f"Precision@{k}_gsp":      round(rank_gsp.get(f"Precision@{k}", 0), 4) for k in _EVAL_KS},
            **{f"NDCG@{k}_baseline":      round(rank_base.get(f"NDCG@{k}", 0), 4) for k in _EVAL_KS},
            **{f"NDCG@{k}_gsp":           round(rank_gsp.get(f"NDCG@{k}", 0), 4) for k in _EVAL_KS},
        })

        # ── Analytics ─────────────────────────────────────────────────────────
        try:
            run_analytics_pipeline(
                model_name=model_name,
                output_dir=out_dir,
                base_user_emb=ue_base,
                base_item_emb=ie_base,
                gsp_user_emb=ue_proj,
                gsp_item_emb=ie_gsp,
                gsp_super_emb=H_super,
                num_users=num_users,
                num_items=num_items,
                num_super=num_super,
                user_to_super=user_to_super,
                gsp_out=gsp_out,
                base_edge_count=edges_orig,
                gsp_edge_count=edges_gsp,
                seen_train=seen_train,
                test_positives=test_pos,
                baseline_summary=rank_base,
                gsp_summary=rank_gsp,
                curvature_mode=args.curvature_mode,
                fraction=args.target_fraction if args.target_fraction is not None else 1.0,
                min_shared=args.min_shared,
                dataset_name="movielens25m",
            )
        except Exception as _analytics_exc:
            print(f"[{_TAG}] WARNING: analytics failed for {model_name}: {_analytics_exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Final output
    # ─────────────────────────────────────────────────────────────────────────
    _section("RESULTS SUMMARY")
    _print_paper_table(metrics_rows)

    _save_csv(metrics_rows, os.path.join(out_dir, "results_table.csv"))
    _save_csv(speedup_rows, os.path.join(out_dir, "speedup_results.csv"))

    total_wall = time.perf_counter() - t_wall
    _write_json(os.path.join(out_dir, "full_results.json"), {
        "dataset": dataset_summary,
        "gsp": gsp_paper_stats,
        "metrics": metrics_rows,
        "speedup": speedup_rows,
        "total_wall_time_s": round(total_wall, 2),
    })

    print(f"\n[{_TAG}] Total wall-clock time : {total_wall/60:.1f} min")
    print(f"[{_TAG}] Results written to    : {out_dir}/")
    print(f"[{_TAG}]   results_table.csv")
    print(f"[{_TAG}]   speedup_results.csv")
    print(f"[{_TAG}]   full_results.json")

    print("\n--- GSP Compression Summary ---")
    for k, v in gsp_paper_stats.items():
        print(f"  {k:<45}: {v}")


if __name__ == "__main__":
    main()
