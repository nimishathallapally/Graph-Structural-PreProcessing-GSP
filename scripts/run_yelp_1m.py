#!/usr/bin/env python3
"""
run_yelp_1m.py  –  Fast paper-results script for the Yelp dataset
==================================================================

Targets ~1 M interactions (configurable) by taking the most-active
users after k-core filtering.  Skips slow semantic-feature extraction.
Outputs a paper-ready LaTeX table + CSVs.

Typical usage
-------------
    # Quick single-model run (~1 M interactions, LightGCN only)
    python scripts/run_yelp_1m.py --data_dir ./data --epochs 30

    # All four models
    python scripts/run_yelp_1m.py --data_dir ./data \\
        --models lightgcn,gat,graphsage,gcn --epochs 30

    # Explicitly set interaction target
    python scripts/run_yelp_1m.py --data_dir ./data \\
        --target_interactions 1000000 --models lightgcn --epochs 50

    # Resume / reuse GSP cache from a previous run
    python scripts/run_yelp_1m.py --data_dir ./data \\
        --output_dir outputs/yelp_1m --epochs 30
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

# ── make gsprec importable ────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gsprec.data.yelp_dataset import build_yelp_dataset, compute_bipartite_graph_stats
from gsprec.graph.gsp_ops import gsp_preprocess
from gsprec.graph.embedding_store import project_embeddings
from gsprec.models.architectures import get_model, ModelConfig
from gsprec.models.trainer import TrainConfig, train_model
from gsprec.models.gnn import RankingEvalConfig, evaluate_ranking_from_embeddings, rmse_mae
from gsprec.utils.hardware_info import (
    collect_hardware_info, rss_mb,
    gpu_max_memory_allocated_mb, reset_gpu_peak_memory,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Yelp ~1M-interaction paper-results pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",  default="./data",
                   help="Directory with Yelp JSONL files (or parent of yelp_dataset/)")
    p.add_argument("--output_dir", default="outputs/yelp_1m",
                   help="Where to write results")
    p.add_argument("--target_interactions", type=int, default=0,
                   help="Approx interaction count via top-user selection; 0 = use full filtered dataset")
    p.add_argument("--max_users", type=int, default=0,
                   help="Hard cap on user count (0 = derive from --target_interactions)")
    p.add_argument("--min_interactions", type=int, default=5,
                   help="k-core filter: minimum interactions per user/item")
    p.add_argument("--models", default="lightgcn,gat,graphsage,gcn",
                   help="Comma-separated model list: lightgcn,gat,graphsage,gcn")
    p.add_argument("--epochs", type=int, default=200,
                   help="Training epochs per model")
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch_size", type=int, default=65536)
    p.add_argument("--neg_ratio", type=int, default=4)
    p.add_argument("--implicit_threshold", type=float, default=3.5,
                   help="Minimum star rating to treat as a positive interaction")
    p.add_argument("--eval_k", type=int, default=10)
    p.add_argument("--eval_negatives", type=int, default=99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stopping_patience", type=int, default=10,
                   help="Stop after this many epochs with no loss improvement. 0 = disabled.")
    p.add_argument("--curvature_percentile", type=float, default=97.0,
                   help="Stage-I HC percentile (higher = fewer edges retained = smaller clusters). "
                        "Ignored when --curvature_topk is set. For large sparse graphs like Yelp, "
                        "prefer --curvature_topk to avoid the adaptive-retry heuristic.")
    p.add_argument("--curvature_topk", type=int, default=None,
                   help="Keep exactly this many top-curvature UU edges (absolute count). "
                        "Overrides --curvature_percentile and disables the adaptive-retry. "
                        "For Yelp-scale (277k users) aim for avg degree < 1, e.g. 50000-100000.")
    p.add_argument("--max_cluster_size", type=int, default=10,
                   help="Cap on users per super-node cluster (0 = no cap). "
                        "Only effective when HC subgraph forms small components.")
    p.add_argument("--er_eigvecs", type=int, default=16,
                   help="Effective-resistance eigenvectors (arpack/lobpcg)")
    p.add_argument("--er_node_limit", type=int, default=0,
                   help="Skip ER when num_users > this; 0 = always run ER (default)")
    p.add_argument("--er_solver", default="jl",
                   choices=["arpack", "lobpcg", "jl"],
                   help="ER solver: arpack (accurate/slow), lobpcg (no factorisation), "
                        "jl (JL-sketch via MINRES, fastest for large graphs)")
    p.add_argument("--er_sketches", type=int, default=32,
                   help="Number of JL random probes (jl solver only)")
    p.add_argument("--min_shared", type=int, default=2,
                   help="Min shared interactions for a UU edge to be retained")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable mixed-precision training")
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
    print(f"\n[1M] {bar}")
    print(f"[1M] ║  {title}  ║")
    print(f"[1M] {bar}")


def _l2_norm(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Resource monitoring
# ─────────────────────────────────────────────────────────────────────────────

def _proc_cpu_time_s() -> float:
    """Return this process's cumulative CPU time (user+sys) in seconds."""
    try:
        with open("/proc/self/stat") as fh:
            fields = fh.read().split()
        ticks_per_s = os.sysconf("SC_CLK_TCK")
        return (int(fields[13]) + int(fields[14])) / ticks_per_s
    except Exception:
        return 0.0


def _sys_loadavg() -> str:
    """1-minute load average string (Linux only)."""
    try:
        return f"{os.getloadavg()[0]:.2f}"
    except Exception:
        return "n/a"


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _subset_to_target(
    ratings_df: pd.DataFrame,
    target: int,
    max_users: int,
) -> Tuple[pd.DataFrame, int, int]:
    """Keep the most-active users until we hit ~target interactions.

    If target <= 0 and max_users <= 0, the full filtered dataset is used.
    Returns (subset_df, num_users, num_items) with contiguous IDs.
    """
    counts = ratings_df.groupby("UserID")["BusinessID"].count().sort_values(ascending=False)

    if max_users > 0:
        selected_users = counts.iloc[:max_users].index
    elif target <= 0:
        # Use all users (full filtered dataset)
        selected_users = counts.index
    else:
        cumulative = counts.cumsum()
        cutoff = int((cumulative >= target).idxmax())
        pos = int(counts.index.get_loc(cutoff))
        # include the user that pushed us over the target
        selected_users = counts.iloc[: pos + 1].index

    sub = ratings_df[ratings_df["UserID"].isin(selected_users)].copy()

    # Remap to contiguous 0-based IDs
    uniq_u = np.sort(sub["UserID"].unique())
    uniq_i = np.sort(sub["BusinessID"].unique())
    sub["UserID"]  = np.searchsorted(uniq_u, sub["UserID"].to_numpy())
    sub["BusinessID"] = np.searchsorted(uniq_i, sub["BusinessID"].to_numpy())
    sub = sub.reset_index(drop=True)
    return sub, int(len(uniq_u)), int(len(uniq_i))


def _split_leave_one_out(
    df: pd.DataFrame, threshold: float, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-One-Out split (standard RecSys protocol: NCF, LightGCN, BPR).

    For each user who has at least 2 positive interactions (rating >= threshold),
    randomly hold out exactly 1 positive as the test item.  All other interactions
    (positive and negative) remain in the training set so the model sees the full
    graph structure.  Users with fewer than 2 positives are kept in training only.

    Returns
    -------
    train_df : all rows except the held-out ones
    test_df  : one row per qualifying user (the held-out positive)
    """
    rng = np.random.default_rng(seed)
    test_idx: List[int] = []
    for uid, grp in df.groupby("UserID"):
        pos_mask = grp["Rating"].to_numpy(dtype=np.float32) >= threshold
        pos_idxs = grp.index[pos_mask].to_numpy()
        if len(pos_idxs) >= 2:  # need at least 1 positive left in training
            chosen = int(rng.choice(pos_idxs))
            test_idx.append(chosen)
    mask = df.index.isin(test_idx)
    return df.loc[~mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)


def _build_seen(df: pd.DataFrame) -> Dict[int, Set[int]]:
    seen: Dict[int, Set[int]] = {}
    for row in df.itertuples(index=False):
        seen.setdefault(int(row.UserID), set()).add(int(row.BusinessID))
    return seen


def _build_positives(df: pd.DataFrame, threshold: float) -> Dict[int, List[int]]:
    pos: Dict[int, List[int]] = {}
    for row in df.itertuples(index=False):
        if float(row.Rating) >= threshold:
            pos.setdefault(int(row.UserID), []).append(int(row.BusinessID))
    return pos


def _build_edge_index(
    df: pd.DataFrame, user_col: str, item_col: str, item_offset: int
) -> torch.Tensor:
    su = df[user_col].to_numpy(dtype=np.int64)
    it = df[item_col].to_numpy(dtype=np.int64) + item_offset
    src = np.concatenate([su, it])
    dst = np.concatenate([it, su])
    return torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rmse_mae(
    test_df: pd.DataFrame,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    train_mean: float,
) -> Dict[str, float]:
    y_true, y_pred = [], []
    for row in test_df.itertuples(index=False):
        u, i, r = int(row.UserID), int(row.BusinessID), float(row.Rating)
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


_EVAL_KS: Tuple[int, ...] = (10, 20, 50)


def _infer(model: torch.nn.Module, edge_index: torch.Tensor, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(edge_index.to(device)).detach().cpu().float().numpy()


def _eval_multi_k(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    test_positives: Dict[int, List[int]],
    seen_positives: Dict[int, Set[int]],
    num_negatives: int,
    seed: int,
    ks: Tuple[int, ...] = _EVAL_KS,
) -> Dict[str, float]:
    """Evaluate ranking metrics at multiple cutoffs, sharing the same candidate pool."""
    merged: Dict[str, float] = {}
    users_eval = 0.0
    for k in ks:
        cfg = RankingEvalConfig(k=k, num_negatives=num_negatives, seed=seed)
        m = evaluate_ranking_from_embeddings(user_emb, item_emb, test_positives, seen_positives, cfg)
        users_eval = m.pop("UsersEvaluated", users_eval)
        merged.update(m)
    merged["UsersEvaluated"] = users_eval
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Paper table printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_paper_table(rows: List[Dict], ks: Tuple[int, ...] = _EVAL_KS) -> None:
    """Print a LaTeX-ready results table to stdout."""
    cols_rank = [f"{m}@{k}" for k in ks for m in ("Precision", "Recall", "NDCG", "HitRate")]
    cols_reg  = ["RMSE", "MAE"]
    
    header = (
        f"{'Model':<18}  {'Type':<16}  "
        + "  ".join(f"{c:>14}" for c in cols_rank + cols_reg)
        + "  " + f"{'Train(s)':>10}"
    )
    print("\n" + "=" * len(header))
    print("PAPER RESULTS TABLE")
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

    # LaTeX version
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
            f"{r.get(c, float('nan')):.4f}"
            for c in cols_rank + cols_reg
        )
        print(f"{model_tex} & {type_tex} & {vals} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def _save_results_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    _mkdir(os.path.dirname(os.path.abspath(path)))
    # Collect all unique keys across all rows to handle rows with different fields
    fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _save_gsp_stats_csv(gsp_stats: Dict, dataset_summary: Dict, path: str) -> None:
    _mkdir(os.path.dirname(os.path.abspath(path)))
    combined = {**dataset_summary, **gsp_stats}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in combined.items():
            w.writerow([k, v])


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    t_wall = time.perf_counter()

    out_dir = args.output_dir
    _mkdir(out_dir)
    _mkdir(os.path.join(out_dir, "checkpoints"))
    _mkdir(os.path.join(out_dir, "cache"))

    rng_np = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_to_run = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    target_desc = "FULL filtered dataset" if args.target_interactions <= 0 else f"~{args.target_interactions:,} interactions"
    print(f"[1M] Device : {device}")
    print(f"[1M] Models : {models_to_run}")
    print(f"[1M] Epochs : {args.epochs}")
    print(f"[1M] Target : {target_desc}")
    print(f"[1M] Load1  : {_sys_loadavg()} (1-min load avg)")
    print(f"[1M] Output : {out_dir}")

    # Save hardware info
    hw = collect_hardware_info()
    hw.update({"device": device, "seed": args.seed, "models": models_to_run,
                "target_interactions": args.target_interactions})
    _write_json(os.path.join(out_dir, "hardware_info.json"), hw)

    # ── STAGE 0: Load full Yelp dataset ──────────────────────────────────────
    _section("STAGE 0: Dataset Loading")
    t0 = time.perf_counter()
    yelp = build_yelp_dataset(
        data_dir=args.data_dir,
        min_user_interactions=args.min_interactions,
        min_business_interactions=args.min_interactions,
        implicit_threshold=args.implicit_threshold,
    )
    load_time = time.perf_counter() - t0
    ratings_full: pd.DataFrame = yelp["ratings_df"]
    print(
        f"[1M] Full dataset: {yelp['num_users']:,} users | {yelp['num_items']:,} items | "
        f"{len(ratings_full):,} interactions | load={load_time:.1f}s"
    )

    # ── STAGE 0b: Subset to target interaction count ──────────────────────────
    _section("STAGE 0b: Subsetting to ~1 M Interactions")
    t0 = time.perf_counter()
    ratings_df, num_users, num_items = _subset_to_target(
        ratings_full,
        target=args.target_interactions,
        max_users=args.max_users,
    )
    subset_time = time.perf_counter() - t0
    n_interactions = len(ratings_df)
    sparsity = 1.0 - n_interactions / max(num_users * num_items, 1)
    print(
        f"[1M] Subset  : {num_users:,} users | {num_items:,} items | "
        f"{n_interactions:,} interactions | sparsity={sparsity:.4%} | {subset_time:.2f}s"
    )
    del ratings_full  # free memory

    dataset_summary: Dict[str, Any] = {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": n_interactions,
        "sparsity": round(sparsity, 6),
        "implicit_threshold": args.implicit_threshold,
        "min_interactions": args.min_interactions,
        "load_time_s": round(load_time, 3),
        "subset_time_s": round(subset_time, 3),
    }

    # ── Train/test split (Leave-One-Out) ──────────────────────────────────────
    # Standard RecSys protocol: hold out 1 positive per user, eval rank vs 99 negatives
    train_df, test_df = _split_leave_one_out(ratings_df, threshold=args.implicit_threshold, seed=args.seed)
    seen_train    = _build_seen(train_df)
    test_pos      = _build_positives(test_df, args.implicit_threshold)
    train_mean    = float(train_df["Rating"].mean())

    dataset_summary.update({
        "train_interactions": len(train_df),
        "test_interactions":  len(test_df),
        "test_users_with_positives": len(test_pos),
        "split_protocol": "leave-one-out",
    })
    print(
        f"[1M] Split   : LOO | train={len(train_df):,} | test={len(test_df):,} | "
        f"test-positive users={len(test_pos):,}"
    )
    _write_json(os.path.join(out_dir, "dataset_stats.json"), dataset_summary)

    # ── STAGE 1: Bipartite graph stats ───────────────────────────────────────
    _section("STAGE 1: Bipartite Graph")
    graph_stats = compute_bipartite_graph_stats(train_df, num_users, num_items)
    dataset_summary.update(graph_stats)
    print(
        f"[1M] Graph   : {graph_stats['num_nodes']:,} nodes | "
        f"{graph_stats['num_edges']:,} edges | "
        f"avg-deg={graph_stats['avg_degree']:.2f} | "
        f"density={graph_stats['density']:.2e} | "
        f"components={graph_stats['num_components']:,} | "
        f"mem={graph_stats['graph_memory_MB']:.1f}MB"
    )

    # ── STAGE I–II: GSP Preprocessing ────────────────────────────────────────
    _section("STAGE I-II: GSP Preprocessing")
    t_gsp = time.perf_counter()
    gsp_out = gsp_preprocess(
        ratings_df=train_df,
        num_users=num_users,
        num_items=num_items,
        implicit_threshold=args.implicit_threshold,
        alpha=0.5,
        curvature_percentile=args.curvature_percentile,
        curvature_topk=args.curvature_topk,
        importance_percentile=50.0,
        importance_topk=None,
        er_num_eigenvectors=args.er_eigvecs,
        max_cluster_size=args.max_cluster_size,
        min_shared_interactions=args.min_shared,
        er_node_limit=args.er_node_limit,
        er_solver=args.er_solver,
        er_sketches=args.er_sketches,
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
        f"[1M] GSP done in {gsp_elapsed:.1f}s | "
        f"compression={gsp_stats['compression_ratio']*100:.1f}% | "
        f"super-nodes={num_super:,} | "
        f"avg_cluster={gsp_stats['avg_cluster_size']:.2f} | "
        f"singleton={gsp_stats.get('singleton_ratio', gsp_stats.get('singleton_fraction', 0))*100:.1f}%"
    )

    gsp_paper_stats = {
        "num_users":               num_users,
        "num_super_nodes":         num_super,
        "compression_ratio_pct":   round(gsp_stats["compression_ratio"] * 100, 2),
        "avg_cluster_size":        round(gsp_stats["avg_cluster_size"], 3),
        "singleton_ratio_pct":     round(gsp_stats.get("singleton_ratio", gsp_stats.get("singleton_fraction", 0)) * 100, 2),
        "largest_cluster":         gsp_stats.get("largest_cluster_size", gsp_stats.get("max_cluster_size", "?")),
        "edge_retention_pct":      round(gsp_stats.get("edge_retention_ratio", gsp_stats.get("uu_hc_fraction", 0)) * 100, 2),
        "uu_edges_original":       gsp_stats.get("uu_edges_before_shared", gsp_stats.get("uu_edges_all", 0)),
        "uu_edges_after_filter":   gsp_stats.get("uu_edges_all", 0),
        "uu_edges_hc":             gsp_stats.get("uu_edges_hc", 0),
        "uu_edges_pruned":         gsp_stats.get("uu_edges_pruned", 0),
        "gsp_preprocessing_time_s": round(gsp_elapsed, 3),
    }

    # ── Build edge indices ─────────────────────────────────────────────────────
    _section("Building Graph Edge Indices")

    # Baseline (original users)
    base_agg = (
        train_df.groupby(["UserID", "BusinessID"], as_index=False)
        .agg(rating=("Rating", "mean"))
        .rename(columns={"UserID": "u_idx", "BusinessID": "i_idx"})
        .astype({"u_idx": np.int64, "i_idx": np.int64})
    )
    edge_index_base = _build_edge_index(base_agg, "u_idx", "i_idx", num_users)

    # GSP (super-nodes)
    train_copy = train_df.copy()
    train_copy["super_idx"] = user_to_super[train_copy["UserID"].to_numpy(dtype=np.int64)]
    coarsened = (
        train_copy.groupby(["super_idx", "BusinessID"], as_index=False)
        .agg(rating=("Rating", "mean"))
        .rename(columns={"BusinessID": "i_idx"})
        .astype({"super_idx": np.int64, "i_idx": np.int64})
    )
    edge_index_gsp = _build_edge_index(coarsened, "super_idx", "i_idx", num_super)
    del train_copy

    nodes_orig = num_users  + num_items
    nodes_gsp  = num_super  + num_items
    edges_orig = int(base_agg.shape[0])
    edges_gsp  = int(coarsened.shape[0])

    gsp_paper_stats.update({
        "bipartite_nodes_original": nodes_orig,
        "bipartite_nodes_gsp":      nodes_gsp,
        "bipartite_edges_original": edges_orig,
        "bipartite_edges_gsp":      edges_gsp,
        "bipartite_edge_reduction_pct": round((1 - edges_gsp / max(edges_orig, 1)) * 100, 2),
    })

    print(
        f"[1M] Baseline graph : {nodes_orig:,} nodes | {edge_index_base.shape[1]:,} edge-slots"
    )
    print(
        f"[1M] GSP graph      : {nodes_gsp:,} nodes | {edge_index_gsp.shape[1]:,} edge-slots"
        f"  ({gsp_paper_stats['bipartite_edge_reduction_pct']:.1f}% reduction)"
    )

    _write_json(os.path.join(out_dir, "gsp_stats.json"), gsp_paper_stats)

    # Model config (shared)
    model_cfg = ModelConfig(
        emb_dim=args.emb_dim,
        hidden_dim=args.emb_dim * 2,
        out_dim=args.emb_dim,
        num_layers=args.num_layers,
        heads=4,
        dropout=0.1,
    )

    base_train_y = torch.tensor(
        base_agg["rating"].to_numpy(dtype=np.float32), dtype=torch.float32
    )
    gsp_train_y = torch.tensor(
        coarsened["rating"].to_numpy(dtype=np.float32), dtype=torch.float32
    )
    gsp_train_super = torch.tensor(
        coarsened["super_idx"].to_numpy(dtype=np.int64), dtype=torch.long
    )
    gsp_train_item = torch.tensor(
        coarsened["i_idx"].to_numpy(dtype=np.int64) + num_super, dtype=torch.long
    )

    metrics_rows: List[Dict] = []
    speedup_rows: List[Dict] = []

    # ═══════════════════════════════════════════════════════════════════════════
    # Per-model loop
    # ═══════════════════════════════════════════════════════════════════════════

    for model_name in models_to_run:
        _section(f"MODEL: {model_name.upper()}")

        # ── IV. Baseline ──────────────────────────────────────────────────────
        print(f"[1M] {model_name} | Baseline  ({nodes_orig:,} nodes, {edges_orig:,} edges)")
        try:
            base_model = get_model(model_name, nodes_orig, model_cfg)
        except Exception as exc:
            print(f"[1M] WARNING: cannot build '{model_name}': {exc}  Skipping.")
            continue

        train_cfg_base = TrainConfig(
            epochs=args.epochs, lr=args.lr, weight_decay=1e-5,
            batch_size=args.batch_size, neg_ratio=args.neg_ratio,
            emb_l2_weight=1e-5, seed=args.seed, use_amp=(not args.no_amp),
            checkpoint_dir=os.path.join(out_dir, "checkpoints"),
            save_epoch_checkpoints=False,  # avoid disk-full on root partition
            metrics_jsonl_path=os.path.join(out_dir, f"training_metrics_{model_name}_baseline.jsonl"),
            training_log_path=os.path.join(out_dir,  f"training_log_{model_name}_baseline.txt"),
            device=device,
            early_stopping_patience=args.early_stopping_patience,
        )

        reset_gpu_peak_memory()
        t_base_start = time.perf_counter()
        cpu_base_start = _proc_cpu_time_s()
        _ = train_model(
            base_model,
            edge_index=edge_index_base,
            train_user_nodes=torch.tensor(
                base_agg["u_idx"].to_numpy(dtype=np.int64), dtype=torch.long
            ),
            train_item_nodes=torch.tensor(
                base_agg["i_idx"].to_numpy(dtype=np.int64) + num_users, dtype=torch.long
            ),
            train_ratings=base_train_y,
            config=train_cfg_base,
            run_name=f"{model_name}_baseline",
        )
        t_base = time.perf_counter() - t_base_start
        cpu_base_s = _proc_cpu_time_s() - cpu_base_start
        gpu_base_mb = gpu_max_memory_allocated_mb()

        t_infer_base_start = time.perf_counter()
        z_base = _infer(base_model, edge_index_base, device)
        t_infer_base = time.perf_counter() - t_infer_base_start
        ue_base = _l2_norm(z_base[:num_users])
        ie_base = _l2_norm(z_base[num_users:])
        rank_base = _eval_multi_k(ue_base, ie_base, test_pos, seen_train, args.eval_negatives, args.seed)
        reg_base  = _compute_rmse_mae(test_df, ue_base, ie_base, train_mean)

        _rank_str_base = "  ".join(
            f"NDCG@{k}={rank_base.get(f'NDCG@{k}', 0):.4f}" for k in _EVAL_KS
        )
        print(
            f"[1M] {model_name} BASELINE  "
            f"{_rank_str_base}  "
            f"RMSE={reg_base['RMSE']:.4f}  MAE={reg_base['MAE']:.4f}  "
            f"train={t_base:.1f}s  infer={t_infer_base:.3f}s  GPU={gpu_base_mb:.0f}MB"
        )
        row_base = {
            "model": model_name, "run_type": "baseline",
            **{k: v for k, v in rank_base.items() if k != "UsersEvaluated"},
            **reg_base,
            "training_time_s": round(t_base, 3),
            "inference_time_s": round(t_infer_base, 4),
            "gpu_peak_MB": round(gpu_base_mb, 1),
            "cpu_time_s": round(cpu_base_s, 3),
            "cpu_efficiency_pct": round(cpu_base_s / max(t_base, 1e-9) * 100, 1),
            "ram_rss_MB": round(rss_mb(), 1),
        }
        metrics_rows.append(row_base)

        # ── V. GSP reduced-graph training ─────────────────────────────────────
        print(f"\n[1M] {model_name} | GSP  ({nodes_gsp:,} nodes, {edges_gsp:,} edges)")
        try:
            gsp_model = get_model(model_name, nodes_gsp, model_cfg)
        except Exception as exc:
            print(f"[1M] WARNING: cannot build GSP '{model_name}': {exc}  Skipping GSP.")
            continue

        train_cfg_gsp = TrainConfig(
            epochs=args.epochs, lr=args.lr, weight_decay=1e-5,
            batch_size=args.batch_size, neg_ratio=args.neg_ratio,
            emb_l2_weight=1e-5, seed=args.seed, use_amp=(not args.no_amp),
            checkpoint_dir=os.path.join(out_dir, "checkpoints"),
            save_epoch_checkpoints=False,  # avoid disk-full on root partition
            metrics_jsonl_path=os.path.join(out_dir, f"training_metrics_{model_name}_gsp.jsonl"),
            training_log_path=os.path.join(out_dir,  f"training_log_{model_name}_gsp.txt"),
            device=device,
            early_stopping_patience=args.early_stopping_patience,
        )

        reset_gpu_peak_memory()
        t_gsp_train_start = time.perf_counter()
        cpu_gsp_start = _proc_cpu_time_s()
        _ = train_model(
            gsp_model,
            edge_index=edge_index_gsp,
            train_user_nodes=gsp_train_super,
            train_item_nodes=gsp_train_item,
            train_ratings=gsp_train_y,
            config=train_cfg_gsp,
            run_name=f"{model_name}_gsp",
        )
        t_gsp_train = time.perf_counter() - t_gsp_train_start
        cpu_gsp_s = _proc_cpu_time_s() - cpu_gsp_start
        gpu_gsp_mb = gpu_max_memory_allocated_mb()

        # ── VI. Projection ────────────────────────────────────────────────────
        t_infer_gsp_start = time.perf_counter()
        z_gsp = _infer(gsp_model, edge_index_gsp, device)
        t_infer_gsp_raw = time.perf_counter() - t_infer_gsp_start
        H_super = z_gsp[:num_super].astype(np.float32)
        H_final, proj_t = project_embeddings(H_super, C)
        t_infer_gsp = t_infer_gsp_raw + proj_t  # total GSP inference = forward + projection

        # ── VII. Evaluation ───────────────────────────────────────────────────
        ie_gsp = _l2_norm(z_gsp[num_super:])

        # 7a. Projected
        ue_proj = _l2_norm(H_final)
        rank_proj = _eval_multi_k(ue_proj, ie_gsp, test_pos, seen_train, args.eval_negatives, args.seed)
        reg_proj  = _compute_rmse_mae(test_df, ue_proj, ie_gsp, train_mean)

        _rank_str_gsp = "  ".join(
            f"NDCG@{k}={rank_proj.get(f'NDCG@{k}', 0):.4f}" for k in _EVAL_KS
        )
        print(
            f"[1M] {model_name} GSP+PROJ  "
            f"{_rank_str_gsp}  "
            f"RMSE={reg_proj['RMSE']:.4f}  MAE={reg_proj['MAE']:.4f}  "
            f"train={t_gsp_train:.1f}s  infer={t_infer_gsp:.3f}s  "
            f"(fwd={t_infer_gsp_raw:.3f}s + proj={proj_t:.3f}s)  GPU={gpu_gsp_mb:.0f}MB"
        )
        row_gsp = {
            "model": model_name, "run_type": "gsp_projected",
            **{k: v for k, v in rank_proj.items() if k != "UsersEvaluated"},
            **reg_proj,
            "training_time_s": round(t_gsp_train, 3),
            "inference_time_s": round(t_infer_gsp, 4),
            "inference_forward_s": round(t_infer_gsp_raw, 4),
            "projection_time_s": round(proj_t, 4),
            "gpu_peak_MB": round(gpu_gsp_mb, 1),
            "cpu_time_s": round(cpu_gsp_s, 3),
            "cpu_efficiency_pct": round(cpu_gsp_s / max(t_gsp_train, 1e-9) * 100, 1),
            "ram_rss_MB": round(rss_mb(), 1),
            "projection_time_s": round(proj_t, 4),
        }
        metrics_rows.append(row_gsp)

        # ── Speedup record ────────────────────────────────────────────────────
        speedup = t_base / max(t_gsp_train, 1e-9)
        infer_speedup = t_infer_base / max(t_infer_gsp, 1e-9)
        speedup_rows.append({
            "model":                      model_name,
            "training_time_baseline_s":   round(t_base, 3),
            "training_time_gsp_s":        round(t_gsp_train, 3),
            "speedup_factor":             round(speedup, 4),
            "inference_time_baseline_s":  round(t_infer_base, 4),
            "inference_time_gsp_s":       round(t_infer_gsp, 4),
            "inference_forward_gsp_s":    round(t_infer_gsp_raw, 4),
            "inference_projection_s":     round(proj_t, 4),
            "inference_speedup":          round(infer_speedup, 4),
            "gsp_preprocessing_s":        round(gsp_elapsed, 3),
            "net_time_saved_s":           round(t_base - t_gsp_train - gsp_elapsed, 3),
            "gpu_baseline_MB":            round(gpu_base_mb, 1),
            "gpu_gsp_MB":                 round(gpu_gsp_mb, 1),
            "gpu_reduction_pct":          round((1 - gpu_gsp_mb / max(gpu_base_mb, 1)) * 100, 2),
            "cpu_time_baseline_s":        round(cpu_base_s, 3),
            "cpu_time_gsp_s":             round(cpu_gsp_s, 3),
            "cpu_efficiency_baseline_pct": round(cpu_base_s / max(t_base, 1e-9) * 100, 1),
            "cpu_efficiency_gsp_pct":     round(cpu_gsp_s / max(t_gsp_train, 1e-9) * 100, 1),
            **{f"Precision@{k}_baseline": round(rank_base.get(f"Precision@{k}", 0), 4) for k in _EVAL_KS},
            **{f"Precision@{k}_gsp":      round(rank_proj.get(f"Precision@{k}", 0), 4) for k in _EVAL_KS},
            **{f"NDCG@{k}_baseline":      round(rank_base.get(f"NDCG@{k}", 0), 4) for k in _EVAL_KS},
            **{f"NDCG@{k}_gsp":           round(rank_proj.get(f"NDCG@{k}", 0), 4) for k in _EVAL_KS},
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Final output
    # ─────────────────────────────────────────────────────────────────────────
    _section("RESULTS SUMMARY")

    _print_paper_table(metrics_rows)

    # Write CSVs
    _save_results_csv(metrics_rows,  os.path.join(out_dir, "results_table.csv"))
    _save_results_csv(speedup_rows,  os.path.join(out_dir, "speedup_results.csv"))
    _save_gsp_stats_csv(gsp_paper_stats, dataset_summary,
                        os.path.join(out_dir, "gsp_stats.csv"))

    total_wall = time.perf_counter() - t_wall
    summary = {
        "dataset": dataset_summary,
        "gsp": gsp_paper_stats,
        "metrics": metrics_rows,
        "speedup": speedup_rows,
        "total_wall_time_s": round(total_wall, 2),
    }
    _write_json(os.path.join(out_dir, "full_results.json"), summary)

    print(f"\n[1M] Total wall-clock time : {total_wall/60:.1f} min")
    print(f"[1M] Results written to    : {out_dir}/")
    print(f"[1M]   results_table.csv")
    print(f"[1M]   speedup_results.csv")
    print(f"[1M]   gsp_stats.csv")
    print(f"[1M]   full_results.json")

    # Print compressed GSP stats for paper
    print("\n--- GSP Compression Summary (for paper) ---")
    for k, v in gsp_paper_stats.items():
        print(f"  {k:<45}: {v}")


if __name__ == "__main__":
    main()
