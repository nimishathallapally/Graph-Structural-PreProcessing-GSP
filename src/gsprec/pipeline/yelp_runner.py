"""
Yelp GSP/ICG Recommendation Pipeline Runner
============================================

Implements the full Graph Structural Pre-conditioning (GSP) /
Inductive Coarsening Graph (ICG) pipeline on the Yelp Academic Dataset.

Pipeline stages
---------------
  0. Dataset statistics
  1. Bipartite graph construction (CSR)
  2. Photo metadata features
  I.  Geometric Grouping  (COO, UU = A×Aᵀ, Forman-Ricci curvature)
  II. Adaptive Sparsification  (effective resistance + importance scoring)
  III. Embedding storage  (numpy memory-mapped arrays)
  IV. Baseline training  (LightGCN, float16, original graph)
  V.  Reduced-graph training  (LightGCN, float16, GSP-coarsened graph)
  VI. Projection step  (H_final = C × H_GNN)
  VII. Evaluation  (Precision/Recall/NDCG@10, RMSE, MAE)
  VIII. Output generation  (all publication-ready CSV/JSON files)

All results are reproducible: random seed, config, hardware, dataset version
are persisted alongside metric files.
"""
from __future__ import annotations

import argparse
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

# ── local imports ─────────────────────────────────────────────────────────────
# Add the src directory to path when running as a script
_SRC = str(Path(__file__).resolve().parent.parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from gsprec.config import ProjectConfig
from gsprec.data.yelp_dataset import (
    build_yelp_dataset,
    compute_bipartite_graph_stats,
)
from gsprec.data.semantic_features import (
    SemanticFeatureConfig,
    extract_semantic_features,
    load_semantic_features,
)
from gsprec.graph.gsp_ops import gsp_preprocess
from gsprec.graph.embedding_store import EmbeddingStore, project_embeddings
from gsprec.models.architectures import LightGCNRecommender, get_model, ModelConfig
from gsprec.models.trainer import TrainConfig, train_model
from gsprec.models.gnn import (
    RankingEvalConfig,
    evaluate_ranking_from_embeddings,
    rmse_mae,
)
from gsprec.utils.hardware_info import (
    HardwareMonitor,
    collect_hardware_info,
    rss_mb,
    gpu_memory_allocated_mb,
    gpu_max_memory_allocated_mb,
    reset_gpu_peak_memory,
)
from gsprec.utils.metrics_export import export_all_results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Any) -> None:
    _mkdir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _append_jsonl(path: str, record: Dict) -> None:
    _mkdir(os.path.dirname(os.path.abspath(path)) or ".")
    with open(path, "a", encoding="utf-8", buffering=1) as fh:
        fh.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _l2_norm(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


def _throughput(n_interactions: int, elapsed_s: float) -> float:
    return float(n_interactions) / max(elapsed_s, 1e-9)


def _reduction_pct(before: int, after: int) -> float:
    return float(before - after) / max(before, 1) * 100.0


def _memory_mb(arr: Optional[sp.spmatrix]) -> float:
    if arr is None:
        return 0.0
    if hasattr(arr, "data"):
        return float(arr.data.nbytes + arr.indices.nbytes + arr.indptr.nbytes) / (1024 ** 2)
    return 0.0


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def _split_ratings(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user ratio split (leave-ratio-out). Returns (train_df, test_df)."""
    rng = np.random.default_rng(seed)
    test_idxs: List[int] = []
    for _, grp in ratings_df.groupby("UserID"):
        idxs = grp.index.to_numpy()
        n_test = max(1, int(round(len(idxs) * test_ratio)))
        n_test = min(n_test, len(idxs) - 1)
        chosen = rng.choice(idxs, size=n_test, replace=False)
        test_idxs.extend(chosen.tolist())
    mask = ratings_df.index.isin(test_idxs)
    train_df = ratings_df.loc[~mask].reset_index(drop=True)
    test_df = ratings_df.loc[mask].reset_index(drop=True)
    return train_df, test_df


def _build_seen_sets(df: pd.DataFrame) -> Dict[int, Set[int]]:
    seen: Dict[int, Set[int]] = {}
    for row in df.itertuples(index=False):
        seen.setdefault(int(row.UserID), set()).add(int(row.BusinessID))
    return seen


def _build_test_positives(
    test_df: pd.DataFrame, threshold: float
) -> Dict[int, List[int]]:
    pos: Dict[int, List[int]] = {}
    for row in test_df.itertuples(index=False):
        if float(row.Rating) >= threshold:
            pos.setdefault(int(row.UserID), []).append(int(row.BusinessID))
    return pos


# ---------------------------------------------------------------------------
# Edge-index builder from interactions DataFrame
# ---------------------------------------------------------------------------

def _build_edge_index(
    interactions: pd.DataFrame,
    user_col: str,
    item_col: str,
    item_offset: int,
) -> torch.Tensor:
    """Build undirected bipartite edge_index (PyG format)."""
    su = interactions[user_col].to_numpy(dtype=np.int64)
    it = interactions[item_col].to_numpy(dtype=np.int64) + item_offset
    src = np.concatenate([su, it])
    dst = np.concatenate([it, su])
    return torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)


# ---------------------------------------------------------------------------
# Graph density
# ---------------------------------------------------------------------------

def _graph_density(n_nodes: int, n_edges: int) -> float:
    if n_nodes <= 1:
        return 0.0
    return float(2 * n_edges) / float(n_nodes * (n_nodes - 1))


# ---------------------------------------------------------------------------
# Inference helper (returns numpy embeddings)
# ---------------------------------------------------------------------------

def _infer_embeddings(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    device: str,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = model(edge_index.to(device)).detach().cpu().float().numpy()
    return z


# ---------------------------------------------------------------------------
# RMSE / MAE from dot-product scores
# ---------------------------------------------------------------------------

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
    bias = train_mean - float(y_p.mean())
    y_p = np.clip(y_p + bias, 1.0, 5.0)
    rmse, mae = rmse_mae(y_t, y_p)
    return {"RMSE": rmse, "MAE": mae}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_yelp_pipeline(config: ProjectConfig) -> None:  # noqa: C901
    """
    Execute the full GSP/ICG pipeline end-to-end on the Yelp dataset.

    Config fields used beyond the base ProjectConfig:
        config.data.dataset_path     path to Yelp dataset directory
        config.data.implicit_threshold
        config.data.test_seed
        config.gsp.*                 GSP hyper-parameters
        config.train.*               LightGCN training hyper-parameters
        config.eval.*                Evaluation settings
        config.output_dir
    """
    t_pipeline_start = time.perf_counter()
    out_dir = config.output_dir
    _mkdir(out_dir)
    _mkdir(os.path.join(out_dir, "checkpoints"))
    _mkdir(os.path.join(out_dir, "embeddings"))
    _mkdir(os.path.join(out_dir, "cache"))

    seed = config.train.seed
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Yelp] Device: {device}")

    # ── Hardware snapshot ───────────────────────────────────────────────────
    hw_info = collect_hardware_info()
    hw_info["random_seed"] = seed
    hw_info["device"] = device
    _write_json(os.path.join(out_dir, "hardware_info.json"), hw_info)

    # Accumulators for output files
    memory_records: List[Dict] = []
    preprocessing_records: List[Dict] = []
    reduction_records: List[Dict] = []
    speedup_records: List[Dict] = []
    metrics_records: List[Dict] = []
    training_log_records: List[Dict] = []

    # ── 0. Dataset loading ──────────────────────────────────────────────────
    print("\n[Yelp] ══ STAGE 0: Dataset Loading ══")
    data_dir = config.data.dataset_path or "./data"

    with HardwareMonitor("dataset_loading") as mon_data:
        yelp = build_yelp_dataset(
            data_dir=data_dir,
            min_user_interactions=max(config.data.min_interactions, 5),
            min_business_interactions=max(config.data.min_interactions, 5),
            implicit_threshold=config.data.implicit_threshold,
        )

    ratings_df: pd.DataFrame = yelp["ratings_df"]
    num_users: int = yelp["num_users"]
    num_items: int = yelp["num_items"]
    dataset_stats: Dict = yelp["stats"]
    photo_features: Optional[pd.DataFrame] = yelp["photo_features"]

    mem_rec_data = {**mon_data.summary(), "stage": "dataset_loading"}
    memory_records.append(mem_rec_data)
    preprocessing_records.append(
        {"stage": "dataset_loading", "elapsed_s": mon_data.elapsed_s,
         "description": "JSONL streaming + k-core filter + dedup"}
    )

    # ── Debug mode: subset to the most-active users ──────────────────────────
    if config.data.debug_mode:
        max_u = config.data.max_debug_users
        print(f"[Yelp] Debug mode: keeping top-{max_u} most-active users")
        counts = ratings_df.groupby("UserID")["BusinessID"].count()
        top_users = counts.nlargest(max_u).index.to_numpy()
        ratings_df = ratings_df[ratings_df["UserID"].isin(top_users)].copy()
        # Remap to contiguous 0-based IDs
        uniq_u = np.sort(ratings_df["UserID"].unique())
        uniq_i = np.sort(ratings_df["BusinessID"].unique())
        ratings_df["UserID"] = np.searchsorted(uniq_u, ratings_df["UserID"].to_numpy())
        ratings_df["BusinessID"] = np.searchsorted(uniq_i, ratings_df["BusinessID"].to_numpy())
        ratings_df = ratings_df.reset_index(drop=True)
        num_users = int(len(uniq_u))
        num_items = int(len(uniq_i))
        dataset_stats["num_users"] = num_users
        dataset_stats["num_items"] = num_items
        dataset_stats["num_interactions"] = len(ratings_df)
        print(f"[Yelp] Debug subset: {num_users:,} users | {num_items:,} items | {len(ratings_df):,} interactions")

    # ── Train/test split ────────────────────────────────────────────────────
    train_df, test_df = _split_ratings(
        ratings_df,
        test_ratio=0.2,
        seed=config.data.test_seed,
    )
    dataset_stats["train_interactions"] = len(train_df)
    dataset_stats["test_interactions"] = len(test_df)
    dataset_stats["train_ratio"] = round(len(train_df) / max(len(ratings_df), 1), 4)
    dataset_stats["test_ratio"] = round(len(test_df) / max(len(ratings_df), 1), 4)

    seen_train: Dict[int, Set[int]] = _build_seen_sets(train_df)
    test_pos: Dict[int, List[int]] = _build_test_positives(
        test_df, config.data.implicit_threshold
    )
    train_mean_rating = float(train_df["Rating"].mean())

    print(
        f"[Yelp] Train={len(train_df):,}  Test={len(test_df):,}  "
        f"Test users with positives={len(test_pos):,}"
    )

    # ── 1. Bipartite graph construction (CSR) ───────────────────────────────
    print("\n[Yelp] ══ STAGE 1: Bipartite Graph Construction ══")

    with HardwareMonitor("graph_construction") as mon_graph:
        graph_stats = compute_bipartite_graph_stats(train_df, num_users, num_items)

    print(
        f"[Yelp] Graph: {graph_stats['num_nodes']:,} nodes | "
        f"{graph_stats['num_edges']:,} edges | "
        f"avg_degree={graph_stats['avg_degree']:.2f} | "
        f"density={graph_stats['density']:.2e} | "
        f"components={graph_stats['num_components']:,} | "
        f"mem={graph_stats['graph_memory_MB']:.1f}MB"
    )
    dataset_stats.update({k: v for k, v in graph_stats.items()})

    memory_records.append({**mon_graph.summary(), "stage": "graph_construction"})
    preprocessing_records.append(
        {"stage": "graph_construction", "elapsed_s": mon_graph.elapsed_s,
         "description": "CSR bipartite graph A + statistics"}
    )

    # Save dataset stats
    _write_json(os.path.join(out_dir, "dataset_stats.json"), dataset_stats)

    # ── 2. Photo metadata features ──────────────────────────────────────────
    print("\n[Yelp] ══ STAGE 2: Photo Metadata Features ══")
    if photo_features is not None:
        photo_path = os.path.join(out_dir, "photo_features.csv")
        photo_features.to_csv(photo_path, index=False)
        print(
            f"[Yelp] Photo features saved: {len(photo_features):,} businesses | "
            f"{photo_path}"
        )
        photo_stats = {
            col: int(photo_features[col].sum())
            for col in ["num_food_photos", "num_drink_photos",
                        "num_inside_photos", "num_outside_photos",
                        "num_menu_photos", "total_photos"]
            if col in photo_features.columns
        }
        _write_json(os.path.join(out_dir, "photo_stats.json"), photo_stats)
        print(f"[Yelp] Photo stats: {photo_stats}")
    else:
        print("[Yelp] No photo features available.")

    # ── 2b. Semantic feature extraction (VADER, TF-IDF, photo counts) ────────
    print("\n[Yelp] ══ STAGE 2b: Semantic Feature Extraction ══")
    sem_feat_matrix: Optional[np.ndarray] = None
    sem_feat_names: List[str] = []

    unique_items: np.ndarray = yelp.get("unique_items", np.array([], dtype=object))
    _sem_features_dir = os.path.join(out_dir, "features")
    _sem_saved_path = os.path.join(_sem_features_dir, "semantic_features.npy")
    if len(unique_items) > 0 and not config.data.debug_mode:
        # Try loading from previously saved file first
        if os.path.exists(_sem_saved_path):
            print(f"[Yelp] Loading semantic features from saved file: {_sem_saved_path}")
            sem_feat_matrix, sem_feat_names = load_semantic_features(_sem_features_dir)
            if sem_feat_matrix is not None:
                print(
                    f"[Yelp] Semantic features loaded: shape={sem_feat_matrix.shape} | "
                    f"dim={sem_feat_matrix.shape[1]} | {len(sem_feat_names)} columns"
                )
            else:
                print("[Yelp] WARNING: Saved semantic feature file found but failed to load.")
        else:
            # In debug mode skip TF-IDF (re-index changes item IDs)
            sem_cfg = SemanticFeatureConfig(
                tfidf_dim=getattr(config.train, "tfidf_dim", 64),
                max_review_chars=400,
                max_reviews_per_item=50,
                include_photo_features=True,
            )
            with HardwareMonitor("semantic_features") as mon_sem:
                try:
                    sem_feat_matrix, sem_feat_names = extract_semantic_features(
                        data_dir=data_dir,
                        unique_items=unique_items,
                        output_dir=_sem_features_dir,
                        config=sem_cfg,
                    )
                    print(
                        f"[Yelp] Semantic features: shape={sem_feat_matrix.shape} | "
                        f"dim={sem_feat_matrix.shape[1]} | {len(sem_feat_names)} columns"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[Yelp] WARNING: Semantic feature extraction failed: {exc}")
                    sem_feat_matrix = None
            memory_records.append({**mon_sem.summary(), "stage": "semantic_features"})
            preprocessing_records.append({
                "stage": "semantic_features",
                "elapsed_s": mon_sem.elapsed_s,
                "description": "VADER sentiment + TF-IDF/SVD + photo counts → float16 mmap",
            })
    else:
        print("[Yelp] Semantic feature extraction skipped (debug mode or no item IDs).")

    # ── I. GSP Preprocessing (Stage I + II) ─────────────────────────────────
    print("\n[Yelp] ══ STAGE I–II: GSP Preprocessing ══")

    cpu_before_gsp = rss_mb()
    with HardwareMonitor("gsp_preprocessing") as mon_gsp:
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
            er_solver=getattr(config.gsp, "er_solver", "arpack"),
            er_sketches=getattr(config.gsp, "er_sketches", 32),
            er_node_limit=0 if getattr(config.gsp, "er_solver", "arpack") == "jl" else 50_000,
            cache_dir=os.path.join(out_dir, "cache"),
            output_dir=out_dir,
            data_load_time_s=yelp["load_time_s"],
        )

    cpu_after_gsp = rss_mb()
    gsp_timing: Dict = gsp_out["timing"]
    gsp_stats: Dict = gsp_out["stats"]

    user_to_super: np.ndarray = gsp_out["user_to_super"]
    num_super: int = gsp_out["num_super"]
    C: sp.csr_matrix = gsp_out["C"]                  # (num_super × num_users)

    # Log per-stage preprocessing times
    for stage_name, desc in [
        ("curvature_time_s", "Stage I: Forman-Ricci curvature (UU = A×Aᵀ, COO)"),
        ("coarsening_time_s", "Stage I: User clustering (connected components)"),
        ("er_time_s", "Stage II: Effective resistance (shift-invert eigsh)"),
        ("sparsification_time_s", "Stage II: Adaptive edge sparsification"),
    ]:
        preprocessing_records.append({
            "stage": stage_name.replace("_time_s", ""),
            "elapsed_s": gsp_timing.get(stage_name, 0.0),
            "description": desc,
        })
    total_preproc = gsp_timing.get("total_preprocessing_time_s", mon_gsp.elapsed_s)
    preprocessing_records.append({
        "stage": "total_gsp_preprocessing",
        "elapsed_s": total_preproc,
        "description": "Total GSP preprocessing (curvature + ER + sparsification)",
    })
    memory_records.append({**mon_gsp.summary(), "stage": "gsp_preprocessing"})

    # ── Build coarsened bipartite graph ──────────────────────────────────────
    print("\n[Yelp] Building coarsened bipartite graph ...")
    t0 = time.perf_counter()

    train_copy = train_df.copy()
    train_copy["super_idx"] = user_to_super[train_copy["UserID"].to_numpy(dtype=np.int64)]
    coarsened = (
        train_copy.groupby(["super_idx", "BusinessID"], as_index=False)
        .agg(rating=("Rating", "mean"), count=("Rating", "size"))
        .rename(columns={"BusinessID": "item_idx"})
        .astype({"super_idx": np.int64, "item_idx": np.int64})
    )

    edge_index_gsp = _build_edge_index(coarsened, "super_idx", "item_idx", num_super)
    t_coarsen_graph = time.perf_counter() - t0

    # Original (un-coarsened) edge index for baseline
    base_agg = (
        train_df.groupby(["UserID", "BusinessID"], as_index=False)
        .agg(rating=("Rating", "mean"), count=("Rating", "size"))
        .rename(columns={"UserID": "super_idx", "BusinessID": "item_idx"})
        .astype({"super_idx": np.int64, "item_idx": np.int64})
    )
    edge_index_base = _build_edge_index(base_agg, "super_idx", "item_idx", num_users)

    # ── Graph reduction statistics ───────────────────────────────────────────
    edges_orig  = int(base_agg.shape[0])
    edges_gsp   = int(coarsened.shape[0])
    nodes_orig  = num_users + num_items
    nodes_gsp   = num_super + num_items

    mem_orig_mb = float(
        edge_index_base.numpy().nbytes +
        base_agg["rating"].to_numpy(dtype=np.float32).nbytes
    ) / (1024 ** 2)
    mem_gsp_mb = float(
        edge_index_gsp.numpy().nbytes +
        coarsened["rating"].to_numpy(dtype=np.float32).nbytes
    ) / (1024 ** 2)

    reduction_records.append({
        "stage":              "graph_coarsening",
        "nodes_before":       nodes_orig,
        "nodes_after":        nodes_gsp,
        "edges_before":       edges_orig,
        "edges_after":        edges_gsp,
        "edges_removed":      edges_orig - edges_gsp,
        "reduction_percent":  _reduction_pct(edges_orig, edges_gsp),
        "avg_degree_before":  round(float(2 * edges_orig / max(nodes_orig, 1)), 4),
        "avg_degree_after":   round(float(2 * edges_gsp  / max(nodes_gsp, 1)),  4),
        "density_before":     _graph_density(nodes_orig, edges_orig),
        "density_after":      _graph_density(nodes_gsp, edges_gsp),
        "memory_before_MB":   round(mem_orig_mb, 3),
        "memory_after_MB":    round(mem_gsp_mb, 3),
    })

    # ── User-user edge reduction (Stage II) ─────────────────────────────────
    uu_before = gsp_stats.get("uu_edges_all", 0)
    uu_hc     = gsp_stats.get("uu_edges_hc", 0)
    uu_pruned = gsp_stats.get("uu_edges_pruned", 0)
    reduction_records.append({
        "stage":              "uu_sparsification",
        "nodes_before":       num_users,
        "nodes_after":        num_super,
        "edges_before":       uu_before,
        "edges_after":        uu_pruned,
        "edges_removed":      uu_before - uu_pruned,
        "reduction_percent":  _reduction_pct(uu_before, uu_pruned),
        "avg_degree_before":  round(float(2 * uu_before / max(num_users, 1)), 4),
        "avg_degree_after":   round(float(2 * uu_pruned / max(num_users, 1)), 4),
        "density_before":     _graph_density(num_users, uu_before),
        "density_after":      _graph_density(num_users, uu_pruned),
        "memory_before_MB":   0.0,
        "memory_after_MB":    0.0,
    })

    print(
        f"[Yelp] Graph reduction: {nodes_orig:,}→{nodes_gsp:,} nodes | "
        f"{edges_orig:,}→{edges_gsp:,} edges | "
        f"reduction={_reduction_pct(edges_orig, edges_gsp):.1f}%"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # IV–VII: Per-model training, projection, evaluation
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[Yelp] ══ STAGE IV: Baseline Training (Original Graph) ══")
    print(f"  Nodes={nodes_orig:,}  Edges={edge_index_base.shape[1]:,}")

    base_train_y = torch.tensor(base_agg["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)

    gsp_train_super = torch.tensor(coarsened["super_idx"].to_numpy(dtype=np.int64), dtype=torch.long)
    gsp_train_item  = torch.tensor(
        coarsened["item_idx"].to_numpy(dtype=np.int64) + num_super, dtype=torch.long
    )
    gsp_train_y = torch.tensor(coarsened["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)

    # ── Build ModelConfig shared across all models ────────────────────────────
    model_cfg = ModelConfig(
        emb_dim=config.train.emb_dim,
        hidden_dim=config.train.hidden_dim,
        out_dim=config.train.out_dim,
        num_layers=config.train.num_layers,
        heads=config.train.heads,
        dropout=config.train.dropout,
    )

    models_to_run: List[str] = list(config.models) if config.models else ["lightgcn"]
    print(f"\n[Yelp] Models to run: {models_to_run}")

    # Track best baseline elapsed time across models for speedup denominator
    _first_base_t: Optional[float] = None
    _first_gsp_t:  Optional[float] = None

    # ═══════════════════════════════════════════════════════════════════════
    # IV + V + VI + VII: Per-model training, projection, evaluation
    # ═══════════════════════════════════════════════════════════════════════

    for model_name in models_to_run:
        print(f"\n[Yelp] ══ MODEL: {model_name.upper()} ══")

        # ── IV. BASELINE training on original graph ──────────────────────────
        print(f"[Yelp] ── STAGE IV: Baseline Training ({model_name}) ──")
        print(f"  Nodes={nodes_orig:,}  Edges={edge_index_base.shape[1]:,}")

        try:
            base_model = get_model(model_name, nodes_orig, model_cfg)
        except Exception as exc:
            print(f"[Yelp] WARNING: Could not build model '{model_name}': {exc}. Skipping.")
            continue

        # Out-dim may differ between models (LightGCN uses emb_dim; SAGEConv uses out_dim)
        base_emb_dim = base_model.out_dim if hasattr(base_model, "out_dim") else config.train.out_dim

        # Warm-start item embeddings with semantic content features so the GNN
        # begins from a content-aware initialisation rather than pure random noise.
        if sem_feat_matrix is not None and sem_feat_matrix.shape[0] == num_items:
            _init_item_embeddings_from_features(
                base_model, sem_feat_matrix, num_users, base_emb_dim, alpha=0.30
            )
            print(f"[Yelp] {model_name} baseline: item embeddings warm-started from semantic features")

        train_cfg = TrainConfig(
            epochs=config.train.epochs,
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
            batch_size=config.train.batch_size,
            neg_ratio=config.train.neg_ratio,
            emb_l2_weight=config.train.emb_l2_weight,
            seed=seed,
            use_amp=config.train.use_amp,
            checkpoint_dir=os.path.join(out_dir, "checkpoints"),
            save_epoch_checkpoints=False,
            metrics_jsonl_path=os.path.join(out_dir, f"training_metrics_{model_name}.jsonl"),
            training_log_path=os.path.join(out_dir, f"training_log_{model_name}.txt"),
            device=device,
        )

        rank_cfg = RankingEvalConfig(
            k=config.eval.k,
            num_negatives=config.eval.num_negatives,
            seed=config.eval.seed,
        )

        def _make_base_eval_cb(ei: torch.Tensor, n_users: int, mn: str = model_name):
            def _cb(m):
                m.eval()
                with torch.no_grad():
                    z = m(ei.to(device)).detach().cpu().float().numpy()
                ue = _l2_norm(z[:n_users])
                ie = _l2_norm(z[n_users:])
                res = evaluate_ranking_from_embeddings(ue, ie, test_pos, seen_train, rank_cfg)
                return {f"base_{k}": v for k, v in res.items() if k != "UsersEvaluated"}
            return _cb

        reset_gpu_peak_memory()
        cpu_base_start = rss_mb()
        with HardwareMonitor(f"baseline_training_{model_name}") as mon_base:
            base_summary = train_model(
                base_model,
                edge_index=edge_index_base,
                train_user_nodes=torch.tensor(
                    base_agg["super_idx"].to_numpy(dtype=np.int64), dtype=torch.long
                ),
                train_item_nodes=torch.tensor(
                    base_agg["item_idx"].to_numpy(dtype=np.int64) + num_users, dtype=torch.long
                ),
                train_ratings=base_train_y,
                config=train_cfg,
                run_name=f"{model_name}_baseline",
                eval_callback=_make_base_eval_cb(edge_index_base, num_users),
            )
        cpu_base_end = rss_mb()
        gpu_base_peak = gpu_max_memory_allocated_mb()

        memory_records.append({**mon_base.summary(), "stage": f"baseline_training_{model_name}"})
        training_log_records.append({"run": f"{model_name}_baseline", **base_summary})

        z_base = _infer_embeddings(base_model, edge_index_base, device)
        user_emb_base = _l2_norm(z_base[:num_users])
        item_emb_base = _l2_norm(z_base[num_users:])

        # Node feature augmentation with semantic features (item side only)
        if sem_feat_matrix is not None and sem_feat_matrix.shape[0] == num_items:
            item_emb_base = _augment_with_features(item_emb_base, sem_feat_matrix)

        rank_base = evaluate_ranking_from_embeddings(
            user_emb_base, item_emb_base, test_pos, seen_train, rank_cfg
        )
        reg_base = _compute_rmse_mae(test_df, user_emb_base, item_emb_base, train_mean_rating)
        print(
            f"[Yelp] {model_name} Baseline  "
            f"P@10={rank_base.get('Precision@10', 0):.4f}  "
            f"R@10={rank_base.get('Recall@10', 0):.4f}  "
            f"NDCG@10={rank_base.get('NDCG@10', 0):.4f}  "
            f"RMSE={reg_base.get('RMSE', 0):.4f}  MAE={reg_base.get('MAE', 0):.4f}"
        )
        metrics_records.append({
            "model": model_name, "run_type": "baseline",
            **{k: v for k, v in rank_base.items() if k != "UsersEvaluated"},
            **reg_base,
            "training_time_s": mon_base.elapsed_s,
            "gpu_peak_MB": gpu_base_peak,
            "cpu_rss_MB": cpu_base_end - cpu_base_start,
            "semantic_features_augmented": sem_feat_matrix is not None,
        })

        # Save baseline embeddings (mmap)
        store_path_base = os.path.join(out_dir, "embeddings", f"{model_name}_baseline_full.npy")
        with EmbeddingStore(store_path_base, nodes_orig, z_base.shape[1], dtype=np.float16) as es:
            es.store(z_base.astype(np.float16))
        print(f"[Yelp] {model_name} baseline embeddings → {store_path_base}")

        if _first_base_t is None:
            _first_base_t = mon_base.elapsed_s

        # ── V. GSP reduced-graph training ────────────────────────────────────
        print(f"\n[Yelp] ── STAGE V: GSP Reduced-Graph Training ({model_name}) ──")
        print(f"  Nodes={nodes_gsp:,}  Edges={edge_index_gsp.shape[1]:,}")

        try:
            gsp_model = get_model(model_name, nodes_gsp, model_cfg)
        except Exception as exc:
            print(f"[Yelp] WARNING: Could not build GSP model '{model_name}': {exc}. Skipping GSP run.")
            continue

        # Warm-start item embeddings for GSP model (items start at num_super)
        if sem_feat_matrix is not None and sem_feat_matrix.shape[0] == num_items:
            gsp_emb_dim = gsp_model.out_dim if hasattr(gsp_model, "out_dim") else config.train.out_dim
            _init_item_embeddings_from_features(
                gsp_model, sem_feat_matrix, num_super, gsp_emb_dim, alpha=0.30
            )
            print(f"[Yelp] {model_name} GSP: item embeddings warm-started from semantic features")

        def _make_gsp_eval_cb(ei: torch.Tensor, n_super: int, u2s: np.ndarray, mn: str = model_name):
            def _cb(m):
                m.eval()
                with torch.no_grad():
                    z = m(ei.to(device)).detach().cpu().float().numpy()
                super_emb = _l2_norm(z[:n_super])
                ie = _l2_norm(z[n_super:])
                ue = super_emb[u2s]
                res = evaluate_ranking_from_embeddings(ue, ie, test_pos, seen_train, rank_cfg)
                return {f"gsp_{k}": v for k, v in res.items() if k != "UsersEvaluated"}
            return _cb

        reset_gpu_peak_memory()
        cpu_gsp_start = rss_mb()
        with HardwareMonitor(f"gsp_training_{model_name}") as mon_gsp_train:
            gsp_summary = train_model(
                gsp_model,
                edge_index=edge_index_gsp,
                train_user_nodes=gsp_train_super,
                train_item_nodes=gsp_train_item,
                train_ratings=gsp_train_y,
                config=train_cfg,
                run_name=f"{model_name}_gsp",
                eval_callback=_make_gsp_eval_cb(edge_index_gsp, num_super, user_to_super),
            )
        cpu_gsp_end = rss_mb()
        gpu_gsp_peak = gpu_max_memory_allocated_mb()

        memory_records.append({**mon_gsp_train.summary(), "stage": f"gsp_training_{model_name}"})
        training_log_records.append({"run": f"{model_name}_gsp", **gsp_summary})

        z_gsp = _infer_embeddings(gsp_model, edge_index_gsp, device)
        store_path_gsp = os.path.join(out_dir, "embeddings", f"{model_name}_gsp_super.npy")
        with EmbeddingStore(store_path_gsp, nodes_gsp, z_gsp.shape[1], dtype=np.float16) as es:
            es.store(z_gsp.astype(np.float16))
        print(f"[Yelp] {model_name} GSP embeddings → {store_path_gsp}")

        if _first_gsp_t is None:
            _first_gsp_t = mon_gsp_train.elapsed_s

        # ── VI. Projection H_final = Cᵀ × H_GNN ─────────────────────────────
        print(f"\n[Yelp] ── STAGE VI: Projection ({model_name}) ──")

        with HardwareMonitor(f"projection_{model_name}") as mon_proj:
            H_super = z_gsp[:num_super].astype(np.float32)
            H_final, proj_time_s = project_embeddings(H_super, C)

        preprocessing_records.append({
            "stage": f"projection_{model_name}",
            "elapsed_s": proj_time_s,
            "description": f"H_final = Cᵀ × H_GNN ({model_name})",
        })
        memory_records.append({**mon_proj.summary(), "stage": f"projection_{model_name}"})

        proj_path = os.path.join(out_dir, "embeddings", f"{model_name}_projected_user.npy")
        with EmbeddingStore(proj_path, num_users, H_final.shape[1], dtype=np.float16) as es:
            es.store(H_final.astype(np.float16))
        print(
            f"[Yelp] {model_name} projected embeddings → {proj_path} | "
            f"shape={H_final.shape} | {proj_time_s:.3f}s"
        )

        # ── VII. Evaluation ───────────────────────────────────────────────────
        item_emb_gsp = _l2_norm(z_gsp[num_super:])

        # Augment item embeddings with semantic features
        if sem_feat_matrix is not None and sem_feat_matrix.shape[0] == num_items:
            item_emb_gsp = _augment_with_features(item_emb_gsp, sem_feat_matrix)

        # 7a. Projected embeddings
        user_emb_proj = _l2_norm(H_final)
        rank_proj = evaluate_ranking_from_embeddings(
            user_emb_proj, item_emb_gsp, test_pos, seen_train, rank_cfg
        )
        reg_proj = _compute_rmse_mae(test_df, user_emb_proj, item_emb_gsp, train_mean_rating)
        print(
            f"[Yelp] {model_name} GSP+Proj  "
            f"P@10={rank_proj.get('Precision@10', 0):.4f}  "
            f"R@10={rank_proj.get('Recall@10', 0):.4f}  "
            f"NDCG@10={rank_proj.get('NDCG@10', 0):.4f}  "
            f"RMSE={reg_proj.get('RMSE', 0):.4f}  MAE={reg_proj.get('MAE', 0):.4f}"
        )
        metrics_records.append({
            "model": model_name, "run_type": "gsp_projected",
            **{k: v for k, v in rank_proj.items() if k != "UsersEvaluated"},
            **reg_proj,
            "training_time_s": mon_gsp_train.elapsed_s,
            "gpu_peak_MB": gpu_gsp_peak,
            "cpu_rss_MB": cpu_gsp_end - cpu_gsp_start,
            "semantic_features_augmented": sem_feat_matrix is not None,
        })

        # 7b. Direct super-node lookup (no projection)
        user_emb_direct = _l2_norm(z_gsp[:num_super][user_to_super])
        rank_direct = evaluate_ranking_from_embeddings(
            user_emb_direct, item_emb_gsp, test_pos, seen_train, rank_cfg
        )
        reg_direct = _compute_rmse_mae(test_df, user_emb_direct, item_emb_gsp, train_mean_rating)
        metrics_records.append({
            "model": model_name, "run_type": "gsp_direct",
            **{k: v for k, v in rank_direct.items() if k != "UsersEvaluated"},
            **reg_direct,
            "training_time_s": mon_gsp_train.elapsed_s,
            "gpu_peak_MB": gpu_gsp_peak,
            "cpu_rss_MB": cpu_gsp_end - cpu_gsp_start,
            "semantic_features_augmented": sem_feat_matrix is not None,
        })

        # ── Speedup for this model ────────────────────────────────────────────
        t_base_m = mon_base.elapsed_s
        t_gsp_m  = mon_gsp_train.elapsed_s
        speedup_factor = float(t_base_m / max(t_gsp_m, 1e-9))
        gpu_mem_red = (
            float(gpu_base_peak - gpu_gsp_peak) / max(gpu_base_peak, 1e-3) * 100.0
            if gpu_base_peak > 0 else 0.0
        )
        cpu_mem_red = (
            float((cpu_base_end - cpu_base_start) - (cpu_gsp_end - cpu_gsp_start))
            / max(abs(cpu_base_end - cpu_base_start), 1e-3) * 100.0
        )
        print(
            f"[Yelp] {model_name} speedup={speedup_factor:.2f}× | "
            f"GPU mem: {gpu_base_peak:.0f}→{gpu_gsp_peak:.0f}MB ({gpu_mem_red:.1f}%↓)"
        )
        speedup_records.append({
            "model":                       model_name,
            "training_time_original_s":    round(t_base_m, 4),
            "training_time_reduced_s":     round(t_gsp_m, 4),
            "speedup_factor":              round(speedup_factor, 4),
            "GPU_memory_original_MB":      round(gpu_base_peak, 2),
            "GPU_memory_reduced_MB":       round(gpu_gsp_peak, 2),
            "GPU_memory_reduction_pct":    round(gpu_mem_red, 2),
            "CPU_memory_original_MB":      round(cpu_base_end - cpu_base_start, 2),
            "CPU_memory_reduced_MB":       round(cpu_gsp_end - cpu_gsp_start, 2),
            "CPU_memory_reduction_pct":    round(cpu_mem_red, 2),
            "training_throughput_original": round(
                _throughput(len(base_agg) * config.train.epochs, t_base_m), 1
            ),
            "training_throughput_reduced":  round(
                _throughput(len(coarsened) * config.train.epochs, t_gsp_m), 1
            ),
            "GPU_utilization_original_pct": round(mon_base.gpu_util_end, 1),
            "GPU_utilization_reduced_pct":  round(mon_gsp_train.gpu_util_end, 1),
            "preprocessing_overhead_s": round(
                gsp_timing.get("total_preprocessing_time_s", 0.0), 4
            ),
            "training_time_saved_s": round(t_base_m - t_gsp_m, 4),
            "net_gain_s": round(
                t_base_m - t_gsp_m - gsp_timing.get("total_preprocessing_time_s", 0.0), 4
            ),
        })

    # ── Overall speedup summary ───────────────────────────────────────────────
    if speedup_records:
        sr0 = speedup_records[0]
        print(
            f"\n[Yelp] ══ SPEEDUP SUMMARY ({sr0['model']}) ══\n"
            f"  Training speedup     : {sr0['speedup_factor']:.2f}×\n"
            f"  GPU mem reduction    : {sr0['GPU_memory_reduction_pct']:.1f}%\n"
            f"  CPU mem reduction    : {sr0['CPU_memory_reduction_pct']:.1f}%\n"
            f"  Edge reduction       : {_reduction_pct(edges_orig, edges_gsp):.1f}%\n"
            f"  User compression     : {gsp_stats.get('compression_ratio', 0)*100:.1f}%"
        )

    # ── Total pipeline time ──────────────────────────────────────────────────
    t_total = time.perf_counter() - t_pipeline_start
    preprocessing_records.append({
        "stage": "total_pipeline",
        "elapsed_s": round(t_total, 3),
        "description": "End-to-end wall-clock time",
    })

    # ── Reproducibility record ──────────────────────────────────────────────
    repro_info = {
        "random_seed":        seed,
        "dataset_name":       "yelp_academic_dataset",
        "dataset_path":       data_dir,
        "dataset_version":    "2021",
        "num_users":          num_users,
        "num_items":          num_items,
        "num_interactions":   dataset_stats["num_interactions"],
        "implicit_threshold": config.data.implicit_threshold,
        "test_ratio":         0.2,
        "gsp_alpha":          config.gsp.alpha,
        "gsp_curvature_percentile": config.gsp.curvature_percentile,
        "gsp_importance_percentile": config.gsp.importance_percentile,
        "gsp_er_num_eigenvectors": config.gsp.er_num_eigenvectors,
        "models":             list(models_to_run),
        "emb_dim":            config.train.emb_dim,
        "num_layers":         config.train.num_layers,
        "epochs":             config.train.epochs,
        "lr":                 config.train.lr,
        "device":             device,
        "semantic_features":  sem_feat_names,
        "semantic_feature_dim": int(sem_feat_matrix.shape[1]) if sem_feat_matrix is not None else 0,
        "hardware":           hw_info,
    }

    # ═══════════════════════════════════════════════════════════════════════
    # VIII. OUTPUT FILES
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[Yelp] ══ STAGE VIII: Writing Output Files ══")
    export_all_results(
        output_dir=out_dir,
        metrics_records=metrics_records,
        memory_records=memory_records,
        speedup_records=speedup_records,
        reduction_records=reduction_records,
        preprocessing_records=preprocessing_records,
        training_log_records=training_log_records,
        hardware_info=hw_info,
        dataset_stats=dataset_stats,
        reproducibility_info=repro_info,
    )

    _write_json(os.path.join(out_dir, "gsp_stats.json"), {**gsp_stats, **gsp_timing})

    print(
        f"\n[Yelp] Pipeline complete in {t_total:.1f}s\n"
        f"       Results in: {os.path.abspath(out_dir)}"
    )


# ---------------------------------------------------------------------------
# Feature augmentation helper
# ---------------------------------------------------------------------------

def _augment_with_features(
    emb: np.ndarray,
    feat: np.ndarray,
) -> np.ndarray:
    """
    Blend GNN item embeddings with a random projection of semantic features.

    The semantic feature matrix (num_items, feat_dim) is projected into the
    GNN embedding space (num_items, emb_dim) via a fixed random orthogonal
    matrix so that users and items remain in the *same* space and dot-product
    scoring is numerically valid.

    The blend is 80% GNN + 20% content, both L2-normalised before mixing.
    This preserves the collaborative-filtering signal while injecting content
    bias from review sentiment, photo diversity, and TF-IDF topics.

    Parameters
    ----------
    emb   (num_items, emb_dim)   float32 – GNN item embeddings (L2-normalised)
    feat  (num_items, feat_dim)  float32 – semantic feature matrix (standardised)

    Returns
    -------
    (num_items, emb_dim) float32 – blended, L2-normalised embeddings
    """
    n       = min(emb.shape[0], feat.shape[0])
    emb_d   = emb.shape[1]
    feat_d  = feat.shape[1]
    emb_n   = emb[:n]                            # (n, emb_dim)
    feat_n  = feat[:n].astype(np.float32)        # (n, feat_dim)

    # Deterministic random projection: feat_dim → emb_dim
    # Fixed seed ensures the same projection is used every time.
    rng  = np.random.default_rng(seed=42)
    proj = rng.standard_normal((feat_d, emb_d)).astype(np.float32)
    # Orthonormalise columns for a stable isometric mapping
    proj, _ = np.linalg.qr(proj) if feat_d >= emb_d else (proj, None)
    proj = proj[:feat_d, :emb_d].astype(np.float32)  # ensure shape (feat_d, emb_d)

    feat_proj = feat_n @ proj            # (n, emb_dim)

    # L2-normalise the projected content vector
    f_norms   = np.linalg.norm(feat_proj, axis=1, keepdims=True)
    feat_proj = feat_proj / np.maximum(f_norms, 1e-8)

    # Weighted blend: collaborative signal dominates (α=0.80)
    aug   = 0.80 * emb_n + 0.20 * feat_proj
    norms = np.linalg.norm(aug, axis=1, keepdims=True)
    return aug / np.maximum(norms, 1e-8)


def _init_item_embeddings_from_features(
    model: torch.nn.Module,
    sem_feat: np.ndarray,
    num_users: int,
    emb_dim: int,
    alpha: float = 0.30,
) -> None:
    """
    Warm-start item embeddings with a linear projection of semantic features.

    Adds ``alpha`` × projected_content to the randomly initialised item rows
    of ``model.embedding.weight``, giving the GNN a content-aware starting
    point that significantly reduces the number of epochs needed to learn
    quality item representations.

    Parameters
    ----------
    model     : LightGCN / GraphSAGE / GAT model with ``embedding`` attribute.
    sem_feat  : (num_items, feat_dim) float32 standardised semantic features.
    num_users : offset into embedding table where items begin.
    emb_dim   : embedding dimension.
    alpha     : blend strength (0 = no effect, 1 = replace random init).
    """
    if not hasattr(model, "embedding"):
        return

    num_items = sem_feat.shape[0]
    feat_dim  = sem_feat.shape[1]

    # Random projection feat_dim → emb_dim (same seed as _augment_with_features)
    rng  = np.random.default_rng(seed=42)
    proj = rng.standard_normal((feat_dim, emb_dim)).astype(np.float32)
    if feat_dim >= emb_dim:
        proj, _ = np.linalg.qr(proj)
    proj = proj[:feat_dim, :emb_dim].astype(np.float32)

    content_emb = (sem_feat.astype(np.float32) @ proj)  # (num_items, emb_dim)
    # L2-normalise per row, then scale to match the normal-init std (~0.01)
    c_norms = np.linalg.norm(content_emb, axis=1, keepdims=True)
    content_emb = content_emb / np.maximum(c_norms, 1e-8) * 0.01

    with torch.no_grad():
        n = min(num_items, model.embedding.weight.shape[0] - num_users)
        delta = torch.from_numpy(content_emb[:n]).to(model.embedding.weight.device)
        model.embedding.weight[num_users: num_users + n] += alpha * delta


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSP/ICG Recommendation Pipeline — Yelp Academic Dataset"
    )
    parser.add_argument(
        "--config", type=str, default="configs/yelp.json",
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--data_dir", type=str, default="",
        help="Override data directory (useful when running without a full config)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="",
        help="Override output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists():
        config = ProjectConfig.from_json(str(cfg_path))
        print(f"[Yelp] Loaded config from: {cfg_path}")
    else:
        print(f"[Yelp] Config not found ({cfg_path}), using defaults")
        config = ProjectConfig()
        config.data.dataset_name = "yelp"

    if args.data_dir:
        config.data.dataset_path = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir

    if not config.data.dataset_path:
        config.data.dataset_path = "./data"

    run_yelp_pipeline(config)


if __name__ == "__main__":
    main()
