#!/usr/bin/env python3
"""
run_yelp_compressible.py  –  Yelp subset optimised for GSP compression & speedup
==================================================================================

MOTIVATION
----------
Standard "top-active user" subsetting yields high-degree users whose
Forman-Ricci curvature F(u,v) = 4 - deg(u) - deg(v) + |shared_items(u,v)|
is almost always negative → users never cluster → ~0 % compression → no speedup.

This script selects a *compressible* subset instead by running a four-stage
graph-theoretic pipeline before any model training:

PIPELINE
--------
  1. Load the full k-core filtered Yelp dataset (≥ min_interactions per user
     and item).

  2. Build the sparse user-user co-occurrence matrix UU = A @ A^T, where A is
     the binary user-item interaction matrix.  Keep only upper-triangle entries
     with shared ≥ min_shared to filter out noise.

  2b. [Cosine-similarity normalisation]  (enabled by default; disable with
     --no_cosine_sim)
     Normalise every UU edge by cosine similarity:
         S(u,v) = shared(u,v) / sqrt(deg(u) * deg(v))
     This removes the heavy-user bias so that merges reflect *taste similarity*
     rather than raw review volume.  The effective shared count fed into the
     curvature formula is:
         shared_eff(u,v) = S(u,v) * sqrt(deg(u) * deg(v))   (= shared(u,v),
     but the threshold uses S(u,v) to equalise high- and low-degree users.)

  3. Compute the Forman-Ricci curvature for every UU edge:
         F(u,v) = 4 - deg(u) - deg(v) + shared_eff(u,v)
     Score each user as the sum of its positive-curvature edge values:
         score(u) = Σ_{v∈N(u)} max(0, F(u,v))

  3b. [Heavy-Edge Matching — "Soft Cluster" step]  (enabled by default; disable
     with --no_hem_matching)
     Greedily merge each high-score user u with its unmatched neighbour v that
     maximises F(u,v) > 0.  Matched pairs form a Super-User and are both
     retained in the candidate pool.  This directly reduces the singleton count
     and increases the downstream GSP compression ratio.

  4. Keep only users whose score exceeds the score_percentile threshold (default
     75th percentile → top quarter).  Within that pool, greedily select users in
     descending score order until the interaction count reaches
     ~target_interactions.  Remap user/item IDs to contiguous 0-based integers.

  5. Run the baseline → GSP training → evaluation pipeline (identical to
     run_yelp_1m.py) on the compressible subset.

EVALUATION
----------
Leave-One-Out (LOO) split: for each user with ≥ 2 positives, hold out 1 positive
as the test item.  Ranking metrics are computed against 99 uniformly-sampled
negative items (the standard BPR/NCF/LightGCN protocol) at k = 10, 20, 50.

FLAGS
-----
Key knobs for the compressible-selection stage:

  --target_interactions   Target interaction count for the subset  [500 000]
                          Ignored when --target_fraction is set.
  --target_fraction       Fraction of filtered dataset to use      [None]
                          E.g. 0.25 → 25 % of filtered interactions.
                          Overrides --target_interactions.
  --min_shared            Min shared items for a UU edge           [2]
  --score_percentile      Percentile threshold for user selection  [75.0]
                          Lower = bigger pool; higher = denser clusters.
                          Try 40–60 for more users, 80–90 for tighter clusters.
  --use_cosine_sim /      Cosine-normalise UU before curvature     [on]
  --no_cosine_sim         (off = use raw shared counts)
  --hem_matching /        Run Heavy-Edge Matching soft cluster step [on]
  --no_hem_matching       (off = keep original per-user selection)

Typical usage
-------------
    # Quick single model, ~500 k interactions, compressible subset
    python scripts/run_yelp_compressible.py --data_dir ./data --epochs 30 \\
        --target_interactions 500000 --models lightgcn

    # All models, ~1 M interactions
    python scripts/run_yelp_compressible.py --data_dir ./data \\
        --target_interactions 1000000 --models lightgcn,gat,graphsage,gcn --epochs 50

    # Fraction-based targeting: use 25 % of the filtered dataset
    python scripts/run_yelp_compressible.py --data_dir ./data \\
        --target_fraction 0.25 --models lightgcn,gat,graphsage,gcn --epochs 50

    # Wider user pool (40th-percentile threshold) for sparser graphs
    python scripts/run_yelp_compressible.py --data_dir ./data \\
        --target_interactions 1000000 --score_percentile 40.0 --epochs 50
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
import pandas as pd
import scipy.sparse as sp
import torch
from collections import defaultdict

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
from gsprec.analytics import run_analytics_pipeline
from gsprec.data.semantic_features import (
    SemanticFeatureConfig,
    extract_semantic_features,
    load_semantic_features,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Yelp *compressible* subset pipeline – selects users that "
                    "maximise GSP compression and training speedup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", default="./data",
                   help="Directory with Yelp JSONL files (or parent of yelp_dataset/)")
    p.add_argument("--output_dir", default="outputs/yelp_compressible",
                   help="Where to write results")
    p.add_argument("--target_interactions", type=int, default=500_000,
                   help="Target interaction count for the compressible subset. "
                        "Ignored when --target_fraction is set.")
    p.add_argument("--target_fraction", type=float, default=None,
                   help="Keep the top X%% most active users by interaction count "
                        "(e.g. 0.25 = top 25%% of users).  Fast, deterministic "
                        "alternative to curvature-based selection. "
                        "Overrides --target_interactions.  Range: (0.0, 1.0].")
    p.add_argument("--min_interactions", type=int, default=10,
                   help="k-core filter: minimum interactions per user/item")
    p.add_argument("--min_shared", type=int, default=2,
                   help="Minimum shared items for a UU edge (used both in user "
                        "selection scoring and in gsp_preprocess)")
    p.add_argument("--score_percentile", type=float, default=90.0,
                   help="Only consider users with positive-curvature score above "
                        "this percentile of the score distribution (0 = all users, "
                        "90 = top decile for aggressive coarsening).  Lower = bigger pool.")
    p.add_argument("--use_cosine_sim", action="store_true", default=True,
                   help="Normalise UU co-occurrence by cosine similarity "
                        "(S_ij = shared / sqrt(deg_i * deg_j)) before computing "
                        "Forman-Ricci curvature, removing the heavy-user bias.")
    p.add_argument("--no_cosine_sim", dest="use_cosine_sim", action="store_false",
                   help="Disable cosine-similarity normalisation and use raw shared counts.")
    p.add_argument("--hem_matching", action="store_true", default=True,
                   help="Run Heavy-Edge Matching after scoring: greedily merge each "
                        "high-score user with their highest-curvature neighbour to "
                        "form Super-Users, reducing singleton count.")
    p.add_argument("--no_hem_matching", dest="hem_matching", action="store_false",
                   help="Disable Heavy-Edge Matching; keep original per-user selection.")
    p.add_argument("--full_dataset", action="store_true", default=False,
                   help="Skip compressible user selection and train on the full "
                        "k-core filtered dataset.  All other flags (GSP, eval, "
                        "etc.) are unchanged.  Useful for a direct baseline "
                        "comparison against the compressible subset.")
    p.add_argument("--models", default="lightgcn,gat,graphsage,gcn",
                   help="Comma-separated model list")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--early_stopping_patience", type=int, default=10,
                   help="Stop after this many epochs with no loss improvement. 0 = disabled.")
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch_size", type=int, default=65536)
    p.add_argument("--neg_ratio", type=int, default=4)
    p.add_argument("--implicit_threshold", type=float, default=3.5,
                   help="Minimum star rating to treat as a positive interaction")
    p.add_argument("--eval_k", type=int, default=10)
    p.add_argument("--eval_negatives", type=int, default=99,
                   help="Number of negative items sampled per user at evaluation "
                        "(standard LOO-99 protocol)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tfidf_dim", type=int, default=64,
                   help="TruncatedSVD output dimension for TF-IDF semantic features")
    # GSP knobs (same as run_yelp_1m)
    p.add_argument("--curvature_percentile", type=float, default=50.0,
                   help="Stage I: keep edges with curvature above this percentile "
                        "(lower = more edges retained = more clusters). "
                        "Default 50.0 keeps the top half of UU edges for clustering.")
    p.add_argument("--curvature_mode", default="cosine",
                   choices=["cosine", "forman_ricci"],
                   help="Curvature metric used for UU edge scoring in gsp_preprocess. "
                        "'cosine': F=shared/sqrt(deg_u*deg_v), always in (0,1], "
                        "works well for sparse graphs like Yelp. "
                        "'forman_ricci': F=4-deg_u-deg_v+shared (classic formula), "
                        "produces positive values only when shared > deg_u+deg_v-4; "
                        "best for dense graphs (MovieLens-25M, Amazon).")
    p.add_argument("--curvature_topk", type=int, default=None)
    p.add_argument("--max_cluster_size", type=int, default=50,
                   help="Maximum users per super-node. 50 enables 20-30%% node compression.")
    p.add_argument("--clustering_method", default="hem",
                   choices=["hem", "connected_components"],
                   help="Clustering algorithm for Stage I-c. "
                        "'hem' (default) = Heavy-Edge Matching: greedily pairs users "
                        "by highest curvature, giving ~40-50%% compression on dense "
                        "graphs like Yelp (no giant-component issue). "
                        "'connected_components' = original; fails on dense graphs where "
                        "all users land in one giant component.")
    p.add_argument("--er_eigvecs", type=int, default=16)
    p.add_argument("--er_node_limit", type=int, default=0)
    p.add_argument("--er_solver", default="dwlv",
                   choices=["arpack", "lobpcg", "jl", "dwlv"],
                   help="ER solver. 'dwlv' = Degree-weighted Local Variation O(nnz) "
                        "approximation: <60s on 173M Yelp edges, no eigenvectors. "
                        "'jl' = JL-sketch MINRES (~170min). 'arpack' = exact but slow.")
    p.add_argument("--er_sketches", type=int, default=32)
    p.add_argument("--no_amp", action="store_true")
    # ── Semantic integration ───────────────────────────────────────────────
    p.add_argument("--sentence_transformer_model", default="all-MiniLM-L6-v2",
                   help="HuggingFace Sentence-Transformer model for user review embeddings. "
                        "Embeddings are cached after first run. Set to empty string to skip.")
    p.add_argument("--semantic_alpha", type=float, default=0.5,
                   help="Blend weight for Stage-I hybrid curvature: "
                        "S(u,v)=semantic_alpha*F_graph(u,v)+(1-semantic_alpha)*TextCosine(u,v). "
                        "0=pure text similarity, 1=pure graph curvature.")
    p.add_argument("--max_reviews_per_user", type=int, default=5,
                   help="Max reviews per user to encode for semantic embeddings.")
    p.add_argument("--max_review_chars", type=int, default=256,
                   help="Max characters per review for semantic encoding.")
    p.add_argument("--st_batch_size", type=int, default=2048,
                   help="Batch size for sentence-transformer encoding.")
    # ── GAT architecture ───────────────────────────────────────────────────
    p.add_argument("--gat_heads", type=int, default=8,
                   help="Number of attention heads in GAT layer 1 (default 8 for "
                        "richer attention on semantically-pruned edges).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Any) -> None:
    _mkdir(os.path.dirname(os.path.abspath(path)))

    def _default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def _section(title: str) -> None:
    bar = "═" * (len(title) + 4)
    print(f"\n[COMP] {bar}")
    print(f"[COMP] ║  {title}  ║")
    print(f"[COMP] {bar}")


def _l2_norm(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


def _proc_cpu_time_s() -> float:
    try:
        with open("/proc/self/stat") as fh:
            fields = fh.read().split()
        return (int(fields[13]) + int(fields[14])) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return 0.0


def _sys_loadavg() -> str:
    try:
        return f"{os.getloadavg()[0]:.2f}"
    except Exception:
        return "n/a"


# ─────────────────────────────────────────────────────────────────────────────
# Semantic user embeddings via Sentence-Transformer
# ─────────────────────────────────────────────────────────────────────────────

def _compute_user_semantic_embeddings(
    data_dir: str,
    unique_users: np.ndarray,       # Original string user IDs (sorted, 0-based index = integer user ID)
    model_name: str = "all-MiniLM-L6-v2",
    max_reviews_per_user: int = 5,
    max_chars_per_review: int = 256,
    batch_size: int = 2048,
    cache_dir: str = "outputs/cache",
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """Compute or load cached user-level text embeddings from review data.

    For each user, concatenates up to ``max_reviews_per_user`` review snippets
    (each truncated to ``max_chars_per_review`` chars) and encodes them with a
    Sentence-Transformer model.  Embeddings are L2-normalised.

    Returns an (num_users, emb_dim) float32 ndarray indexed by integer user ID,
    or None if sentence-transformers is not installed.

    The result is cached at ``cache_dir/user_sem_emb_{num_users}_{model}.npy``
    so subsequent runs are instant.
    """
    if not model_name:
        print("[COMP] Sentence-Transformer model name empty; skipping semantic embeddings.")
        return None

    # --- Try to import sentence_transformers ---
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "[COMP] WARNING: sentence-transformers not installed. "
            "Install with:  pip install sentence-transformers\n"
            "[COMP] Falling back to pure graph-curvature scoring."
        )
        return None

    num_users = len(unique_users)
    safe_model = model_name.replace("/", "_")
    cache_path = os.path.join(
        cache_dir, f"user_sem_emb_{num_users}_{safe_model}.npy"
    )
    if os.path.exists(cache_path):
        emb = np.load(cache_path)
        if emb.shape[0] == num_users:
            print(
                f"[COMP] Loaded cached user semantic embeddings: "
                f"shape={emb.shape}"
            )
            return emb.astype(np.float32)
        print("[COMP] Cached user embeddings shape mismatch; recomputing ...")

    # --- Build user_id → review texts index (streaming, no full load) ---
    from gsprec.data.yelp_dataset import _resolve_data_dir, _REVIEW_FILE
    import json as _json

    data_dir_resolved = _resolve_data_dir(data_dir)
    review_file = os.path.join(data_dir_resolved, _REVIEW_FILE)
    if not os.path.exists(review_file):
        print(f"[COMP] Review file not found at '{review_file}'; skipping semantic embeddings.")
        return None

    print(f"[COMP] Building user→review index from {review_file} ...")
    # Build a set of target user IDs for fast lookup
    target_users: set = set(unique_users.tolist())
    user_texts: dict = defaultdict(list)
    t0 = time.perf_counter()
    with open(review_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except Exception:
                continue
            uid = obj.get("user_id", "")
            text = obj.get("text", "").strip()
            if uid in target_users and text:
                if len(user_texts[uid]) < max_reviews_per_user:
                    user_texts[uid].append(text[:max_chars_per_review])
    print(
        f"[COMP] User→review index built in {time.perf_counter()-t0:.1f}s | "
        f"{len(user_texts):,}/{num_users:,} users have review text"
    )

    # --- Build per-user text strings (sorted by unique_users order → integer ID order) ---
    all_texts: List[str] = []
    for uid in unique_users:  # unique_users is sorted → index aligns with integer user ID
        reviews = user_texts.get(str(uid), [])
        if reviews:
            all_texts.append(" ".join(reviews))
        else:
            all_texts.append("")   # empty → will produce a near-zero embedding

    # --- Encode with Sentence-Transformer ---
    st_device = device if device.startswith("cuda") else "cpu"
    print(
        f"[COMP] Encoding {num_users:,} user texts with '{model_name}' "
        f"(batch={batch_size}, device={st_device}) ..."
    )
    t0 = time.perf_counter()
    st_model = SentenceTransformer(model_name, device=st_device)

    # Replace empty strings with a generic placeholder so the model
    # doesn't produce degenerate outputs
    texts_to_encode = [t if t else "[UNK]" for t in all_texts]
    emb = st_model.encode(
        texts_to_encode,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise → cosine = dot product
    ).astype(np.float32)

    print(
        f"[COMP] Sentence-Transformer encoding done in "
        f"{time.perf_counter()-t0:.1f}s  shape={emb.shape}"
    )

    # Zero out embeddings for users with no reviews (avoids spurious cosine similarity)
    empty_mask = np.array([t == "" for t in all_texts])
    emb[empty_mask] = 0.0

    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, emb)
    print(f"[COMP] User semantic embeddings cached -> {cache_path}")
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Compressible-user selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_compressible_users(
    ratings_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    target_interactions: int,
    min_shared: int,
    score_percentile: float,
    seed: int,
    use_cosine_sim: bool = True,
    hem_matching: bool = True,
    implicit_threshold: float = 0.0,
    user_text_emb: Optional[np.ndarray] = None,   # (num_users, emb_dim) or None
    semantic_alpha: float = 0.5,                   # weight for graph curvature in hybrid score
) -> Tuple[pd.DataFrame, int, int]:
    """Select a compressible subset of users targeting ~target_interactions.

    Strategy
    --------
    1. Build binary user-item CSR matrix A (U × I).
    2. Compute sparse UU = A @ A^T; keep only upper-triangle with shared ≥ 1.
    2b. (optional) Normalise UU by cosine similarity:
           S(u,v) = shared(u,v) / sqrt(deg(u) * deg(v))
    3. Compute Forman-Ricci curvature for each UU edge.
    3a. (NEW) If user_text_emb is provided, compute text cosine similarity for
        each UU edge and blend with graph curvature:
           F_hybrid(u,v) = semantic_alpha * F_graph(u,v)
                          + (1-semantic_alpha) * TextCosine(u,v)
        This incorporates review-text semantic similarity into Stage I scoring,
        enabling clustering of users who are semantically similar even when
        their interaction graphs do not perfectly overlap.
    3b. (optional) Heavy-Edge Matching: greedily merge each high-score user with
        their best positive-curvature neighbour into a Super-User.
    4. Score each user = sum of max(0, F_hybrid(u,v)) over its neighbours.
       Keep users above score_percentile threshold.
    5. Greedy target-size selection, then remap IDs to contiguous 0-based integers.

    Parameters
    ----------
    user_text_emb : (num_users, emb_dim) float32 ndarray or None.
        Pre-computed, L2-normalised sentence-transformer embeddings for each user.
        If None, only graph curvature is used.
    semantic_alpha : float in [0, 1].
        Blend weight: 1.0 = pure graph curvature, 0.0 = pure text similarity.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Step 1: binary user-item CSR — use only positive interactions
    # (same filter applied by gsp_preprocess so deg values are consistent).
    if implicit_threshold > 0.0:
        pos_mask = ratings_df["Rating"].to_numpy(dtype=np.float32) >= implicit_threshold
        pos_df = ratings_df.loc[pos_mask]
        print(
            f"[COMP] Using ratings ≥ {implicit_threshold} for UU graph: "
            f"{int(pos_mask.sum()):,} / {len(ratings_df):,} interactions"
        )
    else:
        pos_df = ratings_df
    users = pos_df["UserID"].to_numpy(dtype=np.int32)
    items = pos_df["BusinessID"].to_numpy(dtype=np.int32)
    A = sp.csr_matrix(
        (np.ones(len(users), dtype=np.float32), (users, items)),
        shape=(num_users, num_items),
        dtype=np.float32,
    )
    # binarize
    A.data[:] = 1.0

    # User degree (number of unique items rated)
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)

    # Step 2: UU co-occurrence (sparse); compute only upper-triangle
    UU = A @ A.T  # (U × U) sparse, diagonal = deg
    UU = sp.triu(UU, k=1).tocoo()  # upper triangle, no diagonal

    row = UU.row.astype(np.int32)
    col = UU.col.astype(np.int32)
    shared = UU.data.astype(np.float32)

    # Build UU adjacency using shared ≥ 1 for maximum connectivity, then
    # separately track edges meeting the stricter min_shared threshold for
    # curvature quality metrics.  This decouples graph connectivity (needs
    # many edges) from curvature scoring quality (prefers high-overlap pairs).
    adjacency_mask = shared >= 1
    row, col, shared = row[adjacency_mask], col[adjacency_mask], shared[adjacency_mask]

    print(
        f"[COMP] UU edges (shared ≥ 1): {len(row):,}  "
        f"({time.perf_counter()-t0:.1f}s)"
    )
    # Report how many meet the stricter min_shared threshold
    n_strong = int((shared >= min_shared).sum())
    print(
        f"[COMP] Strong UU edges (shared ≥ {min_shared}): {n_strong:,} "
        f"({100*n_strong/max(len(row),1):.1f}%)"
    )

    # Step 2b: Cosine-similarity normalisation (removes heavy-user bias).
    # -----------------------------------------------------------------------
    # Raw shared counts inflate curvature for heavy users even when their
    # taste overlap is low.  We normalise by cosine similarity:
    #     S(u,v) = shared(u,v) / sqrt(deg(u) * deg(v))
    # so that the curvature signal reflects *taste similarity*, not volume.
    #
    # For the curvature formula we then use:
    #     shared_eff(u,v) = S(u,v) * (deg(u) + deg(v)) / 2
    # which re-scales back to a per-user-pair magnitude while keeping the
    # cosine direction. This makes F(u,v) positive whenever S(u,v) > 1 -
    # 2/(deg_u + deg_v), i.e. for moderately similar users regardless of
    # their absolute degree.
    if use_cosine_sim:
        denom = np.sqrt(deg[row].astype(np.float64) * deg[col].astype(np.float64))
        denom = np.where(denom > 0, denom, 1.0)
        cosine_sim = (shared.astype(np.float64) / denom).astype(np.float32)
        # Clamp to [0, 1] (floating-point safety)
        cosine_sim = np.clip(cosine_sim, 0.0, 1.0)
        # Rescale: shared_eff replaces raw shared in the curvature formula
        avg_deg = 0.5 * (deg[row] + deg[col])
        shared_eff = (cosine_sim * avg_deg).astype(np.float32)
        print(
            f"[COMP] Cosine-sim: S in [{float(cosine_sim.min()):.4f}, "
            f"{float(cosine_sim.max()):.4f}]  "
            f"shared_eff in [{float(shared_eff.min()):.2f}, {float(shared_eff.max()):.2f}]"
        )
    else:
        shared_eff = shared
        cosine_sim = None

    # Step 3: Forman-Ricci curvature per edge using (normalised) shared count.
    # -----------------------------------------------------------------------
    # F(u,v) = 4 - deg(u) - deg(v) + shared_eff(u,v)
    # With cosine normalisation, shared_eff ≈ S * avg_deg, so for users with
    # cosine_sim > 1 - 4/avg_deg the curvature is positive — achievable for
    # users sharing > ~10–20 % of their respective histories.
    curvature = 4.0 - deg[row] - deg[col] + shared_eff

    # Step 3a: Hybrid scoring with text embeddings (if available).
    # -----------------------------------------------------------------------
    # S_hybrid(u,v) = semantic_alpha * F_normalised(u,v)
    #               + (1-semantic_alpha) * TextCosine(u,v)
    # TextCosine is in [0, 1] for L2-normalised embeddings (dot product).
    # F_normalised is min-max normalised into [0, 1] to put both signals on
    # the same scale before blending.
    if user_text_emb is not None and semantic_alpha < 1.0:
        print(
            f"[COMP] Computing text cosine similarity for {len(row):,} UU edges "
            f"(semantic_alpha={semantic_alpha}) ..."
        )
        t_sem = time.perf_counter()
        # Batch dot-product between L2-normalised embeddings → cosine in [-1, 1]
        # For L2-normalised vectors this equals cosine similarity directly.
        emb_row = user_text_emb[row].astype(np.float64)
        emb_col = user_text_emb[col].astype(np.float64)
        text_cosine = np.einsum("ij,ij->i", emb_row, emb_col).astype(np.float32)
        text_cosine = np.clip(text_cosine, 0.0, 1.0)  # clamp to [0, 1]

        # Normalise graph curvature to [0, 1]
        curv_min = float(curvature.min())
        curv_max = float(curvature.max())
        curv_norm = (curvature - curv_min) / max(curv_max - curv_min, 1e-8)

        # Hybrid: weighted blend of normalised graph signal and text similarity
        curvature = (
            semantic_alpha * curv_norm + (1.0 - semantic_alpha) * text_cosine
        ).astype(np.float32)
        print(
            f"[COMP] Hybrid curvature in [{float(curvature.min()):.4f}, "
            f"{float(curvature.max()):.4f}]  "
            f"({time.perf_counter()-t_sem:.1f}s)"
        )
    else:
        if user_text_emb is not None:
            print("[COMP] semantic_alpha=1.0: using pure graph curvature (text emb ignored).")
        else:
            print("[COMP] No text embeddings; using pure graph curvature.")

    # Step 4: positive-curvature score per user
    pos_curv = np.maximum(curvature, 0.0)
    score = np.zeros(num_users, dtype=np.float32)
    np.add.at(score, row, pos_curv)
    np.add.at(score, col, pos_curv)

    positive_edge_flag = curvature > 0
    n_pos_edges = int(positive_edge_flag.sum())
    print(
        f"[COMP] Positive-curvature edges: {n_pos_edges:,} / {len(row):,}  "
        f"({100*n_pos_edges/max(len(row),1):.1f}%)"
    )

    # Step 3b: Heavy-Edge Matching (HEM) — "Soft Cluster" step.
    # -----------------------------------------------------------------------
    # For every high-score user u, find the unmatched neighbour v that
    # maximises F(u,v) > 0 and merge them into a Super-User.  Both u and v
    # are added to the candidate pool (not just the higher-score one).
    # This directly reduces singletons before the GSP step.
    if hem_matching and n_pos_edges > 0:
        # Build a sorted list of positive-curvature edges (descending F)
        pos_mask_he = curvature > 0
        pos_row = row[pos_mask_he]
        pos_col = col[pos_mask_he]
        pos_F   = curvature[pos_mask_he]

        # Sort edges by curvature descending (greedy heavy-edge first)
        sort_idx   = np.argsort(-pos_F)
        pos_row    = pos_row[sort_idx]
        pos_col    = pos_col[sort_idx]

        matched     = np.zeros(num_users, dtype=bool)
        hem_partners: dict = {}   # user_id -> partner_id

        for i in range(len(pos_row)):
            u_i = int(pos_row[i])
            v_i = int(pos_col[i])
            if not matched[u_i] and not matched[v_i]:
                matched[u_i] = True
                matched[v_i] = True
                hem_partners[u_i] = v_i
                hem_partners[v_i] = u_i

        n_pairs = len(hem_partners) // 2
        print(
            f"[COMP] HEM: {n_pairs:,} Super-User pairs formed "
            f"({len(hem_partners):,} users matched)"
        )

        # Boost the score of HEM-matched users so they rank above singletons
        hem_boost = float(score.max()) + 1.0
        for uid in hem_partners:
            score[uid] += hem_boost
    else:
        hem_partners = {}
        if hem_matching:
            print("[COMP] HEM: skipped (no positive-curvature edges before matching)")

    # Step 5: thresholding + greedy selection.
    # -----------------------------------------------------------------------
    # Threshold is computed on *all* nonzero scores (HEM-boosted users are
    # always above the threshold since their score > original max).
    if score_percentile > 0.0 and score_percentile < 100.0:
        nonzero_scores = score[score > 0]
        if len(nonzero_scores) > 0:
            threshold = float(np.percentile(nonzero_scores, score_percentile))
        else:
            threshold = 0.0
    else:
        threshold = 0.0

    candidate_mask = score >= threshold
    # Always include both members of a HEM pair if either is a candidate
    if hem_partners:
        partner_ids = np.array(list(hem_partners.keys()), dtype=np.int64)
        candidate_mask[partner_ids] = True

    n_candidates = int(candidate_mask.sum())
    print(
        f"[COMP] Candidate users (score ≥ {threshold:.4f}, "
        f"percentile={score_percentile}): {n_candidates:,}"
    )

    candidate_ids = np.where(candidate_mask)[0]
    user_interaction_count = (
        ratings_df.groupby("UserID")["BusinessID"].count()
        .reindex(candidate_ids, fill_value=0)
        .to_numpy(dtype=np.int64)
    )
    # Primary sort: score descending; secondary: interaction count descending
    order = np.lexsort((-user_interaction_count, -score[candidate_ids]))
    sorted_candidates = candidate_ids[order]

    # Greedy cumulative selection
    cumulative_interactions = np.cumsum(user_interaction_count[order])
    if target_interactions > 0:
        cutoff_idx = int(np.searchsorted(cumulative_interactions, target_interactions))
        # include the user that pushed us over
        cutoff_idx = min(cutoff_idx + 1, len(sorted_candidates))
        selected_users = sorted_candidates[:cutoff_idx]
    else:
        selected_users = sorted_candidates

    selected_set = set(selected_users.tolist())
    sub = ratings_df[ratings_df["UserID"].isin(selected_set)].copy()

    # Step 6: remap to contiguous 0-based IDs
    uniq_u = np.sort(sub["UserID"].unique())
    uniq_i = np.sort(sub["BusinessID"].unique())
    sub["UserID"] = np.searchsorted(uniq_u, sub["UserID"].to_numpy())
    sub["BusinessID"] = np.searchsorted(uniq_i, sub["BusinessID"].to_numpy())
    sub = sub.reset_index(drop=True)

    new_num_users = int(len(uniq_u))
    new_num_items = int(len(uniq_i))

    elapsed = time.perf_counter() - t0
    print(
        f"[COMP] Selected {new_num_users:,} users | {new_num_items:,} items | "
        f"{len(sub):,} interactions | {elapsed:.1f}s"
    )
    return sub, new_num_users, new_num_items, uniq_i


def _select_top_frequent_users(
    ratings_df: pd.DataFrame,
    target_fraction: float,
    seed: int = 42,
) -> Tuple[pd.DataFrame, int, int, np.ndarray]:
    """Select the top X% most active users by interaction count.

    Sorts all users by their number of rated items (descending) and retains
    the top ``ceil(target_fraction * num_users)`` users.  IDs are remapped to
    contiguous 0-based integers.

    Parameters
    ----------
    target_fraction : float in (0, 1]
        Fraction of unique users to keep (e.g. 0.25 retains the 25% most active).

    Returns
    -------
    sub             : filtered + remapped DataFrame
    new_num_users   : number of retained users
    new_num_items   : number of unique items in the filtered subset
    uniq_i          : original 0-based item IDs (for semantic-feature alignment)
    """
    t0 = time.perf_counter()
    interaction_counts = ratings_df.groupby("UserID").size()
    num_users_total = interaction_counts.shape[0]
    num_to_select = max(1, int(math.ceil(target_fraction * num_users_total)))
    top_users = interaction_counts.nlargest(num_to_select).index.to_numpy()
    selected_set = set(top_users.tolist())
    sub = ratings_df[ratings_df["UserID"].isin(selected_set)].copy()
    # Remap to contiguous 0-based IDs
    uniq_u = np.sort(sub["UserID"].unique())
    uniq_i = np.sort(sub["BusinessID"].unique())
    sub["UserID"] = np.searchsorted(uniq_u, sub["UserID"].to_numpy())
    sub["BusinessID"] = np.searchsorted(uniq_i, sub["BusinessID"].to_numpy())
    sub = sub.reset_index(drop=True)
    new_num_users = int(len(uniq_u))
    new_num_items = int(len(uniq_i))
    elapsed = time.perf_counter() - t0
    print(
        f"[COMP] Top-frequent-users: kept {new_num_users:,}/{num_users_total:,} users "
        f"(fraction={target_fraction:.4f}) | {new_num_items:,} items | "
        f"{len(sub):,} interactions | {elapsed:.1f}s"
    )
    return sub, new_num_users, new_num_items, uniq_i


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers (same protocol as run_yelp_1m.py)
# ─────────────────────────────────────────────────────────────────────────────

def _split_leave_one_out(
    df: pd.DataFrame, threshold: float, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standard LOO split: hold out 1 positive per user (≥2 positives required).

    All other interactions remain in training so the model sees the full graph.
    """
    rng = np.random.default_rng(seed)
    test_idx: List[int] = []
    for uid, grp in df.groupby("UserID"):
        pos_mask = grp["Rating"].to_numpy(dtype=np.float32) >= threshold
        pos_idxs = grp.index[pos_mask].to_numpy()
        if len(pos_idxs) >= 2:
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


def _augment_with_features(
    item_emb: np.ndarray,
    sem_feat: np.ndarray,
    content_weight: float = 0.20,
    seed: int = 42,
) -> np.ndarray:
    """Blend GNN item embeddings with projected semantic features.

    Uses a fixed random orthogonal projection to map the (feat_dim)-dimensional
    semantic vector into the same (emb_dim)-dimensional space as the GNN output,
    then blends the two with a weighted sum so dot-product scoring remains valid.
    """
    n = min(item_emb.shape[0], sem_feat.shape[0])
    emb_d = item_emb.shape[1]
    feat_d = sem_feat.shape[1]

    emb_n = item_emb[:n].astype(np.float32)
    feat_n = sem_feat[:n].astype(np.float64)

    # Standardise features before projection
    feat_std = feat_n.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std > 0, feat_std, 1.0)
    feat_n = (feat_n / feat_std).astype(np.float32)

    # Fixed random orthogonal projection: feat_dim → emb_dim
    rng = np.random.default_rng(seed)
    proj_raw = rng.standard_normal((feat_d, emb_d))
    proj, _ = np.linalg.qr(proj_raw) if feat_d >= emb_d else np.linalg.qr(proj_raw.T)
    if feat_d < emb_d:
        proj = proj.T  # (feat_d, emb_d)

    feat_proj = feat_n @ proj  # (n, emb_d)

    # Normalise projected features to match GNN embedding scale
    feat_scale = np.linalg.norm(feat_proj, axis=1, keepdims=True).mean()
    emb_scale = np.linalg.norm(emb_n, axis=1, keepdims=True).mean()
    if feat_scale > 1e-8:
        feat_proj = feat_proj * (emb_scale / feat_scale)

    blended = (1.0 - content_weight) * emb_n + content_weight * feat_proj
    result = item_emb.copy()
    result[:n] = blended
    return result


def _init_item_embeddings_from_features(
    model: torch.nn.Module,
    sem_feat: np.ndarray,
    num_users: int,
    emb_dim: int,
    alpha: float = 0.30,
    seed: int = 42,
) -> None:
    """Warm-start item rows of the GNN embedding table from semantic features.

    Applies the same fixed random orthogonal projection used in
    _augment_with_features, blending the projected content vector with the
    randomly-initialised embedding (alpha=content share).
    """
    if not hasattr(model, "embedding"):
        return

    feat_d = sem_feat.shape[1]
    n_items = sem_feat.shape[0]

    rng = np.random.default_rng(seed)
    proj_raw = rng.standard_normal((feat_d, emb_dim))
    proj, _ = np.linalg.qr(proj_raw) if feat_d >= emb_dim else np.linalg.qr(proj_raw.T)
    if feat_d < emb_dim:
        proj = proj.T  # (feat_d, emb_dim)

    feat = sem_feat.astype(np.float64)
    feat_std = feat.std(axis=0, keepdims=True)
    feat = (feat / np.where(feat_std > 0, feat_std, 1.0)).astype(np.float32)

    feat_proj = feat @ proj  # (n_items, emb_dim)
    norms = np.linalg.norm(feat_proj, axis=1, keepdims=True)
    feat_proj = feat_proj / np.where(norms > 1e-8, norms, 1.0)
    feat_proj *= 0.01  # scale to match typical nn.init.normal_(std=0.01)

    with torch.no_grad():
        W = model.embedding.weight  # (num_users + num_items, emb_dim)
        n = min(n_items, W.shape[0] - num_users)
        feat_t = torch.from_numpy(feat_proj[:n]).to(W.device, dtype=W.dtype)
        W[num_users: num_users + n] = (
            (1.0 - alpha) * W[num_users: num_users + n] + alpha * feat_t
        )


def _eval_multi_k(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    test_positives: Dict[int, List[int]],
    seen_positives: Dict[int, Set[int]],
    num_negatives: int,
    seed: int,
    ks: Tuple[int, ...] = _EVAL_KS,
) -> Dict[str, float]:
    """Evaluate ranking at multiple k values sharing the same 99-negative pool."""
    merged: Dict[str, float] = {}
    users_eval = 0.0
    for k in ks:
        cfg = RankingEvalConfig(k=k, num_negatives=num_negatives, seed=seed)
        m = evaluate_ranking_from_embeddings(
            user_emb, item_emb, test_positives, seen_positives, cfg
        )
        users_eval = m.pop("UsersEvaluated", users_eval)
        merged.update(m)
    merged["UsersEvaluated"] = users_eval
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_paper_table(rows: List[Dict], ks: Tuple[int, ...] = _EVAL_KS) -> None:
    cols_rank = [f"{m}@{k}" for k in ks for m in ("Precision", "Recall", "NDCG", "HitRate")]
    cols_reg = ["RMSE", "MAE"]
    header = (
        f"{'Model':<18}  {'Type':<18}  "
        + "  ".join(f"{c:>14}" for c in cols_rank + cols_reg)
        + f"  {'Train(s)':>10}"
    )
    print("\n" + "=" * len(header))
    print("PAPER RESULTS TABLE  (compressible subset)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = [f"{r.get(c, float('nan')):>14.4f}" for c in cols_rank + cols_reg]
        print(
            f"{r['model']:<18}  {r['run_type']:<18}  "
            + "  ".join(vals)
            + f"  {r.get('training_time_s', 0):>10.1f}"
        )
    print("=" * len(header))

    # LaTeX
    n_cols = len(cols_rank) + len(cols_reg)
    print("\n--- LaTeX snippet ---")
    print(r"\begin{tabular}{ll" + "r" * n_cols + r"}")
    print(r"\toprule")
    rank_headers = " & ".join(cols_rank)
    print(f"Model & Type & {rank_headers} & RMSE & MAE \\\\")
    print(r"\midrule")
    for r in rows:
        model_tex = r["model"].replace("_", r"\_")
        type_tex = r["run_type"].replace("_", r"\_")
        vals = " & ".join(
            f"{r.get(c, float('nan')):.4f}" for c in cols_rank + cols_reg
        )
        print(f"{model_tex} & {type_tex} & {vals} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def _save_results_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    _mkdir(os.path.dirname(os.path.abspath(path)))
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_to_run = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    # Build preliminary selection-strategy label from CLI args (curvature_mode
    # controls gsp_preprocess; use_cosine_sim + hem_matching control user selection)
    _prelim_curv = "cosine_sim" if args.use_cosine_sim else "forman_ricci"
    _prelim_hem = "_hem" if args.hem_matching else ""
    _prelim_strategy = (
        "full_dataset"
        if args.full_dataset
        else f"curvature_{_prelim_curv}{_prelim_hem}_score_selection"
    )

    print(f"[COMP] Device         : {device}")
    print(f"[COMP] Models         : {models_to_run}")
    print(f"[COMP] Epochs         : {args.epochs}")
    print(f"[COMP] Curvature mode : {args.curvature_mode}  (GSP preprocessing)")
    print(f"[COMP] Selection      : {_prelim_strategy}")
    if args.target_fraction is not None:
        print(f"[COMP] Target         : {args.target_fraction:.4f} \u00d7 filtered dataset size")
    else:
        print(f"[COMP] Target         : ~{args.target_interactions:,} interactions")
    print(f"[COMP] Load1          : {_sys_loadavg()} (1-min load avg)")
    print(f"[COMP] Output         : {out_dir}")

    hw = collect_hardware_info()
    hw.update({
        "device": device,
        "seed": args.seed,
        "models": models_to_run,
        "target_interactions": args.target_interactions,
        "target_fraction": args.target_fraction,
        "selection_strategy": _prelim_strategy,
        "gsp_curvature_mode": args.curvature_mode,
        "use_cosine_sim_selection": args.use_cosine_sim,
        "hem_matching": args.hem_matching,
        "semantic_integration": bool(args.sentence_transformer_model),
        "semantic_alpha": args.semantic_alpha,
        "gat_heads": args.gat_heads,
    })
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
    unique_items_full: np.ndarray = yelp.get("unique_items", np.array([], dtype=object))
    unique_users_full: np.ndarray = yelp.get("unique_users", np.array([], dtype=object))
    print(
        f"[COMP] Full dataset: {yelp['num_users']:,} users | {yelp['num_items']:,} items | "
        f"{len(ratings_full):,} interactions | load={load_time:.1f}s"
    )

    # ── Resolve effective subset-size target (fraction OR raw count) ──────────
    if args.full_dataset:
        effective_target = 0  # unused — full k-core dataset is retained
        subset_method_label = "full_dataset (k-core filtered, no subset selection)"
    elif args.target_fraction is not None:
        if not (0.0 < args.target_fraction <= 1.0):
            raise ValueError(
                f"--target_fraction must be in (0, 1], got {args.target_fraction}"
            )
        effective_target = 0  # resolved per-user inside _select_top_frequent_users
        _n_full_users = ratings_full["UserID"].nunique()
        _n_selected = max(1, int(math.ceil(args.target_fraction * _n_full_users)))
        subset_method_label = (
            f"top_frequent_users fraction={args.target_fraction:.4f} "
            f"\u2192 top {_n_selected:,}/{_n_full_users:,} users"
        )
    else:
        effective_target = args.target_interactions
        subset_method_label = (
            f"target_interactions={args.target_interactions:,} (fixed count)"
        )
    print(f"[COMP] Subset target  : {subset_method_label}")

    # ── STAGE 0a: Compute user semantic embeddings ───────────────────────────
    _section("STAGE 0a: User Semantic Embeddings")
    user_text_emb: Optional[np.ndarray] = None
    if args.target_fraction is None and len(unique_users_full) > 0 and args.sentence_transformer_model:
        user_text_emb = _compute_user_semantic_embeddings(
            data_dir=args.data_dir,
            unique_users=unique_users_full,
            model_name=args.sentence_transformer_model,
            max_reviews_per_user=args.max_reviews_per_user,
            max_chars_per_review=args.max_review_chars,
            batch_size=args.st_batch_size,
            cache_dir=os.path.join(out_dir, "cache"),
            device=device,
        )
    else:
        if args.target_fraction is not None:
            print("[COMP] Skipping user semantic embeddings (top-frequent-users path; no UU scoring needed).")
        else:
            print("[COMP] Skipping semantic embeddings (no unique_users returned or model empty).")

    # ── STAGE 0b: Curvature-based compressible user selection ─────────────────
    _section("STAGE 0b: Selecting Compressible User Subset")
    if args.full_dataset:
        print("[COMP] --full_dataset set: skipping subset selection, using full k-core dataset.")
        # Remap IDs to contiguous 0-based integers (same as subset path)
        ratings_full = ratings_full.copy()
        uniq_u = np.sort(ratings_full["UserID"].unique())
        uniq_i = np.sort(ratings_full["BusinessID"].unique())
        ratings_full["UserID"] = np.searchsorted(uniq_u, ratings_full["UserID"].to_numpy())
        ratings_full["BusinessID"] = np.searchsorted(uniq_i, ratings_full["BusinessID"].to_numpy())
        ratings_df = ratings_full.reset_index(drop=True)
        num_users = int(len(uniq_u))
        num_items = int(len(uniq_i))
        del ratings_full
        selection_strategy = "full_dataset"
        unique_items_subset = unique_items_full  # all items retained
    elif args.target_fraction is not None:
        # Top-frequent-users path: select top X% of users by interaction count
        ratings_df, num_users, num_items, sub_item_ids = _select_top_frequent_users(
            ratings_full,
            target_fraction=args.target_fraction,
            seed=args.seed,
        )
        del ratings_full
        selection_strategy = "top_frequent_users_fraction"
        if len(unique_items_full) > 0:
            unique_items_subset = unique_items_full[sub_item_ids]
        else:
            unique_items_subset = np.array([], dtype=object)
    else:
        # Curvature-based path: score users by Forman-Ricci / cosine curvature
        ratings_df, num_users, num_items, sub_item_ids = _select_compressible_users(
            ratings_full,
            num_users=yelp["num_users"],
            num_items=yelp["num_items"],
            target_interactions=effective_target,
            min_shared=args.min_shared,
            score_percentile=args.score_percentile,
            seed=args.seed,
            use_cosine_sim=args.use_cosine_sim,
            hem_matching=args.hem_matching,
            implicit_threshold=args.implicit_threshold,
            user_text_emb=user_text_emb,
            semantic_alpha=args.semantic_alpha,
        )
        del ratings_full
        _curv_tag = "cosine_sim" if args.use_cosine_sim else "forman_ricci"
        _hem_tag = "_hem" if args.hem_matching else ""
        selection_strategy = f"curvature_{_curv_tag}{_hem_tag}_score_selection"
        if len(unique_items_full) > 0:
            unique_items_subset = unique_items_full[sub_item_ids]
        else:
            unique_items_subset = np.array([], dtype=object)

    n_interactions = len(ratings_df)
    sparsity = 1.0 - n_interactions / max(num_users * num_items, 1)
    print(
        f"[COMP] {'Full' if args.full_dataset else 'Subset'}: "
        f"{num_users:,} users | {num_items:,} items | "
        f"{n_interactions:,} interactions | sparsity={sparsity:.4%}"
    )

    dataset_summary: Dict[str, Any] = {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": n_interactions,
        "sparsity": round(sparsity, 6),
        "implicit_threshold": args.implicit_threshold,
        "min_interactions": args.min_interactions,
        "min_shared": args.min_shared,
        "score_percentile": args.score_percentile,
        "use_cosine_sim": args.use_cosine_sim,
        "hem_matching": args.hem_matching,
        "full_dataset": args.full_dataset,
        "selection_strategy": selection_strategy,
        "subset_method_label": subset_method_label,
        "target_fraction": args.target_fraction,
        "effective_target_interactions": effective_target,
        "gsp_curvature_mode": args.curvature_mode,
        "load_time_s": round(load_time, 3),
    }

    # ── Train/test split (LOO + 99-negative evaluation) ───────────────────────
    _section("STAGE 0c: Leave-One-Out Split")
    train_df, test_df = _split_leave_one_out(
        ratings_df, threshold=args.implicit_threshold, seed=args.seed
    )
    seen_train = _build_seen(train_df)
    test_pos = _build_positives(test_df, args.implicit_threshold)
    train_mean = float(train_df["Rating"].mean())

    dataset_summary.update({
        "train_interactions": len(train_df),
        "test_interactions": len(test_df),
        "test_users_with_positives": len(test_pos),
        "split_protocol": "leave-one-out",
        "eval_protocol": f"LOO + {args.eval_negatives} sampled negatives",
    })
    print(
        f"[COMP] LOO split : train={len(train_df):,} | test={len(test_df):,} | "
        f"test-positive users={len(test_pos):,} | eval vs {args.eval_negatives} negatives"
    )
    _write_json(os.path.join(out_dir, "dataset_stats.json"), dataset_summary)

    # ── EXPERIMENT CONFIGURATION SUMMARY ─────────────────────────────────────
    _section("EXPERIMENT CONFIGURATION")
    _n_pos_train = int((train_df["Rating"] >= args.implicit_threshold).sum())
    _steps_per_epoch = max(1, math.ceil(_n_pos_train / args.batch_size))
    _total_opt_steps = args.epochs * _steps_per_epoch
    print(f"[COMP] \u250c\u2500 Dataset \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print(f"[COMP] \u2502  Filtered interaction count        : {n_interactions:>14,}")
    print(f"[COMP] \u2502  Positive train interactions      : {_n_pos_train:>14,}")
    print(f"[COMP] \u2502  Train / Test split               : {len(train_df):>14,} / {len(test_df):,}")
    print(f"[COMP] \u251c\u2500 Training \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print(f"[COMP] \u2502  Batch size                        : {args.batch_size:>14,}")
    print(f"[COMP] \u2502  Epochs                            : {args.epochs:>14,}")
    print(f"[COMP] \u2502  Steps / epoch  ceil({_n_pos_train:,}/{args.batch_size:,})  : {_steps_per_epoch:>14,}")
    print(f"[COMP] \u2502  Total optimisation steps          : {_total_opt_steps:>14,}  ({args.epochs} epochs \u00d7 {_steps_per_epoch} steps)")
    print(f"[COMP] \u2502  LR scheduler                      :  ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-5)")
    print(f"[COMP] \u251c\u2500 Curvature \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print(f"[COMP] \u2502  GSP curvature mode (preprocessing) : {args.curvature_mode:>14s}")
    print(f"[COMP] \u2502  Cosine normalisation (selection)  : {str(args.use_cosine_sim):>14s}")
    print(f"[COMP] \u2502  HEM matching (selection)          : {str(args.hem_matching):>14s}")
    print(f"[COMP] \u251c\u2500 Subset Selection \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print(f"[COMP] \u2502  Method label                      :  {selection_strategy}")
    print(f"[COMP] \u2502  Target specification              :  {subset_method_label}")
    print(f"[COMP] \u2502  Final interaction count           : {n_interactions:>14,}")
    print(f"[COMP] \u2502  Score percentile threshold        : {args.score_percentile:>14.1f}")
    print(f"[COMP] \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")

    # ── STAGE 1: Bipartite graph stats ───────────────────────────────────────
    _section("STAGE 1: Bipartite Graph")
    graph_stats = compute_bipartite_graph_stats(train_df, num_users, num_items)
    dataset_summary.update(graph_stats)
    print(
        f"[COMP] Graph: {graph_stats['num_nodes']:,} nodes | "
        f"{graph_stats['num_edges']:,} edges | "
        f"avg-deg={graph_stats['avg_degree']:.2f} | "
        f"density={graph_stats['density']:.2e} | "
        f"components={graph_stats['num_components']:,} | "
        f"mem={graph_stats['graph_memory_MB']:.1f}MB"
    )

    # ── STAGE 2b: Semantic Feature Extraction ─────────────────────────────────
    _section("STAGE 2b: Semantic Feature Extraction")
    sem_feat_matrix: Optional[np.ndarray] = None
    sem_feat_names: List[str] = []
    _sem_dir = os.path.join(out_dir, "features")
    _sem_npy = os.path.join(_sem_dir, "semantic_features.npy")
    if len(unique_items_subset) > 0:
        # Validate cache shape matches current num_items before re-using
        if os.path.exists(_sem_npy):
            _cache_feat, _cache_names = load_semantic_features(_sem_dir)
            if _cache_feat is not None and _cache_feat.shape[0] == num_items:
                sem_feat_matrix = _cache_feat
                sem_feat_names = _cache_names
                print(
                    f"[COMP] Loaded semantic features from cache: "
                    f"shape={sem_feat_matrix.shape} | {len(sem_feat_names)} features"
                )
            else:
                print("[COMP] Cache shape mismatch — re-extracting semantic features.")
        if sem_feat_matrix is None:
            sem_cfg = SemanticFeatureConfig(
                tfidf_dim=args.tfidf_dim,
                max_review_chars=400,
                max_reviews_per_item=50,
                include_photo_features=True,
            )
            try:
                sem_feat_matrix, sem_feat_names = extract_semantic_features(
                    data_dir=args.data_dir,
                    unique_items=unique_items_subset,
                    output_dir=_sem_dir,
                    config=sem_cfg,
                )
                print(
                    f"[COMP] Semantic features extracted: shape={sem_feat_matrix.shape} | "
                    f"{len(sem_feat_names)} features "
                    f"({sem_feat_matrix.shape[1] - args.tfidf_dim} hand-crafted + "
                    f"{args.tfidf_dim} SVD topic dims)"
                )
            except Exception as _sem_exc:
                print(f"[COMP] WARNING: Semantic feature extraction failed: {_sem_exc}")
                sem_feat_matrix = None
    else:
        print("[COMP] No item IDs available; semantic features skipped.")

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
        curvature_mode=args.curvature_mode,
        clustering_method=args.clustering_method,
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

    compression_ratio = gsp_stats["compression_ratio"]
    singleton_ratio = gsp_stats.get("singleton_ratio", gsp_stats.get("singleton_fraction", 0))
    print(
        f"[COMP] GSP done in {gsp_elapsed:.1f}s | "
        f"compression={compression_ratio*100:.1f}% | "
        f"super-nodes={num_super:,} | "
        f"avg_cluster={gsp_stats['avg_cluster_size']:.2f} | "
        f"singleton={singleton_ratio*100:.1f}%"
    )

    # Compressibility verdict
    if compression_ratio >= 0.10:
        print(f"[COMP] *** Compressibility: GOOD ({compression_ratio*100:.1f}% reduction) — "
              f"speedup expected ***")
    elif compression_ratio >= 0.03:
        print(f"[COMP] *** Compressibility: MODERATE ({compression_ratio*100:.1f}%) ***")
    else:
        print(f"[COMP] WARNING: Low compression ({compression_ratio*100:.1f}%) — "
              f"try lowering --score_percentile or --min_shared")

    gsp_paper_stats = {
        "num_users": num_users,
        "num_super_nodes": num_super,
        "compression_ratio_pct": round(compression_ratio * 100, 2),
        "avg_cluster_size": round(gsp_stats["avg_cluster_size"], 3),
        "singleton_ratio_pct": round(singleton_ratio * 100, 2),
        "largest_cluster": gsp_stats.get("largest_cluster_size", gsp_stats.get("max_cluster_size", "?")),
        "edge_retention_pct": round(gsp_stats.get("edge_retention_ratio", gsp_stats.get("uu_hc_fraction", 0)) * 100, 2),
        "uu_edges_original": gsp_stats.get("uu_edges_before_shared", gsp_stats.get("uu_edges_all", 0)),
        "uu_edges_after_filter": gsp_stats.get("uu_edges_all", 0),
        "uu_edges_hc": gsp_stats.get("uu_edges_hc", 0),
        "uu_edges_pruned": gsp_stats.get("uu_edges_pruned", 0),
        "gsp_preprocessing_time_s": round(gsp_elapsed, 3),
        "gsp_curvature_mode": args.curvature_mode,
        "selection_strategy": selection_strategy,
        "score_percentile": args.score_percentile,
    }

    # ── Build edge indices ─────────────────────────────────────────────────────
    _section("Building Graph Edge Indices")

    base_agg = (
        train_df.groupby(["UserID", "BusinessID"], as_index=False)
        .agg(rating=("Rating", "mean"))
        .rename(columns={"UserID": "u_idx", "BusinessID": "i_idx"})
        .astype({"u_idx": np.int64, "i_idx": np.int64})
    )
    edge_index_base = _build_edge_index(base_agg, "u_idx", "i_idx", num_users)

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

    nodes_orig = num_users + num_items
    nodes_gsp = num_super + num_items
    edges_orig = int(base_agg.shape[0])
    edges_gsp = int(coarsened.shape[0])
    edge_reduction_pct = round((1 - edges_gsp / max(edges_orig, 1)) * 100, 2)

    gsp_paper_stats.update({
        "bipartite_nodes_original": nodes_orig,
        "bipartite_nodes_gsp": nodes_gsp,
        "bipartite_edges_original": edges_orig,
        "bipartite_edges_gsp": edges_gsp,
        "bipartite_edge_reduction_pct": edge_reduction_pct,
    })

    print(f"[COMP] Baseline graph : {nodes_orig:,} nodes | {edge_index_base.shape[1]:,} edge-slots")
    print(
        f"[COMP] GSP graph      : {nodes_gsp:,} nodes | {edge_index_gsp.shape[1]:,} edge-slots"
        f"  ({edge_reduction_pct:.1f}% reduction)"
    )

    _write_json(os.path.join(out_dir, "gsp_stats.json"), gsp_paper_stats)

    # Shared model config
    model_cfg = ModelConfig(
        emb_dim=args.emb_dim,
        hidden_dim=args.emb_dim * 2,
        out_dim=args.emb_dim,
        num_layers=args.num_layers,
        heads=args.gat_heads,   # 8-head GAT for richer attention on semantically-pruned edges
        dropout=0.1,
    )

    base_train_y = torch.tensor(base_agg["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)
    gsp_train_y = torch.tensor(coarsened["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)
    gsp_train_super = torch.tensor(coarsened["super_idx"].to_numpy(dtype=np.int64), dtype=torch.long)
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

        # ── Baseline ──────────────────────────────────────────────────────────
        print(f"[COMP] {model_name} | Baseline  ({nodes_orig:,} nodes, {edges_orig:,} edges)")
        try:
            base_model = get_model(model_name, nodes_orig, model_cfg)
        except Exception as exc:
            print(f"[COMP] WARNING: cannot build '{model_name}': {exc}  Skipping.")
            continue

        # Warm-start item embeddings from semantic features
        if sem_feat_matrix is not None and sem_feat_matrix.shape[0] == num_items:
            _base_emb_dim = model_cfg.out_dim
            _init_item_embeddings_from_features(
                base_model, sem_feat_matrix, num_users, _base_emb_dim, alpha=0.30
            )
            print(f"[COMP] {model_name} baseline: item embeddings warm-started from semantic features")

        train_cfg_base = TrainConfig(
            epochs=args.epochs, lr=args.lr, weight_decay=1e-5,
            batch_size=args.batch_size, neg_ratio=args.neg_ratio,
            emb_l2_weight=1e-5, seed=args.seed, use_amp=(not args.no_amp),
            checkpoint_dir=os.path.join(out_dir, "checkpoints"),
            save_epoch_checkpoints=False,
            metrics_jsonl_path=os.path.join(out_dir, f"training_metrics_{model_name}_baseline.jsonl"),
            training_log_path=os.path.join(out_dir, f"training_log_{model_name}_baseline.txt"),
            device=device,
            early_stopping_patience=args.early_stopping_patience,
        )

        reset_gpu_peak_memory()
        t_base_start = time.perf_counter()
        cpu_base_start = _proc_cpu_time_s()
        _ = train_model(
            base_model,
            edge_index=edge_index_base,
            train_user_nodes=torch.tensor(base_agg["u_idx"].to_numpy(dtype=np.int64), dtype=torch.long),
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
        if sem_feat_matrix is not None and sem_feat_matrix.shape[0] == num_items:
            ie_base = _augment_with_features(ie_base, sem_feat_matrix)
        rank_base = _eval_multi_k(ue_base, ie_base, test_pos, seen_train, args.eval_negatives, args.seed)
        reg_base = _compute_rmse_mae(test_df, ue_base, ie_base, train_mean)

        _rank_str_base = "  ".join(
            f"NDCG@{k}={rank_base.get(f'NDCG@{k}', 0):.4f}" for k in _EVAL_KS
        )
        print(
            f"[COMP] {model_name} BASELINE  "
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

        # ── GSP reduced-graph training ─────────────────────────────────────────
        print(f"\n[COMP] {model_name} | GSP  ({nodes_gsp:,} nodes, {edges_gsp:,} edges)")
        try:
            gsp_model = get_model(model_name, nodes_gsp, model_cfg)
        except Exception as exc:
            print(f"[COMP] WARNING: cannot build GSP '{model_name}': {exc}  Skipping GSP.")
            continue

        # Warm-start item embeddings from semantic features
        if sem_feat_matrix is not None and sem_feat_matrix.shape[0] == num_items:
            _gsp_emb_dim = model_cfg.out_dim
            _init_item_embeddings_from_features(
                gsp_model, sem_feat_matrix, num_super, _gsp_emb_dim, alpha=0.30
            )
            print(f"[COMP] {model_name} GSP: item embeddings warm-started from semantic features")

        train_cfg_gsp = TrainConfig(
            epochs=args.epochs, lr=args.lr, weight_decay=1e-5,
            batch_size=args.batch_size, neg_ratio=args.neg_ratio,
            emb_l2_weight=1e-5, seed=args.seed, use_amp=(not args.no_amp),
            checkpoint_dir=os.path.join(out_dir, "checkpoints"),
            save_epoch_checkpoints=False,
            metrics_jsonl_path=os.path.join(out_dir, f"training_metrics_{model_name}_gsp.jsonl"),
            training_log_path=os.path.join(out_dir, f"training_log_{model_name}_gsp.txt"),
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

        # Inference + projection
        t_infer_gsp_start = time.perf_counter()
        z_gsp = _infer(gsp_model, edge_index_gsp, device)
        t_infer_gsp_raw = time.perf_counter() - t_infer_gsp_start
        H_super = z_gsp[:num_super].astype(np.float32)
        H_final, proj_t = project_embeddings(H_super, C)
        t_infer_gsp = t_infer_gsp_raw + proj_t

        ie_gsp = _l2_norm(z_gsp[num_super:])
        ue_proj = _l2_norm(H_final)
        if sem_feat_matrix is not None and sem_feat_matrix.shape[0] == num_items:
            ie_gsp = _augment_with_features(ie_gsp, sem_feat_matrix)

        rank_proj = _eval_multi_k(ue_proj, ie_gsp, test_pos, seen_train, args.eval_negatives, args.seed)
        reg_proj = _compute_rmse_mae(test_df, ue_proj, ie_gsp, train_mean)

        _rank_str_gsp = "  ".join(
            f"NDCG@{k}={rank_proj.get(f'NDCG@{k}', 0):.4f}" for k in _EVAL_KS
        )
        print(
            f"[COMP] {model_name} GSP+PROJ  "
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
        }
        metrics_rows.append(row_gsp)

        # Speedup record
        speedup = t_base / max(t_gsp_train, 1e-9)
        infer_speedup = t_infer_base / max(t_infer_gsp, 1e-9)
        speedup_rows.append({
            "model": model_name,
            "training_time_baseline_s": round(t_base, 3),
            "training_time_gsp_s": round(t_gsp_train, 3),
            "speedup_factor": round(speedup, 4),
            "inference_time_baseline_s": round(t_infer_base, 4),
            "inference_time_gsp_s": round(t_infer_gsp, 4),
            "inference_forward_gsp_s": round(t_infer_gsp_raw, 4),
            "inference_projection_s": round(proj_t, 4),
            "inference_speedup": round(infer_speedup, 4),
            "gsp_preprocessing_s": round(gsp_elapsed, 3),
            "net_time_saved_s": round(t_base - t_gsp_train - gsp_elapsed, 3),
            "gpu_baseline_MB": round(gpu_base_mb, 1),
            "gpu_gsp_MB": round(gpu_gsp_mb, 1),
            "gpu_reduction_pct": round((1 - gpu_gsp_mb / max(gpu_base_mb, 1)) * 100, 2),
            "cpu_time_baseline_s": round(cpu_base_s, 3),
            "cpu_time_gsp_s": round(cpu_gsp_s, 3),
            "cpu_efficiency_baseline_pct": round(cpu_base_s / max(t_base, 1e-9) * 100, 1),
            "cpu_efficiency_gsp_pct": round(cpu_gsp_s / max(t_gsp_train, 1e-9) * 100, 1),
            "compression_ratio_pct": round(compression_ratio * 100, 2),
            "singleton_ratio_pct": round(singleton_ratio * 100, 2),
            **{f"Precision@{k}_baseline": round(rank_base.get(f"Precision@{k}", 0), 4) for k in _EVAL_KS},
            **{f"Precision@{k}_gsp": round(rank_proj.get(f"Precision@{k}", 0), 4) for k in _EVAL_KS},
            **{f"NDCG@{k}_baseline": round(rank_base.get(f"NDCG@{k}", 0), 4) for k in _EVAL_KS},
            **{f"NDCG@{k}_gsp": round(rank_proj.get(f"NDCG@{k}", 0), 4) for k in _EVAL_KS},
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
                gsp_summary=rank_proj,
                curvature_mode=args.curvature_mode,
                fraction=args.target_fraction if args.target_fraction is not None else 1.0,
                min_shared=args.min_shared,
                dataset_name="yelp",
            )
        except Exception as _analytics_exc:
            print(f"[COMP] WARNING: analytics failed for {model_name}: {_analytics_exc}")

        print(
            f"\n[COMP] {model_name.upper()} SPEEDUP: {speedup:.2f}x  "
            f"(baseline={t_base:.1f}s → gsp={t_gsp_train:.1f}s)"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Final output
    # ─────────────────────────────────────────────────────────────────────────
    _section("RESULTS SUMMARY")
    _print_paper_table(metrics_rows)

    _save_results_csv(metrics_rows, os.path.join(out_dir, "results_table.csv"))
    _save_results_csv(speedup_rows, os.path.join(out_dir, "speedup_results.csv"))
    _save_gsp_stats_csv(gsp_paper_stats, dataset_summary, os.path.join(out_dir, "gsp_stats.csv"))

    total_wall = time.perf_counter() - t_wall
    summary = {
        "dataset": dataset_summary,
        "gsp": gsp_paper_stats,
        "metrics": metrics_rows,
        "speedup": speedup_rows,
        "total_wall_time_s": round(total_wall, 2),
    }
    _write_json(os.path.join(out_dir, "full_results.json"), summary)

    print(f"\n[COMP] Total wall-clock time : {total_wall/60:.1f} min")
    print(f"[COMP] Results written to    : {out_dir}/")
    print(f"[COMP]   results_table.csv")
    print(f"[COMP]   speedup_results.csv")
    print(f"[COMP]   gsp_stats.csv")
    print(f"[COMP]   full_results.json")

    print("\n--- GSP Compression Summary (for paper) ---")
    for k, v in gsp_paper_stats.items():
        print(f"  {k:<50}: {v}")

    if speedup_rows:
        print("\n--- Speedup Summary ---")
        for r in speedup_rows:
            print(
                f"  {r['model']:<15}: {r['speedup_factor']:.2f}x training speedup  "
                f"({r['compression_ratio_pct']:.1f}% compression, "
                f"{r['singleton_ratio_pct']:.1f}% singletons)"
            )


if __name__ == "__main__":
    main()
