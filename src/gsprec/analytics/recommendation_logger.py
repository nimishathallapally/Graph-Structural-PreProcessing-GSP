"""
Recommendation Logger – Inference Tracking

Stores top-K recommended items (K = 10, 20, 50) for every test user.
Outputs structured CSV/JSON with:
  - user_id, model, run_type, curvature_mode, fraction, min_shared
  - recommended_items (ranked list), scores, ground_truth
  - overlap@K, rank_shifts, new/removed items vs baseline
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserRecommendation:
    user_id: int
    model: str
    run_type: str                          # "baseline" | "gsp"
    curvature_mode: str = ""               # "cosine" | "forman_ricci"
    fraction: float = 1.0                  # compression fraction
    min_shared: int = 1
    recommended_items: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    ground_truth: List[int] = field(default_factory=list)


@dataclass
class OverlapResult:
    user_id: int
    model: str
    k: int
    overlap_count: int
    overlap_ratio: float
    rank_shifts: List[Tuple[int, int, int]]   # (item_id, rank_baseline, rank_gsp)
    new_items: List[int]
    removed_items: List[int]


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation Logger
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationLogger:
    """
    Logs top-K recommendations for every test user.

    Usage
    -----
    ::
        logger = RecommendationLogger(output_dir="outputs/recs", top_ks=[10, 20, 50])
        logger.log_recommendations(
            user_emb, item_emb,
            test_positives, seen_positives,
            model_name="lightgcn",
            run_type="baseline",
        )
        logger.save()
    """

    TOP_KS = [10, 20, 50]

    def __init__(
        self,
        output_dir: str,
        top_ks: Optional[List[int]] = None,
        curvature_mode: str = "",
        fraction: float = 1.0,
        min_shared: int = 1,
    ):
        self.output_dir = output_dir
        self.top_ks = top_ks or self.TOP_KS
        self.max_k = max(self.top_ks)
        self.curvature_mode = curvature_mode
        self.fraction = fraction
        self.min_shared = min_shared
        os.makedirs(output_dir, exist_ok=True)

        # Store recs: run_type -> List[UserRecommendation]
        self._records: Dict[str, List[UserRecommendation]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Core logging
    # ─────────────────────────────────────────────────────────────────────────

    def log_recommendations(
        self,
        user_emb: np.ndarray,
        item_emb: np.ndarray,
        test_positives: Dict[int, List[int]],
        seen_positives: Dict[int, Set[int]],
        model_name: str,
        run_type: str,
        user_to_super: Optional[np.ndarray] = None,
    ) -> List[UserRecommendation]:
        """
        Compute and store top-K recommendations for all test users.

        Parameters
        ----------
        user_emb       (U, D) user embeddings (already mapped from super-nodes if GSP)
        item_emb       (I, D) item embeddings
        test_positives user_id -> list of positive item indices (ground truth)
        seen_positives user_id -> set of seen items (train set, to exclude)
        model_name     GNN architecture name
        run_type       "baseline" or "gsp"
        user_to_super  mapping from original user -> super-node (GSP only)
        """
        recs: List[UserRecommendation] = []
        num_items = item_emb.shape[0]

        for user_id, pos_items in test_positives.items():
            uid = int(user_id)

            # Map GSP super-node embedding back to original user
            if user_to_super is not None and uid < len(user_to_super):
                u_emb = user_emb[user_to_super[uid]]
            elif uid < user_emb.shape[0]:
                u_emb = user_emb[uid]
            else:
                continue

            # Exclude seen items
            seen: Set[int] = set(seen_positives.get(uid, set()))

            # Score all items (vectorised)
            all_scores = item_emb @ u_emb  # (I,)

            # Mask seen items
            seen_arr = np.array(list(seen), dtype=np.int64)
            if seen_arr.size > 0:
                all_scores[seen_arr] = -np.inf

            # Top-max_k items
            top_indices = np.argpartition(all_scores, -self.max_k)[-self.max_k:]
            top_indices = top_indices[np.argsort(all_scores[top_indices])[::-1]]
            top_scores = all_scores[top_indices].tolist()

            rec = UserRecommendation(
                user_id=uid,
                model=model_name,
                run_type=run_type,
                curvature_mode=self.curvature_mode,
                fraction=self.fraction,
                min_shared=self.min_shared,
                recommended_items=top_indices.tolist(),
                scores=top_scores,
                ground_truth=pos_items,
            )
            recs.append(rec)

        key = f"{model_name}_{run_type}"
        self._records[key] = recs
        return recs

    # ─────────────────────────────────────────────────────────────────────────
    # Comparison: baseline vs GSP
    # ─────────────────────────────────────────────────────────────────────────

    def compute_overlap(
        self,
        model_name: str,
        k: int,
    ) -> List[OverlapResult]:
        """
        Compare baseline vs GSP recommendations for a given model at top-K.

        Returns per-user overlap@K, rank shifts, new/removed items.
        """
        if k not in self.top_ks:
            raise ValueError(f"k={k} not in top_ks={self.top_ks}")

        base_key = f"{model_name}_baseline"
        gsp_key = f"{model_name}_gsp"

        if base_key not in self._records or gsp_key not in self._records:
            return []

        # Build lookup: user_id -> rec
        base_map = {r.user_id: r for r in self._records[base_key]}
        gsp_map  = {r.user_id: r for r in self._records[gsp_key]}

        results: List[OverlapResult] = []

        for uid in set(base_map) & set(gsp_map):
            base_rec = base_map[uid]
            gsp_rec  = gsp_map[uid]

            base_top_k = base_rec.recommended_items[:k]
            gsp_top_k  = gsp_rec.recommended_items[:k]

            base_set = set(base_top_k)
            gsp_set  = set(gsp_top_k)

            overlap = base_set & gsp_set
            new_items     = list(gsp_set  - base_set)
            removed_items = list(base_set - gsp_set)

            # Rank shifts for items in both top-K lists
            base_rank = {item: rank for rank, item in enumerate(base_top_k)}
            gsp_rank  = {item: rank for rank, item in enumerate(gsp_top_k)}
            rank_shifts = [
                (item, base_rank[item], gsp_rank[item])
                for item in overlap
                if base_rank[item] != gsp_rank[item]
            ]

            results.append(OverlapResult(
                user_id=uid,
                model=model_name,
                k=k,
                overlap_count=len(overlap),
                overlap_ratio=len(overlap) / max(k, 1),
                rank_shifts=rank_shifts,
                new_items=new_items,
                removed_items=removed_items,
            ))

        return results

    def compute_all_overlaps(
        self, model_name: str
    ) -> Dict[int, List[OverlapResult]]:
        """Compute overlap@K for all configured K values."""
        return {k: self.compute_overlap(model_name, k) for k in self.top_ks}

    # ─────────────────────────────────────────────────────────────────────────
    # I/O
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, model_name: Optional[str] = None) -> None:
        """Save recommendations to CSV and JSON, and overlap stats to CSV."""
        keys = (
            [k for k in self._records if model_name and k.startswith(model_name)]
            if model_name
            else list(self._records.keys())
        )

        for key in keys:
            recs = self._records[key]
            if not recs:
                continue

            # --- JSON (full ranked lists + scores) ---
            json_path = os.path.join(self.output_dir, f"recommendations_{key}.json")
            payload = []
            for r in recs:
                d = asdict(r)
                # Store separate top-K slices
                d["top_k_slices"] = {
                    str(k): r.recommended_items[:k] for k in self.top_ks
                }
                payload.append(d)
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)

            # --- CSV (flattened, one row per K) ---
            csv_path = os.path.join(self.output_dir, f"recommendations_{key}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "user_id", "model", "run_type", "curvature_mode",
                    "fraction", "min_shared", "k",
                    "recommended_items", "ground_truth",
                    "hit", "ndcg",
                ])
                for r in recs:
                    for k in self.top_ks:
                        top_k = r.recommended_items[:k]
                        gt_set = set(r.ground_truth)
                        hit = int(bool(gt_set & set(top_k)))
                        ndcg = _ndcg_at_k(top_k, gt_set, k)
                        writer.writerow([
                            r.user_id, r.model, r.run_type,
                            r.curvature_mode, r.fraction, r.min_shared, k,
                            ";".join(map(str, top_k)),
                            ";".join(map(str, r.ground_truth)),
                            hit, f"{ndcg:.6f}",
                        ])

        # --- Overlap stats ---
        model_names = set()
        for key in keys:
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                model_names.add(parts[0])

        for mname in model_names:
            overlaps = self.compute_all_overlaps(mname)
            if not any(overlaps.values()):
                continue

            overlap_path = os.path.join(self.output_dir, f"overlap_{mname}.csv")
            with open(overlap_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "user_id", "model", "k",
                    "overlap_count", "overlap_ratio",
                    "new_items_count", "removed_items_count",
                    "rank_shifts_count",
                ])
                for k, results in overlaps.items():
                    for ov in results:
                        writer.writerow([
                            ov.user_id, ov.model, ov.k,
                            ov.overlap_count, f"{ov.overlap_ratio:.4f}",
                            len(ov.new_items), len(ov.removed_items),
                            len(ov.rank_shifts),
                        ])

    def get_aggregate_overlap_stats(self, model_name: str) -> Dict[int, Dict]:
        """Return mean/std overlap stats per K."""
        overlaps = self.compute_all_overlaps(model_name)
        stats = {}
        for k, results in overlaps.items():
            if not results:
                stats[k] = {}
                continue
            ratios = np.array([r.overlap_ratio for r in results])
            new_counts = np.array([len(r.new_items) for r in results])
            rm_counts  = np.array([len(r.removed_items) for r in results])
            shift_counts = np.array([len(r.rank_shifts) for r in results])
            stats[k] = {
                "mean_overlap": float(np.mean(ratios)),
                "std_overlap":  float(np.std(ratios)),
                "mean_new_items":     float(np.mean(new_counts)),
                "mean_removed_items": float(np.mean(rm_counts)),
                "mean_rank_shifts":   float(np.mean(shift_counts)),
                "n_users": len(results),
            }
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ndcg_at_k(ranked_items: List[int], relevant_set: Set[int], k: int) -> float:
    """Compute NDCG@K for a single user."""
    if not relevant_set:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in relevant_set:
            dcg += 1.0 / np.log2(rank + 1)
    ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant_set), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0
