"""
GSP Compression Structural Analyzer

Tracks and analyzes the structural impact of GSP compression:
  - Nodes removed (user clustering / super-nodes)
  - Edges removed
  - Singleton ratio
  - Embedding drift (L2 distance before vs after GSP)
  - Recommendation preservation (overlap@K)
  - Diversity change
  - Popularity bias shift
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
class GraphStats:
    num_original_users: int
    num_super_nodes: int
    num_original_edges: int       # edges in baseline user-item graph
    num_gsp_edges: int            # edges in GSP-compressed graph
    num_uu_edges_before: int = 0  # user-user edges before compression
    num_uu_edges_after: int = 0   # high-curvature edges kept
    singleton_count: int = 0      # super-nodes with only 1 original user
    compression_ratio: float = 0.0
    singleton_ratio: float = 0.0
    edge_reduction_ratio: float = 0.0

    def compute_derived(self) -> None:
        self.compression_ratio = (
            1.0 - self.num_super_nodes / max(self.num_original_users, 1)
        )
        self.singleton_ratio = (
            self.singleton_count / max(self.num_super_nodes, 1)
        )
        self.edge_reduction_ratio = (
            1.0 - self.num_gsp_edges / max(self.num_original_edges, 1)
        )


@dataclass
class UserCompressionProfile:
    user_id: int
    super_node_id: int
    cluster_size: int              # number of original users in this super-node
    embedding_drift: float = 0.0  # L2 distance of user emb before vs after GSP
    cosine_similarity: float = 1.0  # cosine sim between baseline and GSP embeddings


@dataclass
class RecommendationPreservation:
    k: int
    mean_overlap: float
    std_overlap: float
    mean_diversity_baseline: float
    mean_diversity_gsp: float
    diversity_change: float        # gsp - baseline (positive = more diverse)
    mean_popularity_baseline: float
    mean_popularity_gsp: float
    popularity_bias_shift: float   # positive = GSP recs are more popular


# ─────────────────────────────────────────────────────────────────────────────
# Compression Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class CompressionAnalyzer:
    """
    Analyzes the structural and recommendation impact of GSP compression.

    Usage
    -----
    ::
        analyzer = CompressionAnalyzer(output_dir="outputs/compression")
        analyzer.compute_graph_stats(
            num_users, num_super, num_uu_edges, num_hc_edges,
            base_edge_index, gsp_edge_index,
        )
        analyzer.compute_embedding_drift(
            user_to_super, baseline_user_emb, gsp_super_emb
        )
        analyzer.compute_recommendation_preservation(
            baseline_recs, gsp_recs, item_popularity, k=10
        )
        analyzer.save()
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._graph_stats: Optional[GraphStats] = None
        self._user_profiles: List[UserCompressionProfile] = []
        self._preservation_results: List[RecommendationPreservation] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Graph-level statistics
    # ─────────────────────────────────────────────────────────────────────────

    def compute_graph_stats(
        self,
        num_original_users: int,
        num_super_nodes: int,
        num_uu_edges_before: int,
        num_uu_edges_after: int,
        base_edge_index_size: int,    # number of (directed) edges in baseline bipartite graph
        gsp_edge_index_size: int,     # number of (directed) edges in GSP bipartite graph
        user_to_super: np.ndarray,
    ) -> GraphStats:
        """Compute compression statistics from the GSP output."""
        # Count singletons: super-nodes that contain exactly 1 original user
        _, cluster_counts = np.unique(user_to_super, return_counts=True)
        singleton_count = int(np.sum(cluster_counts == 1))

        stats = GraphStats(
            num_original_users=num_original_users,
            num_super_nodes=num_super_nodes,
            num_original_edges=base_edge_index_size,
            num_gsp_edges=gsp_edge_index_size,
            num_uu_edges_before=num_uu_edges_before,
            num_uu_edges_after=num_uu_edges_after,
            singleton_count=singleton_count,
        )
        stats.compute_derived()
        self._graph_stats = stats
        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # Per-user embedding drift
    # ─────────────────────────────────────────────────────────────────────────

    def compute_embedding_drift(
        self,
        user_to_super: np.ndarray,
        baseline_user_emb: np.ndarray,      # (U, D) – from baseline model
        gsp_super_emb: np.ndarray,           # (S, D) – from GSP model (super-nodes)
        max_users: int = 5000,
    ) -> List[UserCompressionProfile]:
        """
        Compute per-user embedding drift: L2 distance between
        baseline embedding and the corresponding GSP super-node embedding.

        Parameters
        ----------
        user_to_super         (U,) mapping of original users to super-nodes
        baseline_user_emb     (U, D) embeddings from baseline model
        gsp_super_emb         (S, D) super-node embeddings from GSP model
        max_users             cap for large datasets
        """
        num_users = min(len(user_to_super), baseline_user_emb.shape[0], max_users)
        profiles: List[UserCompressionProfile] = []

        # Cluster sizes
        unique_supers, cluster_counts = np.unique(user_to_super, return_counts=True)
        cluster_size_map = dict(zip(unique_supers.tolist(), cluster_counts.tolist()))

        for uid in range(num_users):
            super_id = int(user_to_super[uid])
            if super_id >= gsp_super_emb.shape[0]:
                continue

            base_vec = baseline_user_emb[uid].astype(np.float64)
            gsp_vec  = gsp_super_emb[super_id].astype(np.float64)

            # L2 drift
            drift = float(np.linalg.norm(base_vec - gsp_vec))

            # Cosine similarity
            bn = np.linalg.norm(base_vec)
            gn = np.linalg.norm(gsp_vec)
            cos_sim = (
                float(np.dot(base_vec, gsp_vec) / (bn * gn))
                if bn > 1e-9 and gn > 1e-9
                else 0.0
            )

            profiles.append(UserCompressionProfile(
                user_id=uid,
                super_node_id=super_id,
                cluster_size=int(cluster_size_map.get(super_id, 1)),
                embedding_drift=drift,
                cosine_similarity=cos_sim,
            ))

        self._user_profiles = profiles
        return profiles

    def get_drift_summary(self) -> Dict:
        """Aggregate embedding drift statistics."""
        if not self._user_profiles:
            return {}
        drifts = np.array([p.embedding_drift for p in self._user_profiles])
        cos_sims = np.array([p.cosine_similarity for p in self._user_profiles])
        cluster_sizes = np.array([p.cluster_size for p in self._user_profiles])
        return {
            "mean_embedding_drift": float(np.mean(drifts)),
            "std_embedding_drift":  float(np.std(drifts)),
            "max_embedding_drift":  float(np.max(drifts)),
            "median_embedding_drift": float(np.median(drifts)),
            "mean_cosine_similarity": float(np.mean(cos_sims)),
            "std_cosine_similarity":  float(np.std(cos_sims)),
            "mean_cluster_size":  float(np.mean(cluster_sizes)),
            "max_cluster_size":   int(np.max(cluster_sizes)),
            "pct_singleton":      float(np.mean(cluster_sizes == 1) * 100),
            "num_users_analyzed": len(self._user_profiles),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Recommendation preservation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_recommendation_preservation(
        self,
        baseline_recs: Dict[int, List[int]],    # user_id -> ranked item list
        gsp_recs: Dict[int, List[int]],
        item_popularity: np.ndarray,             # (I,) popularity counts
        k: int = 10,
    ) -> RecommendationPreservation:
        """
        Compute overlap, diversity, and popularity bias between
        baseline and GSP recommendations.

        Diversity metric: intra-list diversity (ILD) using item popularity
        as a proxy for item coverage.
        """
        overlaps:              List[float] = []
        diversity_base_list:   List[float] = []
        diversity_gsp_list:    List[float] = []
        pop_base_list:         List[float] = []
        pop_gsp_list:          List[float] = []

        for uid in set(baseline_recs) & set(gsp_recs):
            base_top = baseline_recs[uid][:k]
            gsp_top  = gsp_recs[uid][:k]

            # Overlap@K
            overlap = len(set(base_top) & set(gsp_top)) / max(k, 1)
            overlaps.append(overlap)

            # Popularity (mean popularity of recommended items)
            base_pops = [int(item_popularity[i]) for i in base_top if i < len(item_popularity)]
            gsp_pops  = [int(item_popularity[i]) for i in gsp_top  if i < len(item_popularity)]

            if base_pops:
                pop_base_list.append(float(np.mean(base_pops)))
            if gsp_pops:
                pop_gsp_list.append(float(np.mean(gsp_pops)))

            # Diversity: normalized range of popularity (proxy for coverage spread)
            if len(base_pops) > 1:
                div_base = float(np.std(base_pops) / (np.mean(base_pops) + 1e-9))
                diversity_base_list.append(div_base)
            if len(gsp_pops) > 1:
                div_gsp = float(np.std(gsp_pops) / (np.mean(gsp_pops) + 1e-9))
                diversity_gsp_list.append(div_gsp)

        mean_div_base = float(np.mean(diversity_base_list)) if diversity_base_list else 0.0
        mean_div_gsp  = float(np.mean(diversity_gsp_list))  if diversity_gsp_list  else 0.0
        mean_pop_base = float(np.mean(pop_base_list)) if pop_base_list else 0.0
        mean_pop_gsp  = float(np.mean(pop_gsp_list))  if pop_gsp_list  else 0.0

        result = RecommendationPreservation(
            k=k,
            mean_overlap=float(np.mean(overlaps)) if overlaps else 0.0,
            std_overlap=float(np.std(overlaps))  if overlaps else 0.0,
            mean_diversity_baseline=mean_div_base,
            mean_diversity_gsp=mean_div_gsp,
            diversity_change=mean_div_gsp - mean_div_base,
            mean_popularity_baseline=mean_pop_base,
            mean_popularity_gsp=mean_pop_gsp,
            popularity_bias_shift=mean_pop_gsp - mean_pop_base,
        )
        self._preservation_results.append(result)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # I/O
    # ─────────────────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist all compression analysis to JSON and CSV."""
        # --- Graph stats ---
        if self._graph_stats:
            json_path = os.path.join(self.output_dir, "graph_stats.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(asdict(self._graph_stats), fh, indent=2)

        # --- Embedding drift per user ---
        if self._user_profiles:
            csv_path = os.path.join(self.output_dir, "embedding_drift.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "user_id", "super_node_id", "cluster_size",
                    "embedding_drift", "cosine_similarity",
                ])
                for p in self._user_profiles:
                    writer.writerow([
                        p.user_id, p.super_node_id, p.cluster_size,
                        f"{p.embedding_drift:.6f}", f"{p.cosine_similarity:.6f}",
                    ])

            # Drift summary
            summary = self.get_drift_summary()
            json_path = os.path.join(self.output_dir, "embedding_drift_summary.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)

        # --- Recommendation preservation ---
        if self._preservation_results:
            csv_path = os.path.join(self.output_dir, "recommendation_preservation.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "k", "mean_overlap", "std_overlap",
                    "mean_diversity_baseline", "mean_diversity_gsp", "diversity_change",
                    "mean_popularity_baseline", "mean_popularity_gsp", "popularity_bias_shift",
                ])
                for r in self._preservation_results:
                    writer.writerow([
                        r.k, f"{r.mean_overlap:.4f}", f"{r.std_overlap:.4f}",
                        f"{r.mean_diversity_baseline:.4f}", f"{r.mean_diversity_gsp:.4f}",
                        f"{r.diversity_change:.4f}",
                        f"{r.mean_popularity_baseline:.2f}", f"{r.mean_popularity_gsp:.2f}",
                        f"{r.popularity_bias_shift:.2f}",
                    ])
