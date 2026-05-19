"""
Inference Analyzer – Node/Edge Removal Insights

Analyzes what was removed during GSP compression and whether it was:
  - Noise (weak similarity / low co-interaction) or meaningful signal
  - Beneficial (denser neighborhoods) or lossy (information loss)

Correlates:
  - Compression level vs NDCG change
  - Compression vs recommendation stability (overlap@K)
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
class EdgeRemovalAnalysis:
    """Analysis of a single removed edge (user-user graph)."""
    user_u: int
    user_v: int
    edge_weight: float              # original curvature / similarity score
    common_items: int               # number of co-rated items
    deg_u: int                      # degree of user u
    deg_v: int                      # degree of user v
    removal_reason: str             # "low_curvature" | "low_importance"
    is_weak: bool                   # True if below median edge weight
    is_low_cointeraction: bool      # True if common_items < threshold


@dataclass
class NodeMergeAnalysis:
    """Analysis of a super-node formed by merging users."""
    super_node_id: int
    merged_users: List[int]
    cluster_size: int
    total_items_before: int          # union of items of all merged users
    items_after_merge: int           # items of the super-node post-merging
    overlap_fraction: float          # pairwise item overlap within cluster
    signal_gain: bool                # True if dense overlap → signal improvement
    information_loss: bool           # True if sparse overlap → info lost


@dataclass
class CompressionCorrelation:
    """Correlation between compression level and model performance."""
    compression_ratio: float
    ndcg_baseline: float
    ndcg_gsp: float
    ndcg_change: float               # gsp - baseline
    ndcg_change_pct: float           # % change
    overlap_at_k: float              # mean overlap@K
    stability_score: float           # overlap_at_k as stability proxy
    run_label: str = ""              # identifier for this run (e.g. fraction label)


# ─────────────────────────────────────────────────────────────────────────────
# Inference Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class InferenceAnalyzer:
    """
    Provides structural insights on edge/node removal during GSP compression.

    Usage
    -----
    ::
        analyzer = InferenceAnalyzer(output_dir="outputs/inference")

        # Analyze removed edges
        analyzer.analyze_removed_edges(
            all_uu_edges=(u_all, v_all, weights_all, common_items_all),
            kept_edge_mask=kept_mask,
            user_degrees=user_deg,
        )

        # Analyze merged nodes
        analyzer.analyze_merged_nodes(
            user_to_super=user_to_super,
            train_interactions=seen_train,
        )

        # Correlate compression vs NDCG across sweep runs
        analyzer.add_compression_run(
            compression_ratio=0.4,
            ndcg_baseline=0.12,
            ndcg_gsp=0.11,
            mean_overlap=0.75,
            run_label="frac50_ms3",
        )
        analyzer.save()
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._removed_edges: List[EdgeRemovalAnalysis] = []
        self._merged_nodes: List[NodeMergeAnalysis] = []
        self._correlations: List[CompressionCorrelation] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Edge removal analysis
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_removed_edges(
        self,
        u_arr: np.ndarray,
        v_arr: np.ndarray,
        weights: np.ndarray,
        common_items: np.ndarray,
        kept_mask: np.ndarray,           # bool array: True = kept, False = removed
        user_degrees: np.ndarray,        # (U,) degree of each user
        cointeraction_threshold: int = 2,
        max_edges: int = 10000,
    ) -> None:
        """
        Analyze removed edges to determine if they represent noise or signal.

        Parameters
        ----------
        u_arr, v_arr    edge endpoints (upper-triangle user pairs)
        weights         edge curvature / similarity scores
        common_items    number of co-rated items per edge
        kept_mask       boolean mask: True = edge was kept
        user_degrees    degree of each user in the full interaction graph
        """
        removed_mask = ~kept_mask
        n_removed = int(removed_mask.sum())

        if n_removed == 0:
            return

        # Sample if too many removed edges
        removed_idx = np.where(removed_mask)[0]
        if len(removed_idx) > max_edges:
            rng = np.random.default_rng(42)
            removed_idx = rng.choice(removed_idx, size=max_edges, replace=False)

        # Threshold: below median weight = "weak" edge
        median_weight = float(np.median(weights))
        median_common = float(np.median(common_items))

        analyses: List[EdgeRemovalAnalysis] = []
        for idx in removed_idx.tolist():
            u = int(u_arr[idx])
            v = int(v_arr[idx])
            w = float(weights[idx])
            ci = int(common_items[idx])
            deg_u = int(user_degrees[u]) if u < len(user_degrees) else 0
            deg_v = int(user_degrees[v]) if v < len(user_degrees) else 0

            analyses.append(EdgeRemovalAnalysis(
                user_u=u,
                user_v=v,
                edge_weight=w,
                common_items=ci,
                deg_u=deg_u,
                deg_v=deg_v,
                removal_reason="low_importance",
                is_weak=(w < median_weight),
                is_low_cointeraction=(ci < cointeraction_threshold),
            ))

        self._removed_edges = analyses

    def get_edge_removal_summary(self) -> Dict:
        """Aggregate statistics about removed edges."""
        if not self._removed_edges:
            return {}

        weights   = np.array([e.edge_weight   for e in self._removed_edges])
        commons   = np.array([e.common_items  for e in self._removed_edges])
        deg_u     = np.array([e.deg_u         for e in self._removed_edges])
        deg_v     = np.array([e.deg_v         for e in self._removed_edges])
        weak      = np.array([e.is_weak       for e in self._removed_edges])
        low_co    = np.array([e.is_low_cointeraction for e in self._removed_edges])

        return {
            "num_removed_edges_analyzed": len(self._removed_edges),
            "pct_weak_similarity":        float(np.mean(weak) * 100),
            "pct_low_cointeraction":      float(np.mean(low_co) * 100),
            "mean_removed_edge_weight":   float(np.mean(weights)),
            "std_removed_edge_weight":    float(np.std(weights)),
            "mean_removed_common_items":  float(np.mean(commons)),
            "mean_removed_deg_u":         float(np.mean(deg_u)),
            "mean_removed_deg_v":         float(np.mean(deg_v)),
            "interpretation": (
                "Most removed edges are weak (noise removal)"
                if float(np.mean(weak)) > 0.7
                else "Mix of strong and weak edges removed"
            ),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Node merge analysis
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_merged_nodes(
        self,
        user_to_super: np.ndarray,
        train_interactions: Dict[int, Set[int]],
        overlap_threshold: float = 0.3,
        max_clusters: int = 2000,
    ) -> None:
        """
        Analyze merged super-nodes to determine signal gain vs information loss.

        A cluster has "signal gain" when merged users share many items
        (dense overlap → averaging their embeddings improves signal quality).
        A cluster has "information loss" when merged users have disjoint
        item sets (merging loses individual distinction).

        Parameters
        ----------
        user_to_super       (U,) mapping of original users to super-node IDs
        train_interactions  user_id -> set of item_ids
        overlap_threshold   min pairwise overlap fraction to count as signal gain
        max_clusters        cap to avoid excessive computation
        """
        # Group users by super-node
        super_to_users: Dict[int, List[int]] = {}
        for uid, sid in enumerate(user_to_super.tolist()):
            super_to_users.setdefault(sid, []).append(uid)

        # Only analyze multi-user clusters
        multi_clusters = [(sid, uids) for sid, uids in super_to_users.items() if len(uids) > 1]
        if len(multi_clusters) > max_clusters:
            rng = np.random.default_rng(42)
            idxs = rng.choice(len(multi_clusters), size=max_clusters, replace=False)
            multi_clusters = [multi_clusters[i] for i in idxs.tolist()]

        analyses: List[NodeMergeAnalysis] = []
        for sid, uids in multi_clusters:
            item_sets = [train_interactions.get(u, set()) for u in uids]
            union_items = set().union(*item_sets)
            n_total = len(union_items)

            # Pairwise overlap fraction (Jaccard-like)
            pairwise_overlaps: List[float] = []
            for i in range(len(item_sets)):
                for j in range(i + 1, len(item_sets)):
                    a, b = item_sets[i], item_sets[j]
                    union_ab = len(a | b)
                    if union_ab > 0:
                        pairwise_overlaps.append(len(a & b) / union_ab)

            mean_overlap = float(np.mean(pairwise_overlaps)) if pairwise_overlaps else 0.0

            # Super-node items (union of all merged users)
            analyses.append(NodeMergeAnalysis(
                super_node_id=sid,
                merged_users=uids,
                cluster_size=len(uids),
                total_items_before=n_total,
                items_after_merge=n_total,   # Union (embedding covers all)
                overlap_fraction=mean_overlap,
                signal_gain=(mean_overlap >= overlap_threshold),
                information_loss=(mean_overlap < overlap_threshold / 2),
            ))

        self._merged_nodes = analyses

    def get_node_merge_summary(self) -> Dict:
        """Aggregate statistics about merged nodes."""
        if not self._merged_nodes:
            return {}

        cluster_sizes    = np.array([n.cluster_size    for n in self._merged_nodes])
        overlaps         = np.array([n.overlap_fraction for n in self._merged_nodes])
        signal_gains     = np.array([n.signal_gain      for n in self._merged_nodes])
        info_losses      = np.array([n.information_loss for n in self._merged_nodes])

        return {
            "num_multi_user_clusters": len(self._merged_nodes),
            "mean_cluster_size":       float(np.mean(cluster_sizes)),
            "max_cluster_size":        int(np.max(cluster_sizes)),
            "mean_pairwise_overlap":   float(np.mean(overlaps)),
            "pct_signal_gain":         float(np.mean(signal_gains) * 100),
            "pct_information_loss":    float(np.mean(info_losses) * 100),
            "interpretation": _interpret_merges(
                float(np.mean(signal_gains)),
                float(np.mean(info_losses)),
            ),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Compression vs accuracy correlation
    # ─────────────────────────────────────────────────────────────────────────

    def add_compression_run(
        self,
        compression_ratio: float,
        ndcg_baseline: float,
        ndcg_gsp: float,
        mean_overlap: float,
        run_label: str = "",
    ) -> CompressionCorrelation:
        """Register one compression experiment for cross-run correlation."""
        ndcg_change = ndcg_gsp - ndcg_baseline
        ndcg_change_pct = (
            (ndcg_change / max(abs(ndcg_baseline), 1e-9)) * 100
        )
        cc = CompressionCorrelation(
            compression_ratio=compression_ratio,
            ndcg_baseline=ndcg_baseline,
            ndcg_gsp=ndcg_gsp,
            ndcg_change=ndcg_change,
            ndcg_change_pct=ndcg_change_pct,
            overlap_at_k=mean_overlap,
            stability_score=mean_overlap,
            run_label=run_label,
        )
        self._correlations.append(cc)
        return cc

    def get_correlation_analysis(self) -> Dict:
        """
        Compute correlation between compression ratio and:
        - NDCG change
        - Recommendation stability (overlap@K)
        """
        if len(self._correlations) < 2:
            return {}

        comp_ratios = np.array([c.compression_ratio for c in self._correlations])
        ndcg_changes = np.array([c.ndcg_change      for c in self._correlations])
        stabilities  = np.array([c.stability_score  for c in self._correlations])

        def _pearson(a: np.ndarray, b: np.ndarray) -> float:
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        corr_ndcg     = _pearson(comp_ratios, ndcg_changes)
        corr_stability = _pearson(comp_ratios, stabilities)

        return {
            "num_runs": len(self._correlations),
            "compression_range": [float(comp_ratios.min()), float(comp_ratios.max())],
            "pearson_compression_vs_ndcg_change":    corr_ndcg,
            "pearson_compression_vs_stability":       corr_stability,
            "ndcg_change_range": [float(ndcg_changes.min()), float(ndcg_changes.max())],
            "mean_stability": float(np.mean(stabilities)),
            "interpretation": _interpret_correlation(corr_ndcg, corr_stability),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # I/O
    # ─────────────────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save all inference analyses to CSV and JSON."""
        # --- Removed edges summary ---
        edge_summary = self.get_edge_removal_summary()
        if edge_summary:
            json_path = os.path.join(self.output_dir, "edge_removal_summary.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(edge_summary, fh, indent=2)

        if self._removed_edges:
            csv_path = os.path.join(self.output_dir, "removed_edges_sample.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "user_u", "user_v", "edge_weight", "common_items",
                    "deg_u", "deg_v", "removal_reason",
                    "is_weak", "is_low_cointeraction",
                ])
                for e in self._removed_edges[:5000]:  # save a sample
                    writer.writerow([
                        e.user_u, e.user_v, f"{e.edge_weight:.4f}",
                        e.common_items, e.deg_u, e.deg_v,
                        e.removal_reason, e.is_weak, e.is_low_cointeraction,
                    ])

        # --- Node merge summary ---
        merge_summary = self.get_node_merge_summary()
        if merge_summary:
            json_path = os.path.join(self.output_dir, "node_merge_summary.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(merge_summary, fh, indent=2)

        if self._merged_nodes:
            csv_path = os.path.join(self.output_dir, "merged_nodes_sample.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "super_node_id", "cluster_size", "total_items",
                    "overlap_fraction", "signal_gain", "information_loss",
                ])
                for n in self._merged_nodes[:5000]:
                    writer.writerow([
                        n.super_node_id, n.cluster_size, n.total_items_before,
                        f"{n.overlap_fraction:.4f}", n.signal_gain, n.information_loss,
                    ])

        # --- Compression vs NDCG correlation ---
        if self._correlations:
            csv_path = os.path.join(self.output_dir, "compression_correlation.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "run_label", "compression_ratio",
                    "ndcg_baseline", "ndcg_gsp",
                    "ndcg_change", "ndcg_change_pct",
                    "overlap_at_k", "stability_score",
                ])
                for c in self._correlations:
                    writer.writerow([
                        c.run_label, f"{c.compression_ratio:.4f}",
                        f"{c.ndcg_baseline:.6f}", f"{c.ndcg_gsp:.6f}",
                        f"{c.ndcg_change:.6f}", f"{c.ndcg_change_pct:.2f}",
                        f"{c.overlap_at_k:.4f}", f"{c.stability_score:.4f}",
                    ])

            corr_analysis = self.get_correlation_analysis()
            json_path = os.path.join(self.output_dir, "compression_correlation_analysis.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(corr_analysis, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _interpret_merges(pct_gain: float, pct_loss: float) -> str:
    if pct_gain >= 0.6:
        return "Majority of merged clusters share dense neighborhoods – signal improvement"
    if pct_loss >= 0.5:
        return "Majority of merged clusters have sparse overlap – potential information loss"
    return "Mixed compression: some signal gain, some information loss"


def _interpret_correlation(corr_ndcg: float, corr_stability: float) -> str:
    lines = []
    if abs(corr_ndcg) < 0.2:
        lines.append("Compression ratio has minimal impact on NDCG (near-zero correlation).")
    elif corr_ndcg < -0.4:
        lines.append("Higher compression correlates with NDCG reduction.")
    elif corr_ndcg > 0.4:
        lines.append("Higher compression correlates with NDCG improvement (noise removal effect).")
    else:
        lines.append(f"Weak correlation between compression and NDCG (r={corr_ndcg:.2f}).")

    if corr_stability < -0.4:
        lines.append("Higher compression reduces recommendation stability.")
    elif corr_stability > 0.4:
        lines.append("Higher compression maintains or improves recommendation stability.")
    else:
        lines.append(f"Compression has limited impact on stability (r={corr_stability:.2f}).")
    return " ".join(lines)
