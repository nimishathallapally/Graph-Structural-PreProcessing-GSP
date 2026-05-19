"""
Report Generator – Final Analysis Summary

Produces a comprehensive Markdown + JSON report covering:
  - Sample recommendations (baseline vs GSP)
  - Example explanations
  - Overlap statistics across K values
  - Graph compression impact:
    * Accuracy (NDCG change)
    * Diversity change
    * Robustness / stability
  - Edge/node removal interpretation
  - Compression vs performance correlations
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


class ReportGenerator:
    """
    Assembles all analytics module outputs into a final Markdown report.

    Usage
    -----
    ::
        report = ReportGenerator(output_dir="outputs/report")
        report.set_graph_stats(graph_stats_dict)
        report.set_eval_metrics(baseline_metrics, gsp_metrics, model_name)
        report.set_overlap_stats(overlap_stats_by_k)
        report.set_drift_summary(drift_summary)
        report.set_edge_removal_summary(edge_summary)
        report.set_node_merge_summary(merge_summary)
        report.set_correlation_analysis(corr_analysis)
        report.set_sample_recs(baseline_recs, gsp_recs, n=5)
        report.set_sample_explanations(explanations)
        report.generate()
    """

    def __init__(self, output_dir: str, model_name: str = "", dataset_name: str = ""):
        self.output_dir = output_dir
        self.model_name = model_name
        self.dataset_name = dataset_name
        os.makedirs(output_dir, exist_ok=True)

        self._graph_stats: Dict = {}
        self._eval_metrics: List[Tuple[str, Dict, Dict]] = []   # (model, baseline, gsp)
        self._overlap_stats: Dict[int, Dict] = {}               # k -> stats
        self._drift_summary: Dict = {}
        self._edge_removal_summary: Dict = {}
        self._node_merge_summary: Dict = {}
        self._correlation_analysis: Dict = {}
        self._preservation_results: List[Dict] = []
        self._sample_recs: List[Dict] = []
        self._sample_explanations: List[Dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Setters
    # ─────────────────────────────────────────────────────────────────────────

    def set_graph_stats(self, stats: Dict) -> None:
        self._graph_stats = stats

    def add_eval_metrics(self, model: str, baseline: Dict, gsp: Dict) -> None:
        self._eval_metrics.append((model, baseline, gsp))

    def set_overlap_stats(self, stats: Dict[int, Dict]) -> None:
        self._overlap_stats = stats

    def set_drift_summary(self, summary: Dict) -> None:
        self._drift_summary = summary

    def set_edge_removal_summary(self, summary: Dict) -> None:
        self._edge_removal_summary = summary

    def set_node_merge_summary(self, summary: Dict) -> None:
        self._node_merge_summary = summary

    def set_correlation_analysis(self, analysis: Dict) -> None:
        self._correlation_analysis = analysis

    def set_preservation_results(self, results: List[Dict]) -> None:
        self._preservation_results = results

    def set_sample_recs(
        self,
        baseline_recs: List[Dict],
        gsp_recs: List[Dict],
    ) -> None:
        """
        Store sample recommendations.

        Each rec dict: {user_id, recommended_items, ground_truth, scores}
        """
        self._sample_recs = []
        base_map = {r["user_id"]: r for r in baseline_recs}
        gsp_map  = {r["user_id"]: r for r in gsp_recs}

        for uid in list(set(base_map) & set(gsp_map))[:10]:
            self._sample_recs.append({
                "user_id": uid,
                "baseline_top10": base_map[uid].get("recommended_items", [])[:10],
                "gsp_top10":       gsp_map[uid].get("recommended_items", [])[:10],
                "ground_truth":    base_map[uid].get("ground_truth", []),
            })

    def set_sample_explanations(self, explanations: List[Dict]) -> None:
        self._sample_explanations = explanations[:5]

    # ─────────────────────────────────────────────────────────────────────────
    # Report generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate(self) -> str:
        """Generate Markdown report, save to disk, and return path."""
        md = self._build_markdown()
        md_path = os.path.join(self.output_dir, "analysis_report.md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(md)

        # Also save raw data as JSON for programmatic access
        raw = self._build_raw_json()
        json_path = os.path.join(self.output_dir, "analysis_report.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2)

        print(f"[Report] Saved → {md_path}")
        return md_path

    # ─────────────────────────────────────────────────────────────────────────
    # Internal builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_markdown(self) -> str:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        lines: List[str] = []

        lines += [
            f"# GSP Recommender – Analysis Report",
            f"",
            f"**Generated:** {now}  ",
            f"**Dataset:** {self.dataset_name or 'N/A'}  ",
            f"**Model(s):** {self.model_name or 'Multiple'}",
            f"",
            "---",
            "",
        ]

        # ── 1. Graph Compression Statistics ──────────────────────────────────
        lines.append("## 1. Graph Compression Statistics\n")
        if self._graph_stats:
            gs = self._graph_stats
            lines += [
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Original users | {gs.get('num_original_users', 'N/A'):,} |",
                f"| Super-nodes after GSP | {gs.get('num_super_nodes', 'N/A'):,} |",
                f"| Compression ratio | {gs.get('compression_ratio', 0):.2%} |",
                f"| Singleton ratio | {gs.get('singleton_ratio', 0):.2%} |",
                f"| Baseline bipartite edges | {gs.get('num_original_edges', 'N/A'):,} |",
                f"| GSP bipartite edges | {gs.get('num_gsp_edges', 'N/A'):,} |",
                f"| Edge reduction | {gs.get('edge_reduction_ratio', 0):.2%} |",
                f"| UU edges before compression | {gs.get('num_uu_edges_before', 'N/A'):,} |",
                f"| High-curvature UU edges kept | {gs.get('num_uu_edges_after', 'N/A'):,} |",
                "",
            ]
        else:
            lines.append("_No graph statistics available._\n")

        # ── 2. Accuracy Metrics ───────────────────────────────────────────────
        lines.append("## 2. Accuracy Metrics (Baseline vs GSP)\n")
        if self._eval_metrics:
            for model_name, base, gsp in self._eval_metrics:
                lines.append(f"### Model: `{model_name}`\n")
                lines += [
                    "| Metric | Baseline | GSP | Change |",
                    "|--------|----------|-----|--------|",
                ]
                for metric in ["NDCG@10", "Precision@10", "Recall@10",
                               "HitRate@10", "RMSE", "MAE"]:
                    bv = base.get(metric)
                    gv = gsp.get(metric)
                    if bv is None and gv is None:
                        continue
                    bv_s = f"{bv:.4f}" if isinstance(bv, float) else str(bv)
                    gv_s = f"{gv:.4f}" if isinstance(gv, float) else str(gv)
                    if isinstance(bv, float) and isinstance(gv, float):
                        delta = gv - bv
                        sign  = "+" if delta >= 0 else ""
                        change_s = f"{sign}{delta:.4f}"
                    else:
                        change_s = "—"
                    lines.append(f"| {metric} | {bv_s} | {gv_s} | {change_s} |")
                lines.append("")
        else:
            lines.append("_No evaluation metrics available._\n")

        # ── 3. Recommendation Overlap (Baseline vs GSP) ───────────────────────
        lines.append("## 3. Recommendation Overlap (Baseline vs GSP)\n")
        if self._overlap_stats:
            lines += [
                "| K | Mean Overlap | Std Overlap | Mean New Items | Mean Removed | Mean Rank Shifts |",
                "|---|-------------|-------------|----------------|--------------|------------------|",
            ]
            for k, stats in sorted(self._overlap_stats.items()):
                if not stats:
                    continue
                lines.append(
                    f"| {k} "
                    f"| {stats.get('mean_overlap', 0):.3f} "
                    f"| {stats.get('std_overlap', 0):.3f} "
                    f"| {stats.get('mean_new_items', 0):.1f} "
                    f"| {stats.get('mean_removed_items', 0):.1f} "
                    f"| {stats.get('mean_rank_shifts', 0):.1f} |"
                )
            lines.append("")
        else:
            lines.append("_No overlap statistics available._\n")

        # ── 4. Embedding Drift ─────────────────────────────────────────────────
        lines.append("## 4. Embedding Drift (GSP Compression Effect)\n")
        if self._drift_summary:
            ds = self._drift_summary
            lines += [
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Mean L2 drift | {ds.get('mean_embedding_drift', 0):.4f} |",
                f"| Std L2 drift | {ds.get('std_embedding_drift', 0):.4f} |",
                f"| Max L2 drift | {ds.get('max_embedding_drift', 0):.4f} |",
                f"| Median L2 drift | {ds.get('median_embedding_drift', 0):.4f} |",
                f"| Mean cosine similarity | {ds.get('mean_cosine_similarity', 0):.4f} |",
                f"| Mean cluster size | {ds.get('mean_cluster_size', 1):.2f} |",
                f"| % singleton clusters | {ds.get('pct_singleton', 100):.1f}% |",
                "",
            ]
        else:
            lines.append("_No embedding drift data available._\n")

        # ── 5. Recommendation Preservation ────────────────────────────────────
        lines.append("## 5. Recommendation Preservation (Diversity & Popularity)\n")
        if self._preservation_results:
            lines += [
                "| K | Overlap | Diversity Baseline | Diversity GSP | Δ Diversity | Pop Baseline | Pop GSP | Δ Popularity |",
                "|---|---------|-------------------|--------------|------------|-------------|---------|-------------|",
            ]
            for r in self._preservation_results:
                lines.append(
                    f"| {r.get('k')} "
                    f"| {r.get('mean_overlap', 0):.3f} "
                    f"| {r.get('mean_diversity_baseline', 0):.3f} "
                    f"| {r.get('mean_diversity_gsp', 0):.3f} "
                    f"| {r.get('diversity_change', 0):+.3f} "
                    f"| {r.get('mean_popularity_baseline', 0):.1f} "
                    f"| {r.get('mean_popularity_gsp', 0):.1f} "
                    f"| {r.get('popularity_bias_shift', 0):+.1f} |"
                )
            lines.append("")
        else:
            lines.append("_No preservation data available._\n")

        # ── 6. Edge Removal Analysis ───────────────────────────────────────────
        lines.append("## 6. Edge Removal Analysis\n")
        if self._edge_removal_summary:
            es = self._edge_removal_summary
            lines += [
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Edges analyzed | {es.get('num_removed_edges_analyzed', 0):,} |",
                f"| % weak similarity edges | {es.get('pct_weak_similarity', 0):.1f}% |",
                f"| % low co-interaction edges | {es.get('pct_low_cointeraction', 0):.1f}% |",
                f"| Mean removed edge weight | {es.get('mean_removed_edge_weight', 0):.4f} |",
                f"| Mean common items (removed) | {es.get('mean_removed_common_items', 0):.2f} |",
                "",
                f"**Interpretation:** {es.get('interpretation', 'N/A')}",
                "",
            ]
        else:
            lines.append("_No edge removal data available._\n")

        # ── 7. Node Merge Analysis ─────────────────────────────────────────────
        lines.append("## 7. Node Merge Analysis\n")
        if self._node_merge_summary:
            ns = self._node_merge_summary
            lines += [
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Multi-user clusters analyzed | {ns.get('num_multi_user_clusters', 0):,} |",
                f"| Mean cluster size | {ns.get('mean_cluster_size', 1):.2f} |",
                f"| Max cluster size | {ns.get('max_cluster_size', 1)} |",
                f"| Mean pairwise item overlap | {ns.get('mean_pairwise_overlap', 0):.4f} |",
                f"| % clusters with signal gain | {ns.get('pct_signal_gain', 0):.1f}% |",
                f"| % clusters with information loss | {ns.get('pct_information_loss', 0):.1f}% |",
                "",
                f"**Interpretation:** {ns.get('interpretation', 'N/A')}",
                "",
            ]
        else:
            lines.append("_No node merge data available._\n")

        # ── 8. Compression vs Performance Correlation ─────────────────────────
        lines.append("## 8. Compression vs Performance Correlation\n")
        if self._correlation_analysis:
            ca = self._correlation_analysis
            lines += [
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Runs analyzed | {ca.get('num_runs', 0)} |",
                f"| Compression range | {ca.get('compression_range', [0, 0])[0]:.2%} – {ca.get('compression_range', [0, 0])[1]:.2%} |",
                f"| Pearson r (compression vs NDCG Δ) | {ca.get('pearson_compression_vs_ndcg_change', 0):.3f} |",
                f"| Pearson r (compression vs stability) | {ca.get('pearson_compression_vs_stability', 0):.3f} |",
                f"| Mean recommendation stability | {ca.get('mean_stability', 0):.3f} |",
                "",
                f"**Interpretation:** {ca.get('interpretation', 'N/A')}",
                "",
            ]
        else:
            lines.append("_No correlation data available (need multiple runs)._\n")

        # ── 9. Sample Recommendations ─────────────────────────────────────────
        lines.append("## 9. Sample Recommendations (Baseline vs GSP)\n")
        if self._sample_recs:
            for rec in self._sample_recs[:5]:
                uid = rec["user_id"]
                base_top = rec.get("baseline_top10", [])[:5]
                gsp_top  = rec.get("gsp_top10", [])[:5]
                gt       = rec.get("ground_truth", [])
                overlap  = set(base_top) & set(gsp_top)
                lines += [
                    f"### User {uid}",
                    f"- **Ground truth items:** {gt[:5]}",
                    f"- **Baseline top-5:** {base_top}",
                    f"- **GSP top-5:** {gsp_top}",
                    f"- **Overlap (top-5):** {len(overlap)}/5 items match",
                    "",
                ]
        else:
            lines.append("_No sample recommendations available._\n")

        # ── 10. Sample Explanations ───────────────────────────────────────────
        lines.append("## 10. Sample Explanations\n")
        if self._sample_explanations:
            for user_exp in self._sample_explanations:
                uid = user_exp.get("user_id")
                lines.append(f"### User {uid} ({user_exp.get('run_type', '')} / {user_exp.get('model', '')})\n")
                for item_info in user_exp.get("top_items", [])[:3]:
                    lines += [
                        f"**Item {item_info['item_id']} (rank {item_info['rank']}):**",
                        f"- {item_info['explanation']}",
                        f"- Contributing users: {item_info.get('contributing_users', [])}",
                        f"- Path: {item_info.get('path', 'N/A')}",
                        "",
                    ]
        else:
            lines.append("_No explanations available._\n")

        # ── 11. Key Observations ──────────────────────────────────────────────
        lines += [
            "## 11. Key Observations\n",
            *self._generate_observations(),
            "",
            "---",
            "_Report generated by GSP Recommender Analytics Suite_",
        ]

        return "\n".join(lines)

    def _generate_observations(self) -> List[str]:
        obs: List[str] = []

        # Accuracy observation
        for model, base, gsp in self._eval_metrics:
            bndcg = base.get("NDCG@10")
            gndcg = gsp.get("NDCG@10")
            if isinstance(bndcg, float) and isinstance(gndcg, float):
                delta = gndcg - bndcg
                pct   = delta / max(abs(bndcg), 1e-9) * 100
                sign  = "improved" if delta >= 0 else "decreased"
                obs.append(
                    f"- **{model}**: GSP {sign} NDCG@10 by {abs(pct):.1f}% "
                    f"({bndcg:.4f} → {gndcg:.4f})."
                )

        # Compression observation
        if self._graph_stats:
            cr = self._graph_stats.get("compression_ratio", 0)
            obs.append(
                f"- Graph compressed by **{cr:.1%}** (users: "
                f"{self._graph_stats.get('num_original_users','?')} → "
                f"{self._graph_stats.get('num_super_nodes','?')} super-nodes)."
            )
            er = self._graph_stats.get("edge_reduction_ratio", 0)
            obs.append(f"- Bipartite graph edge reduction: **{er:.1%}**.")

        # Overlap observation
        if self._overlap_stats:
            for k, stats in sorted(self._overlap_stats.items())[:1]:
                ov = stats.get("mean_overlap", 0)
                obs.append(
                    f"- Mean recommendation overlap@{k}: **{ov:.1%}** "
                    "(higher = GSP preserves baseline recommendations)."
                )

        # Drift observation
        if self._drift_summary:
            drift = self._drift_summary.get("mean_embedding_drift", 0)
            cos   = self._drift_summary.get("mean_cosine_similarity", 1)
            obs.append(
                f"- Mean embedding drift (L2): **{drift:.4f}** "
                f"(cosine similarity: {cos:.4f})."
            )

        # Edge removal observation
        if self._edge_removal_summary:
            weak_pct = self._edge_removal_summary.get("pct_weak_similarity", 0)
            obs.append(
                f"- **{weak_pct:.1f}%** of removed edges were weak similarity links "
                "– consistent with noise removal."
            )

        # Node merge observation
        if self._node_merge_summary:
            sg_pct = self._node_merge_summary.get("pct_signal_gain", 0)
            il_pct = self._node_merge_summary.get("pct_information_loss", 0)
            obs.append(
                f"- Node merging: **{sg_pct:.1f}%** of clusters show signal gain, "
                f"**{il_pct:.1f}%** show potential information loss."
            )

        if not obs:
            obs.append("- No observations generated (missing data).")
        return obs

    def _build_raw_json(self) -> Dict:
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "dataset": self.dataset_name,
            "model": self.model_name,
            "graph_stats": self._graph_stats,
            "eval_metrics": [
                {"model": m, "baseline": b, "gsp": g}
                for m, b, g in self._eval_metrics
            ],
            "overlap_stats": {str(k): v for k, v in self._overlap_stats.items()},
            "drift_summary": self._drift_summary,
            "edge_removal_summary": self._edge_removal_summary,
            "node_merge_summary": self._node_merge_summary,
            "correlation_analysis": self._correlation_analysis,
            "preservation_results": self._preservation_results,
            "sample_recs": self._sample_recs,
            "sample_explanations": self._sample_explanations,
        }
