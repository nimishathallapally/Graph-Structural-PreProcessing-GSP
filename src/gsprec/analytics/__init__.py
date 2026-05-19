"""
GSP Recommender Analytics Package

Modules:
- recommendation_logger: Track and save top-K recommendations per user
- explainer: Neighborhood, graph, and path-based explanations
- compression_analyzer: Structural impact of GSP compression
- inference_analyzer: Edge/node removal analysis
- report_generator: Final summary report generation
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from .recommendation_logger import RecommendationLogger
from .explainer import RecommendationExplainer
from .compression_analyzer import CompressionAnalyzer
from .inference_analyzer import InferenceAnalyzer
from .report_generator import ReportGenerator

__all__ = [
    "RecommendationLogger",
    "RecommendationExplainer",
    "CompressionAnalyzer",
    "InferenceAnalyzer",
    "ReportGenerator",
    "run_analytics_pipeline",
]


def run_analytics_pipeline(
    *,
    model_name: str,
    output_dir: str,
    # Pre-computed L2-normalised embeddings
    base_user_emb: np.ndarray,
    base_item_emb: np.ndarray,
    gsp_user_emb: Optional[np.ndarray] = None,
    gsp_item_emb: Optional[np.ndarray] = None,
    gsp_super_emb: Optional[np.ndarray] = None,   # raw (un-projected) super-node embs
    # Dimensions
    num_users: int = 0,
    num_items: int = 0,
    num_super: int = 0,
    user_to_super: Optional[np.ndarray] = None,
    # GSP graph data
    gsp_out: Optional[dict] = None,
    base_edge_count: int = 0,
    gsp_edge_count: int = 0,
    # Interaction data
    seen_train: Optional[Dict[int, Set[int]]] = None,
    test_positives: Optional[Dict[int, List[int]]] = None,
    # Eval summaries (dicts with NDCG@K keys)
    baseline_summary: Optional[Dict] = None,
    gsp_summary: Optional[Dict] = None,
    # Run metadata
    curvature_mode: str = "",
    fraction: float = 1.0,
    min_shared: int = 1,
    dataset_name: str = "movielens",
) -> None:
    """Run the full analytics suite on pre-computed embeddings.

    This is the shared implementation called by ``main.py``, ``run_ml1m.py``,
    ``run_yelp_compressible.py``, and ``run_ml25m.py``.  All heavy computation
    (training, inference) must already be done; this function only post-processes
    embeddings and writes analytics output files.

    Outputs written under ``output_dir/analytics/``:
      recommendations/   – top-K recs per user + overlap CSV
      explanations/      – per-user item explanations
      compression/       – graph stats, embedding drift, preservation
      inference/         – edge/node removal analysis
      report/            – Markdown + JSON summary report
    """
    import os

    seen_train    = seen_train    or {}
    test_positives = test_positives or {}
    baseline_summary = baseline_summary or {}
    gsp_summary      = gsp_summary      or {}

    analytics_dir = os.path.join(output_dir, "analytics")
    os.makedirs(analytics_dir, exist_ok=True)

    print(f"\n[Analytics] Running analytics for model={model_name} …")

    # ── 1. Recommendation Logging ─────────────────────────────────────────────
    print("[Analytics] Logging recommendations …")
    rec_logger = RecommendationLogger(
        output_dir=os.path.join(analytics_dir, "recommendations"),
        top_ks=[10, 20, 50],
        curvature_mode=curvature_mode,
        fraction=fraction,
        min_shared=min_shared,
    )

    base_recs = rec_logger.log_recommendations(
        base_user_emb, base_item_emb,
        test_positives, seen_train,
        model_name=model_name,
        run_type="baseline",
    )

    gsp_recs: list = []
    if gsp_user_emb is not None and gsp_item_emb is not None:
        gsp_recs = rec_logger.log_recommendations(
            gsp_user_emb, gsp_item_emb,
            test_positives, seen_train,
            model_name=model_name,
            run_type="gsp",
            user_to_super=user_to_super,
        )

    rec_logger.save(model_name=model_name)
    overlap_stats = rec_logger.get_aggregate_overlap_stats(model_name)
    _ov = overlap_stats.get(10, {}).get("mean_overlap", "N/A")
    if isinstance(_ov, float):
        print(f"[Analytics] Overlap@10: {_ov:.3f}")
    else:
        print(f"[Analytics] Overlap@10: {_ov}")

    # ── 2. Explainable Recommendations ───────────────────────────────────────
    print("[Analytics] Generating explanations …")
    uu_edges_for_exp = None
    if gsp_out is not None:
        u_hc = gsp_out.get("u_hc")
        v_hc = gsp_out.get("v_hc")
        F_hc = gsp_out.get("F_hc")
        if u_hc is not None and u_hc.size > 0:
            uu_edges_for_exp = (u_hc, v_hc, F_hc)

    explainer = RecommendationExplainer(
        user_emb=base_user_emb,
        item_emb=base_item_emb,
        train_interactions={uid: set(items) for uid, items in seen_train.items()},
        output_dir=os.path.join(analytics_dir, "explanations"),
        uu_edges=uu_edges_for_exp,
        num_neighbors=5,
    )

    base_rec_map = {r.user_id: r.recommended_items for r in base_recs}
    explainer.explain_batch(
        base_rec_map, model_name=model_name, run_type="baseline",
        top_k=10, max_users=500,
    )

    if gsp_recs and gsp_user_emb is not None and gsp_item_emb is not None:
        gsp_explainer = RecommendationExplainer(
            user_emb=gsp_user_emb,
            item_emb=gsp_item_emb,
            train_interactions={uid: set(items) for uid, items in seen_train.items()},
            output_dir=os.path.join(analytics_dir, "explanations"),
            uu_edges=uu_edges_for_exp,
            num_neighbors=5,
        )
        gsp_rec_map = {r.user_id: r.recommended_items for r in gsp_recs}
        gsp_explainer.explain_batch(
            gsp_rec_map, model_name=model_name, run_type="gsp",
            top_k=10, max_users=500,
        )
        for key, recs_list in gsp_explainer._explanations.items():
            explainer._explanations[key] = recs_list

    explainer.save(model_name=model_name)
    sample_exps = explainer.get_sample_explanations(
        f"{model_name}_baseline", n_users=5, items_per_user=3
    )

    # ── 3. Compression Analysis ───────────────────────────────────────────────
    comp_analyzer = CompressionAnalyzer(
        output_dir=os.path.join(analytics_dir, "compression")
    )
    graph_stats_dict: Dict = {}
    drift_summary: Dict = {}
    preservation_dicts: list = []

    if gsp_out is not None and user_to_super is not None and gsp_super_emb is not None:
        print("[Analytics] Computing compression statistics …")
        stats = gsp_out.get("stats", {})
        gs = comp_analyzer.compute_graph_stats(
            num_original_users=num_users,
            num_super_nodes=num_super,
            num_uu_edges_before=int(stats.get("uu_edges_all", 0)),
            num_uu_edges_after=int(stats.get("uu_edges_pruned", 0)),
            base_edge_index_size=base_edge_count,
            gsp_edge_index_size=gsp_edge_count,
            user_to_super=user_to_super,
        )
        graph_stats_dict = {
            "num_original_users":  gs.num_original_users,
            "num_super_nodes":     gs.num_super_nodes,
            "compression_ratio":   gs.compression_ratio,
            "singleton_ratio":     gs.singleton_ratio,
            "num_original_edges":  gs.num_original_edges,
            "num_gsp_edges":       gs.num_gsp_edges,
            "edge_reduction_ratio": gs.edge_reduction_ratio,
            "num_uu_edges_before": gs.num_uu_edges_before,
            "num_uu_edges_after":  gs.num_uu_edges_after,
        }

        print("[Analytics] Computing embedding drift …")
        comp_analyzer.compute_embedding_drift(
            user_to_super=user_to_super,
            baseline_user_emb=base_user_emb,
            gsp_super_emb=gsp_super_emb,
            max_users=5000,
        )
        drift_summary = comp_analyzer.get_drift_summary()

        print("[Analytics] Computing recommendation preservation …")
        base_rec_top_map = {r.user_id: r.recommended_items for r in base_recs}
        gsp_rec_top_map  = {r.user_id: r.recommended_items for r in gsp_recs}

        item_popularity = np.zeros(num_items, dtype=np.int32)
        for items in seen_train.values():
            for it in items:
                if it < num_items:
                    item_popularity[it] += 1

        for k in [10, 20, 50]:
            pres = comp_analyzer.compute_recommendation_preservation(
                base_rec_top_map, gsp_rec_top_map, item_popularity, k=k
            )
            preservation_dicts.append({
                "k":                       pres.k,
                "mean_overlap":            pres.mean_overlap,
                "std_overlap":             pres.std_overlap,
                "mean_diversity_baseline": pres.mean_diversity_baseline,
                "mean_diversity_gsp":      pres.mean_diversity_gsp,
                "diversity_change":        pres.diversity_change,
                "mean_popularity_baseline": pres.mean_popularity_baseline,
                "mean_popularity_gsp":     pres.mean_popularity_gsp,
                "popularity_bias_shift":   pres.popularity_bias_shift,
            })

        comp_analyzer.save()

    # ── 4. Inference Analysis (Edge / Node Removal) ───────────────────────────
    inf_analyzer = InferenceAnalyzer(
        output_dir=os.path.join(analytics_dir, "inference")
    )
    edge_removal_summary: Dict = {}
    node_merge_summary: Dict = {}

    if gsp_out is not None and user_to_super is not None:
        u_hc      = gsp_out.get("u_hc",        np.array([], dtype=np.int64))
        v_hc      = gsp_out.get("v_hc",        np.array([], dtype=np.int64))
        F_hc      = gsp_out.get("F_hc",        np.array([], dtype=np.float32))
        sel       = gsp_out.get("selected_mask", np.ones(u_hc.size, dtype=bool))
        common_hc = gsp_out.get("common_hc",   np.zeros(u_hc.size, dtype=np.float32))
        user_deg  = gsp_out.get("user_deg",    np.zeros(num_users,  dtype=np.float32))

        if u_hc.size > 0:
            print("[Analytics] Analyzing removed edges …")
            inf_analyzer.analyze_removed_edges(
                u_arr=u_hc,
                v_arr=v_hc,
                weights=F_hc,
                common_items=common_hc.astype(np.int32),
                kept_mask=sel,
                user_degrees=user_deg.astype(np.int32),
                cointeraction_threshold=min_shared,
                max_edges=10000,
            )
            edge_removal_summary = inf_analyzer.get_edge_removal_summary()

        print("[Analytics] Analyzing merged nodes …")
        inf_analyzer.analyze_merged_nodes(
            user_to_super=user_to_super,
            train_interactions={uid: set(items) for uid, items in seen_train.items()},
            max_clusters=2000,
        )
        node_merge_summary = inf_analyzer.get_node_merge_summary()

        inf_analyzer.add_compression_run(
            compression_ratio=float(gsp_out.get("stats", {}).get("compression_ratio", 0)),
            ndcg_baseline=float(baseline_summary.get("NDCG@10", 0)),
            ndcg_gsp=float(gsp_summary.get("NDCG@10", 0)),
            mean_overlap=float(overlap_stats.get(10, {}).get("mean_overlap", 0)),
            run_label=f"{model_name}_{curvature_mode}_frac{fraction:.2f}_ms{min_shared}",
        )
        inf_analyzer.save()

    # ── 5. Report ─────────────────────────────────────────────────────────────
    print("[Analytics] Generating report …")
    report = ReportGenerator(
        output_dir=os.path.join(analytics_dir, "report"),
        model_name=model_name,
        dataset_name=dataset_name,
    )
    report.set_graph_stats(graph_stats_dict)
    report.add_eval_metrics(model_name, baseline_summary, gsp_summary)
    report.set_overlap_stats(overlap_stats)
    report.set_drift_summary(drift_summary)
    report.set_edge_removal_summary(edge_removal_summary)
    report.set_node_merge_summary(node_merge_summary)
    report.set_preservation_results(preservation_dicts)
    report.set_sample_recs(
        [{"user_id": r.user_id, "recommended_items": r.recommended_items,
          "ground_truth": r.ground_truth} for r in base_recs[:20]],
        [{"user_id": r.user_id, "recommended_items": r.recommended_items,
          "ground_truth": r.ground_truth} for r in gsp_recs[:20]],
    )
    report.set_sample_explanations(sample_exps)
    report_path = report.generate()
    print(f"[Analytics] Report → {report_path}")
