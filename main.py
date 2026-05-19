"""
main.py – Full GSP recommender pipeline.

Usage examples
--------------
# Quick debug run (single model):
    python main.py --models lightgcn --debug_mode

# Full dataset, multiple models, GSP enabled:
    python main.py --models lightgcn graphsage gat --use_gsp true --epochs 10

# All three models at once (shorthand):
    python main.py --models all

# Full run with all defaults (same as running the runner directly):
    python main.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
import torch
import torch.nn as nn

from gsprec.config import (
    DataConfig,
    EvalConfig,
    GSPConfig,
    ProjectConfig,
    TrainRunConfig,
)
from gsprec.data.pipeline import load_and_build_graph
from gsprec.graph.gsp_ops import gsp_preprocess
from gsprec.models.architectures import ModelConfig, get_model
from gsprec.models.evaluator import (
    BatchedEvaluator,
    EvalConfig as EvalCfg,
    compute_efficiency_metrics,
)
from gsprec.models.trainer import TrainConfig as BPRTrainConfig, train_model
from gsprec.analytics import run_analytics_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_jsonl(path: str, record: dict) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", encoding="utf-8", buffering=1) as fh:
        fh.write(json.dumps(record) + "\n")


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _leave_one_out_split(ratings_df, threshold: float = 4.0, seed: int = 42):
    """Leave-one-out split: hold out exactly 1 positive per user.

    For each user with at least 2 positive interactions (rating >= threshold),
    randomly hold out exactly 1 positive as the test item.  All other
    interactions remain in training.  Users with fewer than 2 positives are
    kept in training only (cannot be evaluated).
    """
    rng = np.random.default_rng(seed)
    test_idx = []
    for uid, grp in ratings_df.groupby("UserID"):
        pos_mask = grp["Rating"].to_numpy(dtype=np.float32) >= threshold
        pos_idxs = grp.index[pos_mask].to_numpy()
        if len(pos_idxs) >= 2:
            chosen = int(rng.choice(pos_idxs))
            test_idx.append(chosen)
    mask = ratings_df.index.isin(test_idx)
    return ratings_df.loc[~mask].reset_index(drop=True), ratings_df.loc[mask].reset_index(drop=True)


def _build_seen_sets(df) -> Dict[int, set]:
    seen: Dict[int, set] = {}
    for row in df.itertuples(index=False):
        seen.setdefault(int(row.UserID), set()).add(int(row.MovieID))
    return seen


def _build_test_positives(test_df, threshold: float = 4.0) -> Dict[int, list]:
    pos: Dict[int, list] = {}
    for row in test_df.itertuples(index=False):
        if float(row.Rating) >= threshold:
            pos.setdefault(int(row.UserID), []).append(int(row.MovieID))
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Graph builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_bipartite_edge_index(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    num_super: int,
) -> torch.Tensor:
    """Build undirected PyG edge_index;  item offsets by num_super."""
    item_shifted = item_idx + num_super
    src = np.concatenate([user_idx, item_shifted])
    dst = np.concatenate([item_shifted, user_idx])
    return torch.from_numpy(np.stack([src, dst])).long()


def _coarsen_train(train_df, user_to_super: np.ndarray, num_super: int):
    """Map each training user → its super-node and aggregate ratings."""
    tc = train_df.copy()
    tc["super_idx"] = user_to_super[tc["UserID"].to_numpy(dtype=np.int64)]
    agg = (
        tc.groupby(["super_idx", "MovieID"], as_index=False)
        .agg(rating=("Rating", "mean"))
        .astype({"super_idx": np.int64, "MovieID": np.int64})
    )
    edge_index = _build_bipartite_edge_index(
        agg["super_idx"].to_numpy(),
        agg["MovieID"].to_numpy(),
        num_super,
    )
    train_u = torch.tensor(agg["super_idx"].to_numpy(), dtype=torch.long)
    train_i = torch.tensor(agg["MovieID"].to_numpy() + num_super, dtype=torch.long)
    train_y = torch.tensor(agg["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)
    return edge_index, train_u, train_i, train_y


def _build_baseline_graph(train_df, num_users: int):
    """Build baseline (no coarsening) bipartite edge_index."""
    tc = train_df.copy()
    agg = (
        tc.groupby(["UserID", "MovieID"], as_index=False)
        .agg(rating=("Rating", "mean"))
        .astype({"UserID": np.int64, "MovieID": np.int64})
    )
    edge_index = _build_bipartite_edge_index(
        agg["UserID"].to_numpy(),
        agg["MovieID"].to_numpy(),
        num_users,
    )
    train_u = torch.tensor(agg["UserID"].to_numpy(), dtype=torch.long)
    train_i = torch.tensor(agg["MovieID"].to_numpy() + num_users, dtype=torch.long)
    train_y = torch.tensor(agg["rating"].to_numpy(dtype=np.float32), dtype=torch.float32)
    return edge_index, train_u, train_i, train_y


# ─────────────────────────────────────────────────────────────────────────────
# Single run (baseline or GSP)
# ─────────────────────────────────────────────────────────────────────────────

def run_single(
    *,
    model_name: str,
    run_tag: str,
    edge_index: torch.Tensor,
    train_u: torch.Tensor,
    train_i: torch.Tensor,
    train_y: torch.Tensor,
    num_nodes: int,
    num_super: int,
    num_users: int,
    num_items: int,
    user_to_super: Optional[np.ndarray],
    test_df,
    test_positives: Dict,
    seen_train: Dict,
    train_cfg: BPRTrainConfig,
    eval_cfg: EvalCfg,
    output_dir: str,
    device: torch.device,
    model_cfg: ModelConfig,
) -> Tuple[Dict, "BatchedEvaluator", "nn.Module"]:
    """Train one model and return (summary_dict, evaluator, model)."""
    _ensure_dir(output_dir)
    _set_seed(train_cfg.seed)

    model = get_model(model_name, num_nodes, model_cfg).to(device)

    evaluator = BatchedEvaluator(
        edge_index=edge_index,
        num_users=num_users,
        num_items=num_items,
        test_positives=test_positives,
        seen_positives=seen_train,
        test_df=test_df,
        eval_cfg=eval_cfg,
        device=device,
        user_to_super=user_to_super,
        num_super=num_super,
    )

    summary = train_model(
        model=model,
        edge_index=edge_index,
        train_user_nodes=train_u,
        train_item_nodes=train_i,
        train_ratings=train_y,
        config=train_cfg,
        run_name=run_tag,
        eval_callback=evaluator.evaluate_and_log_callback(run_tag),
    )

    # Final eval
    final_metrics = evaluator.evaluate(model)
    summary.update(final_metrics)
    _append_jsonl(
        os.path.join(output_dir, "eval_metrics.jsonl"),
        {"run": run_tag, **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in final_metrics.items()}},
    )
    return summary, evaluator, model


# ─────────────────────────────────────────────────────────────────────────────
# Analytics: logging, explanation, compression analysis, report
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _get_embeddings(
    model: "nn.Module",
    edge_index: torch.Tensor,
    num_super: int,
    user_to_super: Optional[np.ndarray],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run final inference and return (user_emb, item_emb)."""
    model.eval()
    ei = edge_index.to(device)
    z = model(ei).detach().cpu().numpy()
    super_emb = z[:num_super]
    item_emb  = z[num_super:]
    if user_to_super is not None:
        user_emb = super_emb[user_to_super]
    else:
        user_emb = super_emb
    return user_emb, item_emb


def _run_analytics(
    *,
    model_name: str,
    output_dir: str,
    device: torch.device,
    # Baseline data
    baseline_model: "nn.Module",
    base_ei: torch.Tensor,
    num_users: int,
    num_items: int,
    # GSP data (may be None if use_gsp=False)
    gsp_model: Optional["nn.Module"],
    gsp_ei: Optional[torch.Tensor],
    num_super: int,
    user_to_super: Optional[np.ndarray],
    gsp_out: Optional[dict],
    base_edge_count: int,
    gsp_edge_count: int,
    # Training data
    train_df,
    seen_train: Dict[int, set],
    test_positives: Dict[int, list],
    # Eval summaries
    baseline_summary: Dict,
    gsp_summary: Dict,
    # Metadata
    curvature_mode: str = "",
    fraction: float = 1.0,
    min_shared: int = 1,
    dataset_name: str = "movielens",
) -> None:
    """Run full analytics suite for one model after training."""
    print("[Analytics] Extracting embeddings …")
    base_user_emb, base_item_emb = _get_embeddings(
        baseline_model, base_ei, num_users, None, device
    )
    # L2-normalise (same as scripts)
    base_user_emb = base_user_emb / np.maximum(
        np.linalg.norm(base_user_emb, axis=1, keepdims=True), 1e-8
    )
    base_item_emb = base_item_emb / np.maximum(
        np.linalg.norm(base_item_emb, axis=1, keepdims=True), 1e-8
    )

    gsp_user_emb = gsp_item_emb = gsp_super_emb = None
    if gsp_model is not None and gsp_ei is not None and user_to_super is not None:
        gsp_super_emb, raw_item_emb = _get_embeddings(
            gsp_model, gsp_ei, num_super, None, device
        )
        gsp_user_emb = gsp_super_emb[user_to_super]
        gsp_user_emb = gsp_user_emb / np.maximum(
            np.linalg.norm(gsp_user_emb, axis=1, keepdims=True), 1e-8
        )
        gsp_item_emb = raw_item_emb / np.maximum(
            np.linalg.norm(raw_item_emb, axis=1, keepdims=True), 1e-8
        )

    run_analytics_pipeline(
        model_name=model_name,
        output_dir=output_dir,
        base_user_emb=base_user_emb,
        base_item_emb=base_item_emb,
        gsp_user_emb=gsp_user_emb,
        gsp_item_emb=gsp_item_emb,
        gsp_super_emb=gsp_super_emb,
        num_users=num_users,
        num_items=num_items,
        num_super=num_super,
        user_to_super=user_to_super,
        gsp_out=gsp_out,
        base_edge_count=base_edge_count,
        gsp_edge_count=gsp_edge_count,
        seen_train=seen_train,
        test_positives=test_positives,
        baseline_summary=baseline_summary,
        gsp_summary=gsp_summary,
        curvature_mode=curvature_mode,
        fraction=fraction,
        min_shared=min_shared,
        dataset_name=dataset_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline (one dataset mode)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_for_mode(
    *,
    model_names: list,
    use_gsp: bool,
    debug_mode: bool,
    epochs: int,
    batch_size: int,
    alpha: float,
    topk: int,
    output_dir: str,
    data_cfg: DataConfig,
    gsp_cfg: GSPConfig,
    train_run_cfg: TrainRunConfig,
    eval_cfg_obj: EvalConfig,
    seed: int,
) -> list:
    """
    End-to-end pipeline for one dataset mode, training each model in model_names.

    Data loading and GSP preprocessing run once; training loops over each model.
    Returns a list of result dicts, one per model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Pipeline: models={model_names}  gsp={use_gsp}  debug={debug_mode}  device={device}")
    print(f"{'='*60}")

    t_pipeline = time.perf_counter()

    # ── Data loading ─────────────────────────────────────────────────────────
    print("[Pipeline] Loading data …")
    data = load_and_build_graph(
        debug_mode=debug_mode,
        max_debug_users=data_cfg.max_debug_users,
        cache_dir=data_cfg.cache_dir,
        force_reload=data_cfg.force_reload,
    )
    ratings    = data["ratings_df"]
    num_users  = data["num_users"]
    num_items  = data["num_items"]
    load_time  = data["load_time_s"]
    print(f"[Pipeline] users={num_users}  items={num_items}  load_time={load_time:.2f}s")

    train_df, test_df = _leave_one_out_split(ratings, threshold=data_cfg.implicit_threshold, seed=seed)
    seen_train    = _build_seen_sets(train_df)
    test_pos      = _build_test_positives(test_df, threshold=data_cfg.implicit_threshold)

    ev_cfg = EvalCfg(
        k=eval_cfg_obj.k,
        num_negatives=eval_cfg_obj.num_negatives,
        seed=eval_cfg_obj.seed,
    )

    # ── Build shared graphs (once, reused across all models) ──────────────────
    print("[Pipeline] Building baseline graph …")
    base_ei, base_u, base_i, base_y = _build_baseline_graph(train_df, num_users)
    base_num_nodes = num_users + num_items

    preprocess_time = 0.0
    user_to_super   = None
    num_super       = num_users
    gsp_ei = gsp_u = gsp_i = gsp_y = None
    gsp_num_nodes   = None
    gsp_out         = None

    if use_gsp:
        print("[Pipeline] Running GSP preprocessing …")
        t_pre = time.perf_counter()
        gsp_out = gsp_preprocess(
            ratings_df=train_df,
            num_users=num_users,
            num_items=num_items,
            implicit_threshold=data_cfg.implicit_threshold,
            alpha=alpha,
            curvature_percentile=gsp_cfg.curvature_percentile,
            curvature_topk=topk if topk > 0 else None,
            importance_percentile=gsp_cfg.importance_percentile,
            importance_topk=topk if topk > 0 else None,
            er_num_eigenvectors=gsp_cfg.er_num_eigenvectors,
            cache_dir=data_cfg.cache_dir,
            output_dir=output_dir,
            data_load_time_s=load_time,
            device=None,
        )
        preprocess_time = time.perf_counter() - t_pre
        user_to_super = gsp_out["user_to_super"]
        num_super     = gsp_out["num_super"]
        gsp_ei, gsp_u, gsp_i, gsp_y = _coarsen_train(train_df, user_to_super, num_super)
        gsp_num_nodes = num_super + num_items

    # ── Per-model training loop ───────────────────────────────────────────────
    all_model_results = []

    for model_name in model_names:
        print(f"\n[Pipeline] ── Model: {model_name} ──")
        t_model = time.perf_counter()
        model_sub_out = os.path.join(output_dir, model_name)
        _ensure_dir(model_sub_out)

        model_cfg = ModelConfig(
            emb_dim=train_run_cfg.emb_dim,
            hidden_dim=train_run_cfg.hidden_dim,
            out_dim=train_run_cfg.out_dim,
            num_layers=train_run_cfg.num_layers,
            heads=train_run_cfg.heads,
            dropout=train_run_cfg.dropout,
        )

        base_train_cfg = BPRTrainConfig(
            epochs=epochs,
            lr=train_run_cfg.lr,
            weight_decay=train_run_cfg.weight_decay,
            batch_size=batch_size,
            neg_ratio=train_run_cfg.neg_ratio,
            emb_l2_weight=train_run_cfg.emb_l2_weight,
            use_amp=train_run_cfg.use_amp,
            seed=seed,
            checkpoint_dir=os.path.join(model_sub_out, "checkpoints"),
            metrics_jsonl_path=os.path.join(model_sub_out, "metrics.jsonl"),
            training_log_path=os.path.join(model_sub_out, "training_log.txt"),
        )

        print(f"[Pipeline] Training BASELINE ({model_name}) …")
        baseline_summary, baseline_evaluator, baseline_model = run_single(
            model_name=model_name,
            run_tag=f"{model_name}_baseline",
            edge_index=base_ei,
            train_u=base_u,
            train_i=base_i,
            train_y=base_y,
            num_nodes=base_num_nodes,
            num_super=num_users,
            num_users=num_users,
            num_items=num_items,
            user_to_super=None,
            test_df=test_df,
            test_positives=test_pos,
            seen_train=seen_train,
            train_cfg=base_train_cfg,
            eval_cfg=ev_cfg,
            output_dir=model_sub_out,
            device=device,
            model_cfg=model_cfg,
        )

        gsp_summary: Dict = {}
        gsp_model = None
        gsp_evaluator = None
        if use_gsp and gsp_ei is not None:
            gsp_train_cfg = BPRTrainConfig(
                epochs=epochs,
                lr=train_run_cfg.lr,
                weight_decay=train_run_cfg.weight_decay,
                batch_size=batch_size,
                neg_ratio=train_run_cfg.neg_ratio,
                emb_l2_weight=train_run_cfg.emb_l2_weight,
                use_amp=train_run_cfg.use_amp,
                seed=seed,
                checkpoint_dir=os.path.join(model_sub_out, "checkpoints"),
                metrics_jsonl_path=os.path.join(model_sub_out, "metrics.jsonl"),
                training_log_path=os.path.join(model_sub_out, "training_log.txt"),
            )

            print(f"[Pipeline] Training GSP ({model_name}) …")
            gsp_summary, gsp_evaluator, gsp_model = run_single(
                model_name=model_name,
                run_tag=f"{model_name}_gsp",
                edge_index=gsp_ei,
                train_u=gsp_u,
                train_i=gsp_i,
                train_y=gsp_y,
                num_nodes=gsp_num_nodes,
                num_super=num_super,
                num_users=num_users,
                num_items=num_items,
                user_to_super=user_to_super,
                test_df=test_df,
                test_positives=test_pos,
                seen_train=seen_train,
                train_cfg=gsp_train_cfg,
                eval_cfg=ev_cfg,
                output_dir=model_sub_out,
                device=device,
                model_cfg=model_cfg,
            )

        model_time = time.perf_counter() - t_model
        result: Dict = {
            "model":             model_name,
            "debug_mode":        debug_mode,
            "use_gsp":           use_gsp,
            "model_time_s":      round(model_time, 3),
            "data_load_time_s":  round(load_time, 3),
            "preprocess_time_s": round(preprocess_time, 3),
            "num_users":         num_users,
            "num_items":         num_items,
            "baseline":          baseline_summary,
        }
        if use_gsp and gsp_summary:
            result["gsp"] = gsp_summary
            eff = compute_efficiency_metrics(baseline_summary, gsp_summary, preprocess_time)
            result["efficiency"] = eff

        all_model_results.append(result)

        # ── Analytics block (non-fatal: errors are logged but don't abort) ──────
        try:
            _run_analytics(
                model_name=model_name,
                output_dir=model_sub_out,
                device=device,
                baseline_model=baseline_model,
                base_ei=base_ei,
                num_users=num_users,
                num_items=num_items,
                gsp_model=gsp_model if use_gsp and gsp_ei is not None else None,
                gsp_ei=gsp_ei if use_gsp else None,
                num_super=num_super,
                user_to_super=user_to_super,
                gsp_out=gsp_out if use_gsp else None,
                base_edge_count=int(base_ei.shape[1]),
                gsp_edge_count=int(gsp_ei.shape[1]) if use_gsp and gsp_ei is not None else 0,
                train_df=train_df,
                seen_train=seen_train,
                test_positives=test_pos,
                baseline_summary=baseline_summary,
                gsp_summary=gsp_summary,
                curvature_mode=gsp_cfg.curvature_mode if use_gsp else "",
                fraction=float(num_super) / max(num_users, 1),
                min_shared=gsp_cfg.min_shared_interactions if use_gsp else 1,
                dataset_name=data_cfg.dataset_name,
            )
        except Exception as _analytics_exc:
            print(f"[Analytics] WARNING: analytics failed for {model_name}: {_analytics_exc}")

    _ = time.perf_counter() - t_pipeline  # total wall-time available if needed
    return all_model_results


# ─────────────────────────────────────────────────────────────────────────────
# Final summary printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison(results: list) -> None:
    """Pretty-print accuracy, speedup, and time breakdown."""
    sep = "─" * 70
    print(f"\n{'═'*70}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'═'*70}")

    for r in results:
        tag = f"model={r['model']}  debug={r['debug_mode']}  gsp={r['use_gsp']}"
        print(f"\n{sep}")
        print(f"  {tag}")
        print(sep)

        base = r.get("baseline", {})
        gsp  = r.get("gsp", {})
        eff  = r.get("efficiency", {})

        k_ndcg   = [k for k in base if "NDCG"      in k]
        k_prec   = [k for k in base if "Precision" in k]
        k_recall = [k for k in base if "Recall"    in k]

        for metric in k_ndcg + k_prec + k_recall + ["RMSE", "MAE"]:
            bv = base.get(metric, "—")
            gv = gsp.get(metric,  "—") if gsp else "—"
            bv_s = f"{bv:.4f}" if isinstance(bv, float) else str(bv)
            gv_s = f"{gv:.4f}" if isinstance(gv, float) else str(gv)
            print(f"  {metric:<25} baseline={bv_s:<12} gsp={gv_s}")

        if eff:
            print()
            print(f"  Epoch speedup (GSP / baseline) : {eff.get('epoch_speedup', '—'):.3f}×")
            print(f"  Memory reduction ratio         : {eff.get('memory_reduction_ratio', 0)*100:.1f}%")
            print(f"  Preprocessing time             : {eff.get('gsp_preprocess_time_s', 0):.2f}s")
            print(f"  GSP training time              : {eff.get('gsp_total_train_time_s', 0):.2f}s")
            print(f"  Baseline training time         : {eff.get('baseline_total_train_time_s', 0):.2f}s")
            print(f"  Preprocessing / training ratio : {eff.get('preprocess_vs_training_ratio', 0):.3f}")

    print(f"\n{'═'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GSP Recommender – full training pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--models", nargs="+", default=["lightgcn"],
        metavar="MODEL",
        help='GNN architecture(s) to train: lightgcn graphsage gat  (or "all")',
    )
    p.add_argument(
        "--use_gsp", type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        metavar="true/false",
        help="Run the GSP pipeline in addition to baseline",
    )
    p.add_argument("--alpha",      type=float, default=0.5,   help="GSP blend weight α")
    p.add_argument("--topk",       type=int,   default=0,     help="Top-k edge selection (0 = percentile)")
    p.add_argument("--epochs",     type=int,   default=10,    help="Training epochs")
    p.add_argument("--batch_size", type=int,   default=65536, help="Training batch size")
    p.add_argument(
        "--debug_mode", action="store_true",
        help="If set, run debug (5k users) FIRST then full dataset",
    )
    p.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    p.add_argument("--config",     type=str, default="",       help="Path to JSON config (overrides CLI)")
    p.add_argument("--seed",       type=int, default=42,       help="Global random seed")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ALL_MODELS = ["lightgcn", "graphsage", "gat"]

    # ── Load JSON config if given ─────────────────────────────────────────────
    if args.config:
        from gsprec.config import ProjectConfig
        proj_cfg = ProjectConfig.from_json(args.config)
        data_cfg     = proj_cfg.data
        gsp_cfg      = proj_cfg.gsp
        train_run_cfg = proj_cfg.train
        eval_cfg_obj  = proj_cfg.eval
        output_dir    = proj_cfg.output_dir
        model_list    = list(proj_cfg.models)
    else:
        data_cfg = DataConfig(
            implicit_threshold=4.0,
            test_seed=args.seed,
            debug_mode=args.debug_mode,
            max_debug_users=5000,
            cache_dir=os.path.join(args.output_dir, "cache"),
            force_reload=False,
        )
        gsp_cfg = GSPConfig(
            alpha=args.alpha,
            curvature_percentile=70.0,
            curvature_topk=args.topk,
            importance_percentile=50.0,
            importance_topk=args.topk,
            er_num_eigenvectors=32,
            min_shared_interactions=2,
        )
        train_run_cfg = TrainRunConfig(
            epochs=args.epochs,
            lr=1e-3,
            weight_decay=1e-5,
            batch_size=args.batch_size,
            neg_ratio=4,
            emb_l2_weight=1e-5,
            emb_dim=64,
            hidden_dim=128,
            out_dim=64,
            num_layers=3,
            heads=4,
            dropout=0.2,
            use_amp=True,
            seed=args.seed,
        )
        eval_cfg_obj = EvalConfig(k=10, num_negatives=99, seed=args.seed)
        output_dir = args.output_dir
        model_list = ALL_MODELS if "all" in args.models else args.models

    _ensure_dir(output_dir)
    _set_seed(args.seed)

    all_results = []

    # ── Flow: debug first, then full ─────────────────────────────────────────
    modes = []
    if args.debug_mode:
        # Step 1: debug sanity-check
        modes.append(True)
    # Step 2: full dataset (always)
    modes.append(False)

    for debug in modes:
        dc = DataConfig(
            implicit_threshold=data_cfg.implicit_threshold,
            test_seed=data_cfg.test_seed,
            debug_mode=debug,
            max_debug_users=data_cfg.max_debug_users,
            cache_dir=data_cfg.cache_dir,
            force_reload=data_cfg.force_reload,
        )
        sub_out = os.path.join(output_dir, "debug" if debug else "full")
        _ensure_dir(sub_out)

        results = run_pipeline_for_mode(
            model_names=model_list,
            use_gsp=args.use_gsp,
            debug_mode=debug,
            epochs=args.epochs,
            batch_size=args.batch_size,
            alpha=args.alpha,
            topk=args.topk,
            output_dir=sub_out,
            data_cfg=dc,
            gsp_cfg=gsp_cfg,
            train_run_cfg=train_run_cfg,
            eval_cfg_obj=eval_cfg_obj,
            seed=args.seed,
        )
        all_results.extend(results)

        # Save per-mode JSON
        summary_path = os.path.join(sub_out, "run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"[main] Summary saved → {summary_path}")

    # Global summary
    global_summary_path = os.path.join(output_dir, "global_summary.json")
    with open(global_summary_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)

    _print_comparison(all_results)

if __name__ == "__main__":
    main()