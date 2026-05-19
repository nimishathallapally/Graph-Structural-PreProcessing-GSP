"""
Metrics export utilities.

Generates all publication-ready output files after a pipeline run:

    metrics.csv              Ranking + regression metrics (P, R, NDCG, RMSE, MAE)
    memory_usage.csv         CPU / GPU memory before/after each stage
    speedup_results.csv      Training-time speedup (original vs. reduced graph)
    reduction_stats.csv      Node / edge counts and reduction percentages
    preprocessing_time.csv   Per-stage preprocessing wall-clock times
    training_logs.json       Full training log (epoch-level metrics)
    hardware_info.json       CPU / GPU / RAM hardware description
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mkdir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) if "." in os.path.basename(path) else path, exist_ok=True)


def _save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"[Export] Saved {path}  ({len(df)} rows)")


def _save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    print(f"[Export] Saved {path}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Individual file generators
# ---------------------------------------------------------------------------

def save_metrics_csv(records: List[Dict], output_dir: str) -> None:
    """
    metrics.csv — ranking + regression metrics for each model/run.

    Expected keys per record:
        model, run_type,
        Precision@10, Recall@10, NDCG@10, RMSE, MAE, [HitRate@10]
    """
    df = pd.DataFrame(records)
    # Ensure canonical column order
    cols = [
        "model", "run_type",
        "Precision@10", "Recall@10", "NDCG@10",
        "HitRate@10", "RMSE", "MAE",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = float("nan")
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]
    _save_csv(df, os.path.join(output_dir, "metrics.csv"))


def save_memory_usage_csv(records: List[Dict], output_dir: str) -> None:
    """
    memory_usage.csv — per-stage CPU/GPU memory snapshots.

    Expected keys per record:
        stage, cpu_rss_start_MB, cpu_rss_end_MB, cpu_rss_delta_MB,
        gpu_alloc_start_MB, gpu_alloc_end_MB, gpu_peak_MB,
        gpu_util_start_pct, gpu_util_end_pct
    """
    df = pd.DataFrame(records)
    cols = [
        "stage",
        "cpu_rss_start_MB", "cpu_rss_end_MB", "cpu_rss_delta_MB",
        "gpu_alloc_start_MB", "gpu_alloc_end_MB", "gpu_peak_MB",
        "gpu_util_start_pct", "gpu_util_end_pct",
        "elapsed_s",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = float("nan")
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]
    _save_csv(df, os.path.join(output_dir, "memory_usage.csv"))


def save_speedup_results_csv(records: List[Dict], output_dir: str) -> None:
    """
    speedup_results.csv — training-time speedup for every model.

    Expected keys per record:
        model,
        training_time_original_s, training_time_reduced_s,
        speedup_factor,
        GPU_memory_original_MB, GPU_memory_reduced_MB,
        GPU_memory_reduction_pct,
        CPU_memory_original_MB, CPU_memory_reduced_MB,
        CPU_memory_reduction_pct,
        training_throughput_original, training_throughput_reduced
    """
    df = pd.DataFrame(records)
    cols = [
        "model",
        "training_time_original_s", "training_time_reduced_s", "speedup_factor",
        "GPU_memory_original_MB", "GPU_memory_reduced_MB", "GPU_memory_reduction_pct",
        "CPU_memory_original_MB", "CPU_memory_reduced_MB", "CPU_memory_reduction_pct",
        "training_throughput_original", "training_throughput_reduced",
        "GPU_utilization_original_pct", "GPU_utilization_reduced_pct",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = float("nan")
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]
    _save_csv(df, os.path.join(output_dir, "speedup_results.csv"))


def save_reduction_stats_csv(records: List[Dict], output_dir: str) -> None:
    """
    reduction_stats.csv — graph size reduction between stages.

    Expected keys per record:
        stage,
        nodes_before, nodes_after,
        edges_before, edges_after, edges_removed,
        reduction_percent,
        avg_degree_before, avg_degree_after,
        density_before, density_after,
        memory_before_MB, memory_after_MB
    """
    df = pd.DataFrame(records)
    cols = [
        "stage",
        "nodes_before", "nodes_after",
        "edges_before", "edges_after", "edges_removed", "reduction_percent",
        "avg_degree_before", "avg_degree_after",
        "density_before", "density_after",
        "memory_before_MB", "memory_after_MB",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = float("nan")
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]
    _save_csv(df, os.path.join(output_dir, "reduction_stats.csv"))


def save_preprocessing_time_csv(records: List[Dict], output_dir: str) -> None:
    """
    preprocessing_time.csv — wall-clock time per preprocessing stage.

    Expected keys per record:
        stage, elapsed_s, description
    """
    df = pd.DataFrame(records)
    cols = ["stage", "elapsed_s", "description"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]
    _save_csv(df, os.path.join(output_dir, "preprocessing_time.csv"))


def save_training_logs_json(records: List[Dict], output_dir: str) -> None:
    """
    training_logs.json — full epoch-level training log for all runs.
    """
    _save_json(records, os.path.join(output_dir, "training_logs.json"))


def save_hardware_info_json(hardware_info: Dict, output_dir: str) -> None:
    """hardware_info.json — CPU / GPU / RAM hardware description."""
    _save_json(hardware_info, os.path.join(output_dir, "hardware_info.json"))


def save_dataset_stats_csv(stats: Dict, output_dir: str) -> None:
    """dataset_stats.csv — top-level dataset statistics."""
    df = pd.DataFrame([stats])
    _save_csv(df, os.path.join(output_dir, "dataset_stats.csv"))


# ---------------------------------------------------------------------------
# Master export function
# ---------------------------------------------------------------------------

def export_all_results(
    output_dir: str,
    *,
    metrics_records: Optional[List[Dict]] = None,
    memory_records: Optional[List[Dict]] = None,
    speedup_records: Optional[List[Dict]] = None,
    reduction_records: Optional[List[Dict]] = None,
    preprocessing_records: Optional[List[Dict]] = None,
    training_log_records: Optional[List[Dict]] = None,
    hardware_info: Optional[Dict] = None,
    dataset_stats: Optional[Dict] = None,
    reproducibility_info: Optional[Dict] = None,
) -> None:
    """
    Write all publication-ready output files to output_dir.

    Any argument left as None is skipped silently.
    """
    os.makedirs(output_dir, exist_ok=True)

    if metrics_records is not None:
        save_metrics_csv(metrics_records, output_dir)

    if memory_records is not None:
        save_memory_usage_csv(memory_records, output_dir)

    if speedup_records is not None:
        save_speedup_results_csv(speedup_records, output_dir)

    if reduction_records is not None:
        save_reduction_stats_csv(reduction_records, output_dir)

    if preprocessing_records is not None:
        save_preprocessing_time_csv(preprocessing_records, output_dir)

    if training_log_records is not None:
        save_training_logs_json(training_log_records, output_dir)

    if hardware_info is not None:
        save_hardware_info_json(hardware_info, output_dir)

    if dataset_stats is not None:
        save_dataset_stats_csv(dataset_stats, output_dir)

    if reproducibility_info is not None:
        _save_json(reproducibility_info, os.path.join(output_dir, "reproducibility.json"))

    print(f"[Export] All output files written to: {output_dir}")
