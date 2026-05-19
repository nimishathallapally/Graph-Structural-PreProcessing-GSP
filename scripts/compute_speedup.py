"""
Compute GSP vs Baseline speedup in time, memory, and recommendation quality.

Usage:
    python3 scripts/compute_speedup.py [--output_dir outputs/amazon_music]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main(output_dir: str) -> None:
    eval_path     = os.path.join(output_dir, "eval_metrics.jsonl")
    pipeline_path = os.path.join(output_dir, "pipeline_metrics.jsonl")

    # ── Load eval metrics ────────────────────────────────────────────────────
    eval_records = load_jsonl(eval_path)
    gsp_metrics:      Dict[str, dict] = {}
    baseline_metrics: Dict[str, dict] = {}

    for rec in eval_records:
        model: str = rec["model"]
        if model.endswith("_gsp"):
            name = model[: -len("_gsp")]
            # Keep last entry (final eval)
            gsp_metrics[name] = rec
        elif model.endswith("_baseline"):
            name = model[: -len("_baseline")]
            baseline_metrics[name] = rec

    # ── Load pipeline metrics ────────────────────────────────────────────────
    speedup_record: dict = {}
    if os.path.exists(pipeline_path):
        for rec in load_jsonl(pipeline_path):
            if rec.get("stage") == "speedup":
                speedup_record = rec
                break

    # ── Report ───────────────────────────────────────────────────────────────
    all_models = sorted(set(gsp_metrics) | set(baseline_metrics))

    print("=" * 72)
    print("  GSP vs BASELINE  —  Recommendation quality")
    print("=" * 72)
    metric_keys = ["NDCG@10", "Precision@10", "Recall@10", "RMSE", "MAE"]
    header = f"{'Model':<14}" + "".join(f"{'GSP':>10}{'Base':>10}{'Delta':>10}" for _ in metric_keys)
    col_headers = f"{'Model':<14}" + "".join(
        f"{k:>10}{'':>10}{'':>10}" for k in metric_keys
    )
    sub_headers = f"{'':14}" + "".join(f"{'GSP':>10}{'Base':>10}{'Δ':>10}" for _ in metric_keys)
    # Re-do cleaner
    print(f"\n{'Model':<14}", end="")
    for k in metric_keys:
        pad = 10 if len(k) <= 10 else len(k) + 2
        print(f"{'GSP':>10}{'Base':>10}{'Δ':>10}", end="")
    print()
    print(f"{'':14}" + "".join(f"  {k:<28}" for k in metric_keys))
    print("-" * (14 + 30 * len(metric_keys)))

    for name in all_models:
        gsp  = gsp_metrics.get(name, {})
        base = baseline_metrics.get(name, {})
        print(f"{name:<14}", end="")
        for k in metric_keys:
            gv = gsp.get(k, float("nan"))
            bv = base.get(k, float("nan"))
            try:
                delta = gv - bv
                sign  = "+" if delta > 0 else ""
                # For RMSE/MAE lower is better, so negative delta is good
                print(f"{gv:>10.4f}{bv:>10.4f}{sign+f'{delta:.4f}':>10}", end="")
            except TypeError:
                print(f"{'N/A':>10}{'N/A':>10}{'N/A':>10}", end="")
        print()

    # ── Time & Memory speedup ────────────────────────────────────────────────
    if not speedup_record:
        print("\n[!] pipeline_metrics.jsonl has no 'speedup' record yet — training may still be running.")
        return

    print("\n" + "=" * 72)
    print("  GSP vs BASELINE  —  Training efficiency (per-epoch)")
    print("=" * 72)
    print(f"\n{'Model':<14}{'Base time':>12}{'GSP time':>12}{'Speedup':>12}{'Base mem':>12}{'GSP mem':>12}{'Mem saved':>12}")
    print("-" * 86)

    for name in all_models:
        base_t   = speedup_record.get(f"{name}_baseline_avg_epoch_time_s", None)
        gsp_t    = speedup_record.get(f"{name}_gsp_avg_epoch_time_s",      None)
        speedup  = speedup_record.get(f"{name}_speedup",                   None)
        base_mem = speedup_record.get(f"{name}_baseline_max_gpu_mem_mb",   None)
        gsp_mem  = speedup_record.get(f"{name}_gsp_max_gpu_mem_mb",        None)

        if base_t is None:
            print(f"{name:<14}{'N/A':>12}")
            continue

        mem_saved_pct = 100.0 * (base_mem - gsp_mem) / base_mem if base_mem and gsp_mem else float("nan")
        speedup_str   = f"{speedup:.2f}×" if speedup else "N/A"
        mem_saved_str = f"{mem_saved_pct:+.1f}%" if not (mem_saved_pct != mem_saved_pct) else "N/A"

        print(
            f"{name:<14}"
            f"{base_t*1000:>10.2f}ms"
            f"{gsp_t*1000:>10.2f}ms"
            f"{speedup_str:>12}"
            f"{base_mem:>10.1f}MB"
            f"{gsp_mem:>10.1f}MB"
            f"{mem_saved_str:>12}"
        )

    print()
    print("Note: Speedup > 1× means GSP trains faster per epoch.")
    print("      Negative Δ for RMSE/MAE means GSP is more accurate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/amazon_music")
    args = parser.parse_args()
    main(args.output_dir)
