#!/usr/bin/env python3
"""
run_yelp_pipeline.py
====================
Entry-point script for the Yelp GSP/ICG recommendation pipeline.

Usage
-----
    # Run with default config (configs/yelp.json):
    python scripts/run_yelp_pipeline.py

    # Specify config explicitly:
    python scripts/run_yelp_pipeline.py --config configs/yelp.json

    # Override data/output directories:
    python scripts/run_yelp_pipeline.py \\
        --data_dir ./data/yelp_dataset \\
        --output_dir ./outputs/yelp_run1

    # Quick test with a small subset of users:
    python scripts/run_yelp_pipeline.py --debug

    # Install dependencies first if needed:
    #   pip install -r requirements.txt
    #   pip install psutil    (recommended for accurate memory tracking)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the src directory so gsprec is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gsprec.config import ProjectConfig, DataConfig, GSPConfig, TrainRunConfig, EvalConfig
from gsprec.pipeline.yelp_runner import run_yelp_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Yelp GSP/ICG Recommendation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, default="configs/yelp.json",
        help="Path to JSON config file (default: configs/yelp.json)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="",
        help="Path to Yelp dataset directory (overrides config)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="",
        help="Output directory for results (overrides config)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: use only first 5000 most-active users",
    )
    parser.add_argument(
        "--epochs", type=int, default=0,
        help="Override number of training epochs (0 = use config value)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--device", type=str, default="",
        choices=["", "cpu", "cuda"],
        help="Force device (default: auto-detect CUDA)",
    )
    parser.add_argument(
        "--er_solver", type=str, default="",
        choices=["", "arpack", "lobpcg", "jl"],
        help="ER solver for GSP Stage II: arpack | lobpcg | jl (default: use config value)",
    )
    parser.add_argument(
        "--er_sketches", type=int, default=0,
        help="Number of JL random probes when --er_solver=jl (0 = use config value)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if cfg_path.exists():
        print(f"[Runner] Loading config from: {cfg_path}")
        config = ProjectConfig.from_json(str(cfg_path))
    else:
        print(f"[Runner] Config not found at {cfg_path}, using GSP defaults")
        config = ProjectConfig()

    # ── Override data directory ───────────────────────────────────────────────
    if args.data_dir:
        config.data.dataset_path = args.data_dir
    if not config.data.dataset_path:
        config.data.dataset_path = "./data"

    # ── Override output directory ─────────────────────────────────────────────
    if args.output_dir:
        config.output_dir = args.output_dir

    # ── Debug mode ────────────────────────────────────────────────────────────
    if args.debug:
        config.data.debug_mode = True
        config.data.max_debug_users = 2000
        config.train.epochs = min(config.train.epochs, 5)
        print("[Runner] Debug mode: 2000 users, 5 epochs max")

    # ── Epoch override ────────────────────────────────────────────────────────
    if args.epochs > 0:
        config.train.epochs = args.epochs

    # ── Device override ───────────────────────────────────────────────────────
    # The pipeline auto-detects CUDA; device arg here is for documentation.
    if args.device:
        import os
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ── ER solver override ────────────────────────────────────────────────────
    if args.er_solver:
        config.gsp.er_solver = args.er_solver
    if args.er_sketches > 0:
        config.gsp.er_sketches = args.er_sketches

    # ── Seed ──────────────────────────────────────────────────────────────────
    config.train.seed = args.seed
    config.data.test_seed = args.seed

    # ── Print effective config ────────────────────────────────────────────────
    print("\n[Runner] Effective configuration:")
    print(f"  dataset_path      : {config.data.dataset_path}")
    print(f"  output_dir        : {config.output_dir}")
    print(f"  min_interactions  : {config.data.min_interactions}")
    print(f"  implicit_threshold: {config.data.implicit_threshold}")
    print(f"  epochs            : {config.train.epochs}")
    print(f"  emb_dim           : {config.train.emb_dim}")
    print(f"  num_layers        : {config.train.num_layers}")
    print(f"  gsp.alpha         : {config.gsp.alpha}")
    print(f"  gsp.er_eigvecs    : {config.gsp.er_num_eigenvectors}")
    print(f"  gsp.er_solver     : {config.gsp.er_solver}")
    print(f"  gsp.er_sketches   : {config.gsp.er_sketches}")
    print(f"  seed              : {config.train.seed}")
    print()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    run_yelp_pipeline(config)


if __name__ == "__main__":
    main()
