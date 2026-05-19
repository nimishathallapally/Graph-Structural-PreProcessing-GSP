"""
Training engine for GSP-based recommender system.

Features
--------
- BPR loss (vectorised negative sampling, no Python loops in hot path)
- Mixed-precision training (torch.cuda.amp)
- Per-epoch checkpointing: outputs/checkpoints/<run_name>/epoch_N.pt
  - saves model / optimizer / scheduler / epoch
  - auto-resumes from the latest checkpoint
- Epoch-level metrics: loss, time, GPU memory
- Logging to every-epoch-append JSONL + training_log.txt
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# AMP scaler – gracefully degrades to no-op on CPU
try:
    # PyTorch >= 2.4 unified API
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
    _AMP_DEVICE = "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_jsonl(path: str, record: dict) -> None:
    """Append one JSON record to a .jsonl file, flush immediately."""
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", encoding="utf-8", buffering=1) as fh:
        fh.write(json.dumps(record) + "\n")


def _setup_txt_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"train.{os.path.basename(log_path)}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 65536
    neg_ratio: int = 4             # negatives per positive (BPR)
    emb_l2_weight: float = 1e-5   # L2 regularisation on embeddings
    seed: int = 42
    use_amp: bool = True           # mixed precision (auto-disabled on CPU)
    checkpoint_dir: str = "outputs/checkpoints"
    save_epoch_checkpoints: bool = True  # set False to skip .pt writes (saves disk)
    metrics_jsonl_path: str = "outputs/metrics.jsonl"
    training_log_path: str = "outputs/training_log.txt"
    # Early stopping
    early_stopping_patience: int = 10   # epochs with no improvement before stopping; 0 = disabled
    early_stopping_min_delta: float = 1e-4  # minimum loss improvement to count as improvement
    # Legacy compat fields (ignored by new trainer, kept for runner interop)
    rating_loss_weight: float = 0.0
    log_jsonl_path: str = ""       # alias for metrics_jsonl_path
    use_rating_weights: bool = True  # weight BPR loss by rating value
    implicit_threshold: float = 3.5  # ratings above this receive full weight
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# BPR loss (fully vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def bpr_loss(
    z: torch.Tensor,
    pos_users: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
    emb_l2_weight: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Bayesian Personalised Ranking loss.

        L_BPR = -mean( log σ(score_pos - score_neg) )
        + emb_l2 * mean( ||e_u||² + ||e_pos||² + ||e_neg||² )

    All operations are vectorised (no Python loops).

    Returns
    -------
    loss, bpr_term, reg_term  (all scalar tensors)
    """
    u_emb   = z[pos_users]    # (B, D)
    pos_emb = z[pos_items]    # (B, D)
    neg_emb = z[neg_items]    # (B, D)

    pos_score = (u_emb * pos_emb).sum(dim=-1)  # (B,)
    neg_score = (u_emb * neg_emb).sum(dim=-1)  # (B,)

    bpr = -F.logsigmoid(pos_score - neg_score).mean()
    reg = emb_l2_weight * (
        u_emb.pow(2).mean() + pos_emb.pow(2).mean() + neg_emb.pow(2).mean()
    )
    return bpr + reg, bpr, reg


def rating_weighted_bpr_loss(
    z: torch.Tensor,
    pos_users: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
    pos_ratings: torch.Tensor,
    emb_l2_weight: float = 1e-5,
    rating_threshold: float = 3.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rating-weighted Bayesian Personalised Ranking loss.

    Each positive-negative pair is weighted by how strongly the user liked
    the positive item:

        w_i = clip((r_i - threshold) / (5 - threshold), 0.05, 1.0)
        L_rBPR = -mean( w_i * log σ(score_pos_i - score_neg_i) )
               + emb_l2 * mean( ||e_u||² + ||e_pos||² + ||e_neg||² )

    This makes 5-star positives exert a full gradient signal while
    just-above-threshold items receive a minimal (0.05) weight.  Items
    below threshold are not in the positive pool (filtered upstream).

    Parameters
    ----------
    pos_ratings : Tensor (B,)  – rating values for each positive pair.
    rating_threshold : float   – minimum star rating for a positive interaction.

    Returns
    -------
    loss, bpr_term, reg_term  (all scalar tensors)
    """
    u_emb   = z[pos_users]
    pos_emb = z[pos_items]
    neg_emb = z[neg_items]

    pos_score = (u_emb * pos_emb).sum(dim=-1)
    neg_score = (u_emb * neg_emb).sum(dim=-1)

    # Weight proportional to rating quality above threshold
    scale = max(5.0 - rating_threshold, 1e-3)
    weights = ((pos_ratings - rating_threshold) / scale).clamp(0.05, 1.0)

    per_pair = -F.logsigmoid(pos_score - neg_score)
    bpr = (per_pair * weights).mean()
    reg = emb_l2_weight * (
        u_emb.pow(2).mean() + pos_emb.pow(2).mean() + neg_emb.pow(2).mean()
    )
    return bpr + reg, bpr, reg


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised negative sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_negatives(
    pos_items: torch.Tensor,
    item_min: int,
    item_max: int,
    neg_ratio: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample ``neg_ratio`` negatives per positive item (vectorised, no loops).

    Simple uniform sampling without replacement per user is approximated by
    uniform sampling over the item range – exact deduplication would require
    a Python loop per user and is skipped for performance.

    Returns
    -------
    pos_users_rep  (B * neg_ratio,)   – user index repeated neg_ratio times
    neg_items_out  (B * neg_ratio,)   – sampled item indices
    """
    B = pos_items.size(0)
    n_neg = B * neg_ratio
    neg_items_out = torch.randint(
        low=item_min, high=item_max + 1, size=(n_neg,), device=device
    )
    return neg_items_out


# ─────────────────────────────────────────────────────────────────────────────
# Latest-checkpoint finder
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return path to the epoch_N.pt with highest N, or None if absent."""
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for fname in os.listdir(ckpt_dir):
        if fname.startswith("epoch_") and fname.endswith(".pt"):
            try:
                n = int(fname[len("epoch_"):-len(".pt")])
                candidates.append((n, os.path.join(ckpt_dir, fname)))
            except ValueError:
                pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# ─────────────────────────────────────────────────────────────────────────────
# Core trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    edge_index: torch.Tensor,
    train_user_nodes: torch.Tensor,
    train_item_nodes: torch.Tensor,
    train_ratings: torch.Tensor,
    config: TrainConfig,
    run_name: str,
    eval_callback: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Train a GNN recommender with BPR loss, AMP, and per-epoch checkpointing.

    Parameters
    ----------
    model                GNN model (will be moved to config.device).
    edge_index           Full graph edge_index (2, 2E) – PyG format.
    train_user_nodes     (N,) – user node indices of positive interactions.
    train_item_nodes     (N,) – item node indices of positive interactions.
    train_ratings        (N,) – rating values (used for compatibility; not
                                used in pure BPR mode).
    config               :class:`TrainConfig`.
    run_name             Unique name for checkpointing sub-directory.
    eval_callback        Optional: fn(model) → dict[str, float].
                         Called after every epoch; result appended to metrics.

    Returns
    -------
    summary : dict with avg_epoch_time_s, max_gpu_mem_mb, best_loss, etc.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device)
    use_amp = config.use_amp and device.type == "cuda"

    model = model.to(device)
    edge_index = edge_index.to(device)
    train_user_nodes = train_user_nodes.to(device)
    train_item_nodes = train_item_nodes.to(device)
    train_ratings = train_ratings.to(device)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    scaler = GradScaler(_AMP_DEVICE, enabled=use_amp)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(config.checkpoint_dir, run_name)
    _ensure_dir(ckpt_dir)

    # Determine log paths (support legacy log_jsonl_path alias)
    metrics_path = config.metrics_jsonl_path or config.log_jsonl_path or "outputs/metrics.jsonl"
    log_txt_path = config.training_log_path or "outputs/training_log.txt"
    _ensure_dir(os.path.dirname(os.path.abspath(metrics_path)))
    _ensure_dir(os.path.dirname(os.path.abspath(log_txt_path)))
    logger = _setup_txt_logger(log_txt_path)

    # ── Auto-resume ───────────────────────────────────────────────────────────
    start_epoch = 1
    latest_ckpt = _find_latest_checkpoint(ckpt_dir)
    if latest_ckpt is not None:
        logger.info(f"[{run_name}] Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        logger.info(f"[{run_name}] Resumed at epoch {start_epoch}")

    # ── Item range for negative sampling ─────────────────────────────────────
    item_min = int(train_item_nodes.min().item())
    item_max = int(train_item_nodes.max().item())
    num_pos = int(train_user_nodes.numel())
    if num_pos == 0:
        raise RuntimeError(f"[{run_name}] No positive training interactions found.")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float("inf")
    best_epoch = start_epoch - 1
    epoch_times: List[float] = []
    peak_mems_mb: List[float] = []
    best_record: Dict = {}
    _no_improve_count: int = 0

    steps_per_epoch = max(1, math.ceil(num_pos / config.batch_size))
    logger.info(
        f"[{run_name}] Training: {num_pos:,} positives | "
        f"batch={config.batch_size:,} | steps/epoch={steps_per_epoch} | "
        f"epochs={config.epochs}"
    )

    for epoch in range(start_epoch, config.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t_start = time.perf_counter()
        model.train()

        # ── Full-pass multi-step epoch ────────────────────────────────────────
        # Shuffle all positives once per epoch, then iterate in batch_size chunks
        # so every interaction is seen exactly once per epoch.
        perm = torch.randperm(num_pos, device=device)

        sum_loss = 0.0
        sum_bpr = 0.0
        sum_reg = 0.0
        nan_steps = 0

        for step in range(steps_per_epoch):
            start_idx = step * config.batch_size
            batch_idx = perm[start_idx: start_idx + config.batch_size]

            pos_u = train_user_nodes[batch_idx]
            pos_i = train_item_nodes[batch_idx]
            pos_r = train_ratings[batch_idx]

            pos_u_rep = pos_u.repeat_interleave(config.neg_ratio)
            pos_i_rep = pos_i.repeat_interleave(config.neg_ratio)
            pos_r_rep = pos_r.repeat_interleave(config.neg_ratio)
            neg_i = sample_negatives(pos_i, item_min, item_max, config.neg_ratio, device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass: GNN runs on the full graph; embeddings are reused
            # across steps within the same epoch via autocast context.
            with autocast(_AMP_DEVICE, enabled=use_amp):
                z = model(edge_index)
                if config.use_rating_weights:
                    loss, bpr_term, reg_term = rating_weighted_bpr_loss(
                        z, pos_u_rep, pos_i_rep, neg_i,
                        pos_ratings=pos_r_rep,
                        emb_l2_weight=config.emb_l2_weight,
                        rating_threshold=config.implicit_threshold,
                    )
                else:
                    loss, bpr_term, reg_term = bpr_loss(
                        z, pos_u_rep, pos_i_rep, neg_i,
                        emb_l2_weight=config.emb_l2_weight,
                    )

            if torch.isnan(loss):
                logger.warning(
                    f"[{run_name}] NaN loss at epoch {epoch} step {step+1} – skipping update."
                )
                nan_steps += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            sum_loss += float(loss.item())
            sum_bpr  += float(bpr_term.item())
            sum_reg  += float(reg_term.item())

        good_steps = steps_per_epoch - nan_steps
        if good_steps == 0:
            logger.warning(f"[{run_name}] All steps NaN at epoch {epoch} – skipping scheduler.")
            continue

        epoch_loss = sum_loss / good_steps
        epoch_bpr  = sum_bpr  / good_steps
        epoch_reg  = sum_reg  / good_steps

        scheduler.step(epoch_loss)

        epoch_time = float(time.perf_counter() - t_start)
        epoch_times.append(epoch_time)
        gpu_mem_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        peak_mems_mb.append(float(gpu_mem_bytes / 1e6))

        record: Dict = {
            "run": run_name,
            "epoch": epoch,
            "loss": epoch_loss,
            "bpr_loss": epoch_bpr,
            "reg_loss": epoch_reg,
            "steps": good_steps,
            "time_s": epoch_time,
            "gpu_mem_mb": float(gpu_mem_bytes / 1e6),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        # ── Evaluation callback ───────────────────────────────────────────────
        if eval_callback is not None:
            try:
                eval_metrics = eval_callback(model)
                record.update(eval_metrics)
            except Exception as exc:
                record["eval_error"] = str(exc)

        _append_jsonl(metrics_path, record)
        logger.info(
            f"[{run_name}] epoch={epoch}/{config.epochs}  "
            f"loss={record['loss']:.5f}  bpr={record['bpr_loss']:.5f}  "
            f"steps={record['steps']}  "
            f"time={epoch_time:.2f}s  mem={record['gpu_mem_mb']:.1f}MB  "
            f"lr={record['lr']:.2e}"
        )

        # ── Per-epoch checkpoint ──────────────────────────────────────────────
        if config.save_epoch_checkpoints:
            ckpt_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "metrics": record,
            }
            torch.save(ckpt_payload, os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))

        # ── Best checkpoint ───────────────────────────────────────────────────
        if epoch_loss < best_loss - config.early_stopping_min_delta:
            best_loss = epoch_loss
            best_epoch = epoch
            best_record = record.copy()
            _no_improve_count = 0
            if config.save_epoch_checkpoints:
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": record},
                    os.path.join(ckpt_dir, "best.pt"),
                )
        else:
            _no_improve_count += 1

        # ── Early stopping ────────────────────────────────────────────────
        if config.early_stopping_patience > 0 and _no_improve_count >= config.early_stopping_patience:
            logger.info(
                f"[{run_name}] Early stopping at epoch {epoch} "
                f"(no improvement for {_no_improve_count} epochs, best_loss={best_loss:.5f})"
            )
            break

    # ── Final model ───────────────────────────────────────────────────────────
    if config.save_epoch_checkpoints:
        final_path = os.path.join(os.path.dirname(ckpt_dir), "final_model.pt")
        torch.save(
            {"run": run_name, "model_state_dict": model.state_dict()},
            final_path,
        )
        logger.info(f"[{run_name}] Final model saved → {final_path}")

    avg_time = float(np.mean(epoch_times)) if epoch_times else 0.0
    max_mem = float(np.max(peak_mems_mb)) if peak_mems_mb else 0.0

    summary: Dict = {
        "best_epoch": float(best_epoch),
        "best_loss": float(best_loss),
        "avg_epoch_time_s": avg_time,
        "max_gpu_mem_mb": max_mem,
        "total_train_time_s": float(sum(epoch_times)),
    }
    # Include best eval metrics (NDCG, Precision, Recall, ...) in summary
    for k, v in best_record.items():
        if k not in {"run", "epoch", "loss", "bpr_loss", "reg_loss", "time_s", "gpu_mem_mb", "lr"}:
            summary[k] = v

    _append_jsonl(
        metrics_path,
        {"type": "summary", "run": run_name, **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in summary.items()}},
    )
    # legacy alias key kept for runner compatibility
    summary["loss_mse"] = best_loss
    return summary
