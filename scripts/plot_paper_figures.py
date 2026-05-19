"""
plot_paper_figures.py
Generates all publication-quality figures for the GSP-based recommender system paper.
Run from the repo root:  python scripts/plot_paper_figures.py
Outputs are saved to figures/ (PNG at 300 DPI + PDF).
"""

import json
import os
import sys
from pathlib import Path
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrowPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "lightgcn": "#4C72B0",
    "gcn":      "#DD8452",
    "gat":      "#55A868",
    "graphsage": "#C44E52",
    "cosine":   "#4C72B0",
    "forman_ricci": "#DD8452",
}
MODEL_LABELS = {
    "lightgcn": "LightGCN",
    "gcn":      "GCN",
    "gat":      "GAT",
    "graphsage": "GraphSAGE",
}
MODELS = ["lightgcn", "gcn", "gat", "graphsage"]
DATASETS = ["ml1m", "yelp"]
DATASET_DIRS = {"ml1m": "sweep_ml1m", "yelp": "sweep_yelp"}
DATASET_LABELS = {"ml1m": "ML-1M", "yelp": "Yelp"}
CURVATURES = ["cosine", "forman_ricci"]
CURVATURE_LABELS = {"cosine": "Cosine", "forman_ricci": "Forman–Ricci"}
FRACS = ["025", "05", "075", "10"]
FRAC_VALS = {"025": 0.25, "05": 0.50, "075": 0.75, "10": 1.00}
MS_VALS = ["1", "3", "5"]


def save(fig, name):
    path_png = FIG_DIR / f"{name}.png"
    path_pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    print(f"  Saved {path_png.name}")
    plt.close(fig)


def load_full_results(sweep_dir, run_dir):
    f = OUT_DIR / sweep_dir / run_dir / "full_results.json"
    if not f.exists():
        return None
    with open(f) as fh:
        return json.load(fh)


def load_gsp_stats(sweep_dir, run_dir):
    f = OUT_DIR / sweep_dir / run_dir / "gsp_stats.json"
    if not f.exists():
        return None
    with open(f) as fh:
        return json.load(fh)


def load_training_metrics(sweep_dir, run_dir, model, run_type):
    f = OUT_DIR / sweep_dir / run_dir / f"training_metrics_{model}_{run_type}.jsonl"
    if not f.exists():
        return []
    rows = []
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("epoch") is not None and rec.get("loss") is not None:
                    rows.append(rec)
            except json.JSONDecodeError:
                pass
    return rows


def get_metric(results_obj, model, run_type, metric):
    if results_obj is None:
        return None
    for rec in results_obj.get("metrics", []):
        if rec["model"] == model and rec["run_type"] in (run_type, f"gsp_{run_type}", run_type):
            return rec.get(metric)
    for rec in results_obj.get("metrics", []):
        if rec["model"] == model:
            if run_type == "gsp" and rec["run_type"] == "gsp_projected":
                return rec.get(metric)
    return None


def get_speedup(results_obj, model, field):
    if results_obj is None:
        return None
    for rec in results_obj.get("speedup", []):
        if rec["model"] == model:
            return rec.get(field)
    return None


def load_sweep_matrix(dataset, curvature, model, metric="NDCG@10", run_type="gsp"):
    """Return (base_matrix, gsp_matrix, speedup_matrix, gpu_red_matrix, pp_time_matrix)
    each shaped (len(FRACS), len(MS_VALS)); NaN where data is missing."""
    sweep_dir = DATASET_DIRS[dataset]
    n_f, n_ms = len(FRACS), len(MS_VALS)
    base_mat  = np.full((n_f, n_ms), np.nan)
    gsp_mat   = np.full((n_f, n_ms), np.nan)
    sf_mat    = np.full((n_f, n_ms), np.nan)   # speedup_factor
    gpu_mat   = np.full((n_f, n_ms), np.nan)   # gpu_reduction_pct
    pp_mat    = np.full((n_f, n_ms), np.nan)   # preprocessing_time_s
    for fi, frac in enumerate(FRACS):
        for mi, ms in enumerate(MS_VALS):
            run_dir = f"{curvature}_frac{frac}_ms{ms}"
            res = load_full_results(sweep_dir, run_dir)
            if res is None:
                continue
            b = get_metric(res, model, "baseline", metric)
            g = get_metric(res, model, run_type,   metric)
            sf  = get_speedup(res, model, "speedup_factor")
            gr  = get_speedup(res, model, "gpu_reduction_pct")
            pp  = get_speedup(res, model, "gsp_preprocessing_s")
            if b is not None: base_mat[fi, mi] = b
            if g is not None: gsp_mat[fi, mi]  = g
            if sf is not None: sf_mat[fi, mi]  = sf
            if gr is not None: gpu_mat[fi, mi] = gr
            if pp is not None: pp_mat[fi, mi]  = pp
    return base_mat, gsp_mat, sf_mat, gpu_mat, pp_mat


# ---------------------------------------------------------------------------
# Figure 1: Pipeline Workflow Diagram
# ---------------------------------------------------------------------------
def fig_pipeline():
    print("Figure 1: Pipeline diagram...")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    def box(cx, cy, w, h, label, sublabel=None, color="#4C72B0", textcolor="white"):
        rect = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(cx, cy + (0.18 if sublabel else 0), label,
                ha="center", va="center", fontsize=9, color=textcolor, fontweight="bold")
        if sublabel:
            ax.text(cx, cy - 0.22, sublabel, ha="center", va="center",
                    fontsize=7.5, color=textcolor, style="italic")

    def arrow(x0, x1, y):
        ax.annotate("", xy=(x1 - 0.02, y), xytext=(x0 + 0.02, y),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.5))

    ys = 2.25  # single row y
    boxes = [
        (0.85,  ys, 1.5, 1.2, "Raw Interaction\nData", "(users × items)", "#6c757d"),
        (2.85,  ys, 1.5, 1.2, "UU Graph\nConstruction", "(cosine / Forman–Ricci)", "#2166ac"),
        (4.85,  ys, 1.5, 1.2, "HEM Clustering", "(hierarchy,\nmin_shared k)", "#1a7837"),
        (6.85,  ys, 1.5, 1.2, "Super-node\nSparsification", "(effective resistance)", "#7b2d8b"),
        (8.85,  ys, 1.5, 1.2, "GNN Training", "(LightGCN / GCN /\nGAT / GraphSAGE)", "#b2182b"),
        (10.9,  ys, 1.5, 1.2, "Projection &\nEvaluation", "(NDCG,\nHitRate)", "#762a83"),
    ]
    xs = [b[0] for b in boxes]
    for bx, by, bw, bh, lbl, sub, col in boxes:
        box(bx, by, bw, bh, lbl, sub, col)

    for i in range(len(xs) - 1):
        arrow(xs[i] + 0.75, xs[i + 1] - 0.75, ys)

    # Stage labels below
    stage_y = 0.8
    stage_texts = [
        (xs[0], "Stage 0:\nData Prep"),
        (xs[1], "Stage 1:\nCurvature"),
        (xs[2], "Stage 2:\nClustering"),
        (xs[3], "Stage 3:\nSparsification"),
        (xs[4], "Stage 4:\nTraining"),
        (xs[5], "Stage 5:\nInference"),
    ]
    for stx, stxt in stage_texts:
        ax.text(stx, stage_y, stxt, ha="center", va="center",
                fontsize=7.5, color="#444444", style="italic")

    # GSP brace
    brace_y = 3.95
    ax.annotate("", xy=(xs[3] + 0.75, brace_y), xytext=(xs[1] - 0.75, brace_y),
                arrowprops=dict(arrowstyle="-", color="#2166ac", lw=2))
    ax.text((xs[1] + xs[3]) / 2, brace_y + 0.2, "GSP Graph Compression",
            ha="center", va="bottom", fontsize=9, color="#2166ac", fontweight="bold")

    ax.set_title("GSP-Enhanced GNN Recommendation Pipeline", fontsize=12, fontweight="bold", pad=12)
    save(fig, "fig01_pipeline")


# ---------------------------------------------------------------------------
# Figure 2: NDCG@10 Baseline vs GSP — Grouped Bar Chart
# ---------------------------------------------------------------------------
def fig_ndcg_comparison():
    print("Figure 2: NDCG@10 comparison bar chart...")

    # Hard-coded from collected data (cosine_frac10_ms1)
    data = {
        "ml1m": {
            "lightgcn": {"baseline": 0.0804, "gsp": 0.0829},
            "gcn":       {"baseline": 0.3468, "gsp": 0.3533},
            "gat":       {"baseline": 0.0888, "gsp": 0.0911},
            "graphsage": {"baseline": 0.3548, "gsp": 0.3506},
        },
        "yelp": {
            "lightgcn": {"baseline": 0.0945, "gsp": 0.0951},
            "gcn":       {"baseline": 0.5950, "gsp": 0.5760},
            "gat":       {"baseline": 0.4426, "gsp": 0.5237},
            "graphsage": {"baseline": 0.5887, "gsp": 0.5785},
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        x = np.arange(len(MODELS))
        width = 0.35
        base_vals = [data[ds][m]["baseline"] for m in MODELS]
        gsp_vals  = [data[ds][m]["gsp"]      for m in MODELS]
        delta_pct = [(g - b) / b * 100 if b > 0 else 0 for b, g in zip(base_vals, gsp_vals)]

        bars_b = ax.bar(x - width / 2, base_vals, width, label="Baseline",
                        color="#4C72B0", alpha=0.85, edgecolor="white")
        bars_g = ax.bar(x + width / 2, gsp_vals,  width, label="GSP",
                        color="#DD8452", alpha=0.85, edgecolor="white")

        # Delta annotations
        for xi, (b, g, dp) in enumerate(zip(base_vals, gsp_vals, delta_pct)):
            top = max(b, g)
            color = "#2ca02c" if dp >= 0 else "#d62728"
            ax.text(xi, top + 0.005, f"{dp:+.1f}%", ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=15, ha="right")
        ax.set_ylabel("NDCG@10")
        ax.set_title(DATASET_LABELS[ds])
        ax.legend(framealpha=0.8)
        ax.set_ylim(0, max(max(base_vals), max(gsp_vals)) * 1.18)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("NDCG@10: Baseline vs GSP (cosine, frac=1.0, ms=1)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig02_ndcg10_comparison")


# ---------------------------------------------------------------------------
# Figure 3: Multi-metric comparison (NDCG@10/20/50, Recall, HitRate) — Yelp
# ---------------------------------------------------------------------------
def fig_multi_metric_yelp():
    print("Figure 3: Multi-metric bar chart (Yelp)...")

    full_data = {
        "lightgcn": {
            "baseline": {"NDCG@10": 0.0945, "NDCG@20": 0.1227, "NDCG@50": 0.1772,
                         "Recall@10": 0.1769, "HitRate@10": 0.1769},
            "gsp":      {"NDCG@10": 0.0951, "NDCG@20": 0.1235, "NDCG@50": 0.1789,
                         "Recall@10": 0.1800, "HitRate@10": 0.1800},
        },
        "gcn": {
            "baseline": {"NDCG@10": 0.5950, "NDCG@20": 0.6102, "NDCG@50": 0.6152,
                         "Recall@10": 0.9125, "HitRate@10": 0.9125},
            "gsp":      {"NDCG@10": 0.5760, "NDCG@20": 0.5922, "NDCG@50": 0.5977,
                         "Recall@10": 0.9052, "HitRate@10": 0.9052},
        },
        "gat": {
            "baseline": {"NDCG@10": 0.4426, "NDCG@20": 0.4718, "NDCG@50": 0.4816,
                         "Recall@10": 0.8296, "HitRate@10": 0.8296},
            "gsp":      {"NDCG@10": 0.5237, "NDCG@20": 0.5450, "NDCG@50": 0.5523,
                         "Recall@10": 0.8743, "HitRate@10": 0.8743},
        },
        "graphsage": {
            "baseline": {"NDCG@10": 0.5887, "NDCG@20": 0.6039, "NDCG@50": 0.6093,
                         "Recall@10": 0.9096, "HitRate@10": 0.9096},
            "gsp":      {"NDCG@10": 0.5785, "NDCG@20": 0.5946, "NDCG@50": 0.5999,
                         "Recall@10": 0.9064, "HitRate@10": 0.9064},
        },
    }

    metrics = ["NDCG@10", "NDCG@20", "NDCG@50", "Recall@10"]
    metric_labels = ["NDCG@10", "NDCG@20", "NDCG@50", "Recall@10"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, m_key, m_lbl in zip(axes, metrics, metric_labels):
        x = np.arange(len(MODELS))
        width = 0.35
        bvals = [full_data[m]["baseline"][m_key] for m in MODELS]
        gvals = [full_data[m]["gsp"][m_key]      for m in MODELS]

        ax.bar(x - width / 2, bvals, width, label="Baseline",
               color="#4C72B0", alpha=0.85, edgecolor="white")
        ax.bar(x + width / 2, gvals, width, label="GSP",
               color="#DD8452", alpha=0.85, edgecolor="white")

        for xi, (b, g) in enumerate(zip(bvals, gvals)):
            dp = (g - b) / b * 100 if b > 0 else 0
            color = "#2ca02c" if dp >= 0 else "#d62728"
            ax.text(xi, max(b, g) + 0.01, f"{dp:+.1f}%",
                    ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=15, ha="right")
        ax.set_ylabel(m_lbl)
        ax.set_title(f"Yelp — {m_lbl}")
        ax.legend(framealpha=0.8)
        ax.set_ylim(0, max(max(bvals), max(gvals)) * 1.18)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Yelp: Multi-metric Comparison — Baseline vs GSP",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig03_multi_metric_yelp")


# ---------------------------------------------------------------------------
# Figure 4: Training Loss Curves — ML-1M
# ---------------------------------------------------------------------------
def fig_loss_curves_ml1m():
    print("Figure 4: Training loss curves (ML-1M)...")
    sweep_dir = "sweep_ml1m"
    run_dir   = "cosine_frac10_ms1"

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, model in zip(axes, MODELS):
        for rt, style, label, col in [("baseline", "-", "Baseline", "#4C72B0"),
                                       ("gsp",      "--", "GSP",      "#DD8452")]:
            rows = load_training_metrics(sweep_dir, run_dir, model, rt)
            if not rows:
                continue
            epochs = [r["epoch"] for r in rows]
            losses = [r["loss"]  for r in rows]
            ax.plot(epochs, losses, linestyle=style, color=col, linewidth=1.8, label=label)

        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BPR Loss")
        ax.legend(framealpha=0.8)
        ax.grid(linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Training Loss Curves — ML-1M (cosine, frac=1.0, ms=1)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig04_loss_curves_ml1m")


# ---------------------------------------------------------------------------
# Figure 5: Training Loss Curves — Yelp
# ---------------------------------------------------------------------------
def fig_loss_curves_yelp():
    print("Figure 5: Training loss curves (Yelp)...")
    sweep_dir = "sweep_yelp"
    run_dir   = "cosine_frac10_ms1"

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, model in zip(axes, MODELS):
        for rt, style, label, col in [("baseline", "-", "Baseline", "#4C72B0"),
                                       ("gsp",      "--", "GSP",      "#DD8452")]:
            rows = load_training_metrics(sweep_dir, run_dir, model, rt)
            if not rows:
                continue
            epochs = [r["epoch"] for r in rows]
            losses = [r["loss"]  for r in rows]
            ax.plot(epochs, losses, linestyle=style, color=col, linewidth=1.8, label=label)

        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BPR Loss")
        ax.legend(framealpha=0.8)
        ax.grid(linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Training Loss Curves — Yelp (cosine, frac=1.0, ms=1)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig05_loss_curves_yelp")



# ---------------------------------------------------------------------------
# Helper: draw one "cosine vs forman-ricci" loss-curve panel (1×4 grid)
# ---------------------------------------------------------------------------
def _loss_curvature_panel(b_sw, b_rd, c_sw, c_rd, f_sw, f_rd, title, figname, note=""):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, model in zip(axes, MODELS):
        b_rows = load_training_metrics(b_sw, b_rd, model, "baseline")
        if b_rows:
            ax.plot([r["epoch"] for r in b_rows], [r["loss"] for r in b_rows],
                    linestyle=":", color="#888888", linewidth=1.4, label="Baseline", zorder=2)
        c_rows = load_training_metrics(c_sw, c_rd, model, "gsp")
        if c_rows:
            ax.plot([r["epoch"] for r in c_rows], [r["loss"] for r in c_rows],
                    linestyle="-", color=COLORS["cosine"], linewidth=1.8, label="Cosine", zorder=3)
        fr_rows = load_training_metrics(f_sw, f_rd, model, "gsp")
        if fr_rows:
            ax.plot([r["epoch"] for r in fr_rows], [r["loss"] for r in fr_rows],
                    linestyle="--", color=COLORS["forman_ricci"], linewidth=1.8,
                    label="Forman–Ricci", zorder=3)
        ax.set_title(MODEL_LABELS[model], fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        if model == "lightgcn":
            ax.set_ylabel("BPR Loss")
        if note:
            ax.text(0.99, 0.97, note, transform=ax.transAxes,
                    ha="right", va="top", fontsize=6, color="#555555",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.85))
        ax.legend(framealpha=0.8, fontsize=8)
        ax.grid(linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, figname)


# ---------------------------------------------------------------------------
# Figure 5b: Loss curves — Cosine vs Forman-Ricci — ML-1M (frac=1.0, ms=1)
# ---------------------------------------------------------------------------
def fig_loss_curves_cosine_vs_forman_ml1m():
    print("Figure 5b: Loss curves cosine vs forman-ricci — ML-1M...")
    _loss_curvature_panel(
        b_sw="sweep_ml1m", b_rd="cosine_frac10_ms1",
        c_sw="sweep_ml1m", c_rd="cosine_frac10_ms1",
        f_sw="sweep_ml1m", f_rd="forman_ricci_frac10_ms1",
        title="ML-1M — BPR Training Loss: Cosine vs Forman–Ricci (frac=1.0, ms=1)",
        figname="fig05b_loss_curves_cosine_forman_ml1m",
    )


# ---------------------------------------------------------------------------
# Figure 5c: Loss curves — Cosine vs Forman-Ricci — Yelp (frac=1.0, ms=1)
# ---------------------------------------------------------------------------
def fig_loss_curves_cosine_vs_forman_yelp():
    print("Figure 5c: Loss curves cosine vs forman-ricci — Yelp...")
    _loss_curvature_panel(
        b_sw="sweep_yelp", b_rd="cosine_frac10_ms1",
        c_sw="sweep_yelp", c_rd="cosine_frac10_ms1",
        f_sw="sweep_yelp", f_rd="forman_ricci_frac10_ms1",
        title="Yelp — BPR Training Loss: Cosine vs Forman–Ricci (frac=1.0, ms=1)",
        figname="fig05c_loss_curves_cosine_forman_yelp",
    )


# ---------------------------------------------------------------------------
# Figure 5d: Loss curves — Cosine vs Forman-Ricci — ML-25M
# cosine  → sweep_ml25m/cosine_frac05_ms1        (best complete cosine run)
# forman  → sweep_ml25m_ordered/forman_ricci_frac10_ms1
# ---------------------------------------------------------------------------
def fig_loss_curves_cosine_vs_forman_ml25m():
    print("Figure 5d: Loss curves cosine vs forman-ricci — ML-25M...")
    _loss_curvature_panel(
        b_sw="sweep_ml25m",         b_rd="cosine_frac05_ms1",
        c_sw="sweep_ml25m",         c_rd="cosine_frac05_ms1",
        f_sw="sweep_ml25m_ordered", f_rd="forman_ricci_frac10_ms1",
        title="ML-25M — BPR Training Loss: Cosine vs Forman–Ricci",
        figname="fig05d_loss_curves_cosine_forman_ml25m",
        note="cosine: frac=0.5 ms=1  |  forman: frac=1.0 ms=1",
    )

# ---------------------------------------------------------------------------
# Figure 6: Training Time Breakdown (stacked bar) & GPU Memory — Both Datasets
# Shows baseline training vs [GSP preprocessing + GSP training], speedup factor annotated
# Data loaded dynamically from full_results.json (cosine_frac10_ms1)
# ---------------------------------------------------------------------------
def fig_resources():
    print("Figure 6: Training time breakdown & GPU memory...")

    # Load data from cosine_frac10_ms1 (representative full-fraction config)
    time_data = {"ml1m": {}, "yelp": {}}
    gpu_data  = {"ml1m": {}, "yelp": {}}
    for ds in DATASETS:
        run_dir = "cosine_frac10_ms1"
        res = load_full_results(DATASET_DIRS[ds], run_dir)
        for model in MODELS:
            t_base  = get_speedup(res, model, "training_time_baseline_s") or 0.0
            t_train = get_speedup(res, model, "training_time_gsp_s")      or 0.0
            t_pp    = get_speedup(res, model, "gsp_preprocessing_s")      or 0.0
            g_base  = get_speedup(res, model, "gpu_baseline_MB")          or 0.0
            g_gsp   = get_speedup(res, model, "gpu_gsp_MB")               or 0.0
            time_data[ds][model] = {"t_base": t_base, "t_train": t_train, "t_pp": t_pp}
            gpu_data[ds][model]  = {"base": g_base, "gsp": g_gsp}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for col_idx, ds in enumerate(DATASETS):
        x      = np.arange(len(MODELS))
        width  = 0.35
        mlabels = [MODEL_LABELS[m] for m in MODELS]

        # ── Top row: stacked time bars ──
        ax_t = axes[0][col_idx]
        t_base  = np.array([time_data[ds][m]["t_base"]  for m in MODELS])
        t_train = np.array([time_data[ds][m]["t_train"] for m in MODELS])
        t_pp    = np.array([time_data[ds][m]["t_pp"]    for m in MODELS])

        ax_t.bar(x - width / 2, t_base, width,
                 label="Baseline training", color="#4C72B0", alpha=0.85, edgecolor="white")
        ax_t.bar(x + width / 2, t_train, width,
                 label="GSP training",     color="#DD8452", alpha=0.85, edgecolor="white")
        ax_t.bar(x + width / 2, t_pp, width, bottom=t_train,
                 label="GSP preprocessing", color="#55A868", alpha=0.7,
                 edgecolor="white", hatch="///")

        # Speedup factor annotation above each group
        y_top = max(t_base.max(), (t_train + t_pp).max()) * 1.04
        for xi, (tb, tt) in enumerate(zip(t_base, t_train)):
            sf = tb / tt if tt > 0 else 1.0
            col = "#2ca02c" if sf >= 1.02 else ("#d62728" if sf < 0.98 else "#555555")
            ax_t.text(xi, y_top, f"{sf:.2f}×",
                      ha="center", va="bottom", fontsize=8, color=col, fontweight="bold")

        ax_t.set_xticks(x)
        ax_t.set_xticklabels(mlabels, rotation=15, ha="right")
        ax_t.set_ylabel("Time (s)")
        ax_t.set_title(f"{DATASET_LABELS[ds]} — Training Time + Preprocessing\n"
                       f"(×  = training-only speedup; frac=1.0, ms=1)")
        ax_t.legend(framealpha=0.8, fontsize=8)
        ax_t.grid(axis="y", linestyle="--", alpha=0.4)
        ax_t.spines[["top", "right"]].set_visible(False)

        # ── Bottom row: GPU memory bars ──
        ax_g = axes[1][col_idx]
        g_base = np.array([gpu_data[ds][m]["base"] for m in MODELS])
        g_gsp  = np.array([gpu_data[ds][m]["gsp"]  for m in MODELS])

        ax_g.bar(x - width / 2, g_base, width, label="Baseline",
                 color="#4C72B0", alpha=0.85, edgecolor="white")
        ax_g.bar(x + width / 2, g_gsp, width, label="GSP",
                 color="#DD8452", alpha=0.85, edgecolor="white")

        for xi, (gb, gg) in enumerate(zip(g_base, g_gsp)):
            dr = (gb - gg) / gb * 100 if gb > 0 else 0
            col = "#2ca02c" if dr >= 0.5 else ("#d62728" if dr < -0.5 else "#555555")
            ax_g.text(xi, max(gb, gg) * 1.01, f"{dr:+.1f}%",
                      ha="center", va="bottom", fontsize=7.5, color=col, fontweight="bold")

        ax_g.set_xticks(x)
        ax_g.set_xticklabels(mlabels, rotation=15, ha="right")
        ax_g.set_ylabel("Peak GPU Memory (MB)")
        ax_g.set_title(f"{DATASET_LABELS[ds]} — Peak GPU Memory")
        ax_g.legend(framealpha=0.8)
        ax_g.grid(axis="y", linestyle="--", alpha=0.4)
        ax_g.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Computational Cost: Training Time (with preprocessing) & GPU Memory",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig06_resources")


# ---------------------------------------------------------------------------
# Figure 7a–7d: Sweep Landscape — all 24 conditions (4 frac × 3 ms)
# One figure per (dataset × curvature).
# Layout: 4 rows (fraction) × 3 cols (min_shared) = 12 subplots.
# Each subplot: 4 bars (one per GNN model) showing NDCG@10 Δ% vs baseline.
# Fully data-driven.
# ---------------------------------------------------------------------------
def _make_sweep_landscape_fig(dataset, curvature, metric="NDCG@10"):
    n_f, n_ms = len(FRACS), len(MS_VALS)
    fig, axes = plt.subplots(n_f, n_ms, figsize=(13, 12), sharey=True)

    x     = np.arange(len(MODELS))
    width = 0.62
    short = ["LightGCN", "GCN", "GAT", "GraphSAGE"]

    # ── first pass: load all data and compute global y-limits ──
    cell_data = {}
    all_deltas = []
    all_base   = {}          # (fi, mi, model) → baseline value
    for fi, frac in enumerate(FRACS):
        for mi, ms in enumerate(MS_VALS):
            run_dir = f"{curvature}_frac{frac}_ms{ms}"
            res     = load_full_results(DATASET_DIRS[dataset], run_dir)
            deltas  = []
            bases   = []
            for model in MODELS:
                b = get_metric(res, model, "baseline", metric) if res else None
                g = get_metric(res, model, "gsp",      metric) if res else None
                if b and g and b > 0:
                    d = (g - b) / b * 100
                    deltas.append(d)
                    bases.append(b)
                    all_deltas.append(d)
                else:
                    deltas.append(np.nan)
                    bases.append(np.nan)
            cell_data[(fi, mi)] = (deltas, bases)

    spread = max(abs(d) for d in all_deltas) if all_deltas else 5.0
    ylim   = max(spread * 1.35, 2.0)

    # ── second pass: draw ──
    for fi, frac in enumerate(FRACS):
        for mi, ms in enumerate(MS_VALS):
            ax             = axes[fi][mi]
            deltas, bases  = cell_data[(fi, mi)]
            model_colors   = [COLORS[m] for m in MODELS]

            bars = ax.bar(x,
                          [d if not np.isnan(d) else 0 for d in deltas],
                          width, color=model_colors, alpha=0.85,
                          edgecolor="white", zorder=3)

            # Hatch bars where GSP is worse than baseline
            for bar, d in zip(bars, deltas):
                if not np.isnan(d) and d < 0:
                    bar.set_hatch("///")
                    bar.set_edgecolor("#555555")
                    bar.set_alpha(0.7)

            ax.axhline(0, color="#333333", linewidth=0.9, zorder=2)
            ax.set_ylim(-ylim, ylim)
            ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
            ax.spines[["top", "right"]].set_visible(False)

            # ── column header (top row only) ──
            if fi == 0:
                ax.set_title(f"min_shared = {ms}", fontsize=10,
                             fontweight="bold", pad=6)

            # ── row label (leftmost col only) ──
            if mi == 0:
                ax.set_ylabel(f"frac = {FRAC_VALS[frac]:.2f}\n{metric} Δ%",
                              fontsize=8.5, fontweight="bold")
            else:
                ax.set_ylabel("")

            # ── x-tick labels: bottom row only ──
            ax.set_xticks(x)
            if fi == n_f - 1:
                ax.set_xticklabels(["LG", "GCN", "GAT", "SGE"],
                                   fontsize=8, rotation=0)
            else:
                ax.set_xticklabels([])

            # ── value annotations ──
            for xi, (d, b) in enumerate(zip(deltas, bases)):
                if np.isnan(d):
                    ax.text(xi, 0, "N/A", ha="center", va="center",
                            fontsize=6.5, color="#aaaaaa")
                    continue
                va  = "bottom" if d >= 0 else "top"
                off = ylim * 0.05 if d >= 0 else -ylim * 0.05
                col = "#006400" if d > 0.5 else ("#8B0000" if d < -0.5 else "#444444")
                ax.text(xi, d + off, f"{d:+.1f}%",
                        ha="center", va=va, fontsize=6.5,
                        color=col, fontweight="bold" if abs(d) > 1.5 else "normal")

    # ── legend ──
    legend_patches = [
        mpatches.Patch(color=COLORS[m], label=MODEL_LABELS[m]) for m in MODELS
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=4,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.025))

    ds_label   = DATASET_LABELS[dataset]
    curv_label = CURVATURE_LABELS[curvature]
    fig.suptitle(
        f"{ds_label}  —  {curv_label} Curvature\n"
        f"{metric} Δ% (GSP vs Baseline) across all 24 sweep conditions "
        f"(4 fractions × 3 min-shared)",
        fontsize=11, fontweight="bold", y=1.06,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def fig_sweep_heatmaps():
    """Figures 7a–7d: sweep landscape (4 frac × 3 ms grid, all models per cell)."""
    combos = [
        ("ml1m", "cosine",       "fig07a_sweep_ml1m_cosine"),
        ("ml1m", "forman_ricci", "fig07b_sweep_ml1m_forman"),
        ("yelp", "cosine",       "fig07c_sweep_yelp_cosine"),
        ("yelp", "forman_ricci", "fig07d_sweep_yelp_forman"),
    ]
    for dataset, curvature, fname in combos:
        print(f"  Figure {fname}...")
        fig = _make_sweep_landscape_fig(dataset, curvature)
        save(fig, fname)


# ---------------------------------------------------------------------------
# Figure 9a–9d: Speedup Landscape — all 24 conditions (4 frac × 3 ms)
# Layout: same 4×3 grid.
# Each subplot: 4 bars (one per GNN model) showing training speedup_factor.
# y = 1.0 reference (= no speedup).  Preprocessing time shown per cell.
# Fully data-driven.
# ---------------------------------------------------------------------------
def _make_speedup_landscape_fig(dataset, curvature):
    n_f, n_ms = len(FRACS), len(MS_VALS)
    fig, axes = plt.subplots(n_f, n_ms, figsize=(13, 12), sharey=True)

    x     = np.arange(len(MODELS))
    width = 0.62

    # ── first pass: load all data ──
    cell_data  = {}
    all_sf     = []
    for fi, frac in enumerate(FRACS):
        for mi, ms in enumerate(MS_VALS):
            run_dir = f"{curvature}_frac{frac}_ms{ms}"
            res     = load_full_results(DATASET_DIRS[dataset], run_dir)
            sfs, gpus, pp_time = [], [], np.nan
            for model in MODELS:
                sf = get_speedup(res, model, "speedup_factor")    if res else None
                gr = get_speedup(res, model, "gpu_reduction_pct") if res else None
                pp = get_speedup(res, model, "gsp_preprocessing_s") if res else None
                sfs.append(sf  if sf  is not None else np.nan)
                gpus.append(gr if gr is not None else np.nan)
                if pp is not None and np.isnan(pp_time):
                    pp_time = pp
            cell_data[(fi, mi)] = (sfs, gpus, pp_time)
            all_sf.extend([v for v in sfs if not np.isnan(v)])

    spread = max(abs(v - 1.0) for v in all_sf) if all_sf else 0.2
    ylim_lo = 1.0 - max(spread * 1.4, 0.15)
    ylim_hi = 1.0 + max(spread * 1.4, 0.15)

    # ── second pass: draw ──
    for fi, frac in enumerate(FRACS):
        for mi, ms in enumerate(MS_VALS):
            ax              = axes[fi][mi]
            sfs, gpus, pp_t = cell_data[(fi, mi)]

            bar_colors = [COLORS[m] for m in MODELS]

            bars = ax.bar(x,
                          [sf if not np.isnan(sf) else 1.0 for sf in sfs],
                          width, color=bar_colors, alpha=0.85,
                          edgecolor="white", zorder=3)

            # Hatch bars where speedup < 1.0 (GSP is slower)
            for bar, sf in zip(bars, sfs):
                if not np.isnan(sf) and sf < 1.0:
                    bar.set_hatch("///")
                    bar.set_edgecolor("#555555")
                    bar.set_alpha(0.7)

            ax.axhline(1.0, color="#333333", linewidth=0.9, zorder=2)
            ax.set_ylim(ylim_lo, ylim_hi)
            ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
            ax.spines[["top", "right"]].set_visible(False)

            # Preprocessing time banner at top of cell
            if not np.isnan(pp_t):
                ax.text(0.5, 0.98, f"pp: {pp_t:.1f}s",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=6.5, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="#bbbbbb", alpha=0.8))

            # ── column header ──
            if fi == 0:
                ax.set_title(f"min_shared = {ms}", fontsize=10,
                             fontweight="bold", pad=6)

            # ── row label ──
            if mi == 0:
                ax.set_ylabel(f"frac = {FRAC_VALS[frac]:.2f}\nSpeedup ×",
                              fontsize=8.5, fontweight="bold")
            else:
                ax.set_ylabel("")

            # ── x-tick labels ──
            ax.set_xticks(x)
            if fi == n_f - 1:
                ax.set_xticklabels(["LG", "GCN", "GAT", "SGE"],
                                   fontsize=8, rotation=0)
            else:
                ax.set_xticklabels([])

            # ── value + GPU annotations ──
            for xi, (sf, gr) in enumerate(zip(sfs, gpus)):
                if np.isnan(sf):
                    ax.text(xi, 1.0, "N/A", ha="center", va="bottom",
                            fontsize=6, color="#aaaaaa")
                    continue
                above = sf >= 1.0
                va  = "bottom" if above else "top"
                off = (ylim_hi - 1.0) * 0.06 if above else -(1.0 - ylim_lo) * 0.06
                col = "#006400" if sf > 1.05 else ("#8B0000" if sf < 0.95 else "#444444")
                ax.text(xi, sf + off, f"{sf:.3f}×",
                        ha="center", va=va, fontsize=6.5,
                        color=col, fontweight="bold" if abs(sf - 1.0) > 0.05 else "normal")
                # GPU % below bar
                if not np.isnan(gr):
                    ax.text(xi, ylim_lo + (1.0 - ylim_lo) * 0.05,
                            f"{gr:+.1f}%",
                            ha="center", va="bottom", fontsize=5.5,
                            color="#888888")

    # ── legend ──
    legend_patches = [
        mpatches.Patch(color=COLORS[m], label=MODEL_LABELS[m]) for m in MODELS
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=4,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.025))

    ds_label   = DATASET_LABELS[dataset]
    curv_label = CURVATURE_LABELS[curvature]
    fig.suptitle(
        f"{ds_label}  —  {curv_label} Curvature\n"
        f"Training Speedup Factor (baseline / GSP epoch time) + pp time + GPU Δ% "
        f"across all 24 sweep conditions",
        fontsize=11, fontweight="bold", y=1.06,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def fig_speedup_heatmaps():
    """Figures 9a–9d: speedup landscape (4 frac × 3 ms grid, all models per cell)."""
    combos = [
        ("ml1m", "cosine",       "fig09a_speedup_ml1m_cosine"),
        ("ml1m", "forman_ricci", "fig09b_speedup_ml1m_forman"),
        ("yelp", "cosine",       "fig09c_speedup_yelp_cosine"),
        ("yelp", "forman_ricci", "fig09d_speedup_yelp_forman"),
    ]
    for dataset, curvature, fname in combos:
        print(f"  Figure {fname}...")
        fig = _make_speedup_landscape_fig(dataset, curvature)
        save(fig, fname)


# ---------------------------------------------------------------------------
# Figure 8: Yelp NDCG @ multiple K — per model
# ---------------------------------------------------------------------------
def fig_ndcg_at_k():
    print("Figure 8: NDCG@K line plots (Yelp)...")

    ks = [10, 20, 50]
    ndcg_data = {
        "lightgcn": {
            "baseline": [0.0945, 0.1227, 0.1772],
            "gsp":      [0.0951, 0.1235, 0.1789],
        },
        "gcn": {
            "baseline": [0.5950, 0.6102, 0.6152],
            "gsp":      [0.5760, 0.5922, 0.5977],
        },
        "gat": {
            "baseline": [0.4426, 0.4718, 0.4816],
            "gsp":      [0.5237, 0.5450, 0.5523],
        },
        "graphsage": {
            "baseline": [0.5887, 0.6039, 0.6093],
            "gsp":      [0.5785, 0.5946, 0.5999],
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, model in zip(axes, MODELS):
        d = ndcg_data[model]
        ax.plot(ks, d["baseline"], "o-", color="#4C72B0", linewidth=2, markersize=6, label="Baseline")
        ax.plot(ks, d["gsp"],      "s--", color="#DD8452", linewidth=2, markersize=6, label="GSP")
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("K")
        ax.set_ylabel("NDCG@K")
        ax.set_xticks(ks)
        ax.legend(framealpha=0.8)
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("NDCG@K — Yelp (cosine, frac=1.0, ms=1)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig08_ndcg_at_k_yelp")


# ---------------------------------------------------------------------------
# Figure 10: GSP Preprocessing Time & Compression vs Fraction — ML-1M
# Data-driven: reads from full_results.json speedup.gsp_preprocessing_s
# and gsp_stats.json (super_nodes / num_super_nodes), averaged over min_shared
# ---------------------------------------------------------------------------
def fig_gsp_stats():
    print("Figure 10: GSP stats (preprocessing time + super-nodes)...")

    frac_nums = [FRAC_VALS[f] for f in FRACS]
    n_original_ml1m = 6040

    # Build (curv → frac → {pp_time, super_nodes}) averaged over MS_VALS
    stats = {}
    for curv in CURVATURES:
        stats[curv] = {}
        for frac in FRACS:
            pp_times, super_nodes = [], []
            for ms in MS_VALS:
                run_dir = f"{curv}_frac{frac}_ms{ms}"
                # Preprocessing time from speedup array in full_results.json
                res = load_full_results(DATASET_DIRS["ml1m"], run_dir)
                if res:
                    for sp in res.get("speedup", []):
                        v = sp.get("gsp_preprocessing_s")
                        if v is not None:
                            pp_times.append(v)
                            break  # same for all models in a run
                # Super-nodes from gsp_stats.json
                d = load_gsp_stats(DATASET_DIRS["ml1m"], run_dir)
                if d:
                    sn = d.get("num_super_nodes", d.get("n_super_nodes"))
                    if sn is not None:
                        super_nodes.append(sn)
            stats[curv][frac] = {
                "pp_time":    np.nanmean(pp_times)    if pp_times    else np.nan,
                "super_nodes": np.nanmean(super_nodes) if super_nodes else np.nan,
            }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for curv, style, label in [("cosine", "o-", "Cosine"), ("forman_ricci", "s--", "Forman–Ricci")]:
        pp_vals = [stats[curv][f]["pp_time"]    for f in FRACS]
        sn_vals = [stats[curv][f]["super_nodes"] for f in FRACS]
        axes[0].plot(frac_nums, pp_vals, style, color=COLORS[curv],
                     linewidth=2, markersize=6, label=label)
        axes[1].plot(frac_nums, sn_vals, style, color=COLORS[curv],
                     linewidth=2, markersize=6, label=label)

    axes[0].set_xlabel("Fraction of UU Edges Retained")
    axes[0].set_ylabel("GSP Preprocessing Time (s, avg over min-shared)")
    axes[0].set_title("ML-1M — GSP Preprocessing Time")
    axes[0].legend(framealpha=0.8)
    axes[0].grid(linestyle="--", alpha=0.4)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].axhline(n_original_ml1m, linestyle=":", color="#aaaaaa", linewidth=1.2,
                    label=f"Original users ({n_original_ml1m:,})")
    axes[1].set_xlabel("Fraction of UU Edges Retained")
    axes[1].set_ylabel("Number of Super-nodes (avg over min-shared)")
    axes[1].set_title("ML-1M — Graph Compression (Super-nodes)")
    axes[1].legend(framealpha=0.8)
    axes[1].grid(linestyle="--", alpha=0.4)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("GSP Preprocessing: Time and Compression vs Edge Fraction (ML-1M)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig10_gsp_stats")


# ---------------------------------------------------------------------------
# Figure 11: Combined Radar / Spider Chart — All models × metrics (Yelp)
# ---------------------------------------------------------------------------
def fig_radar_yelp():
    print("Figure 11: Radar chart (Yelp)...")

    categories = ["NDCG@10", "NDCG@20", "NDCG@50", "Recall@10", "Recall@20"]
    N = len(categories)

    values = {
        "LightGCN (base)": [0.0945, 0.1227, 0.1772, 0.1769, 0.2891],
        "LightGCN (GSP)":  [0.0951, 0.1235, 0.1789, 0.1800, 0.2936],
        "GCN (base)":      [0.5950, 0.6102, 0.6152, 0.9125, 0.9714],
        "GCN (GSP)":       [0.5760, 0.5922, 0.5977, 0.9052, 0.9681],
        "GAT (base)":      [0.4426, 0.4718, 0.4816, 0.8296, 0.9434],
        "GAT (GSP)":       [0.5237, 0.5450, 0.5523, 0.8743, 0.9568],
        "GraphSAGE (base)":[0.5887, 0.6039, 0.6093, 0.9096, 0.9685],
        "GraphSAGE (GSP)": [0.5785, 0.5946, 0.5999, 0.9064, 0.9689],
    }

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    palette = ["#4C72B0", "#4C72B0",
               "#DD8452", "#DD8452",
               "#55A868", "#55A868",
               "#C44E52", "#C44E52"]
    line_styles = ["-", "--", "-", "--", "-", "--", "-", "--"]

    for (label, vals), col, ls in zip(values.items(), palette, line_styles):
        vals_closed = vals + [vals[0]]
        ax.plot(angles, vals_closed, linewidth=1.8, linestyle=ls, color=col, label=label)
        ax.fill(angles, vals_closed, alpha=0.04, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Yelp — Multi-metric Radar: Baseline vs GSP", size=11,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.15), fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    save(fig, "fig10_radar_yelp")


# ---------------------------------------------------------------------------
# Figure 11: Summary Table Figure — Both datasets side by side
# ---------------------------------------------------------------------------
def fig_summary_table():
    print("Figure 11: Summary table...")

    ml1m_rows = [
        ["LightGCN", "Baseline", "0.0804", "0.1057", "0.1057", "20.0", "1080"],
        ["LightGCN", "GSP",      "0.0829", "0.1068", "0.1068", "19.3", "1080"],
        ["GCN",      "Baseline", "0.3468", "0.6449", "0.6449", "18.4",  "650"],
        ["GCN",      "GSP",      "0.3533", "0.6478", "0.6478", "18.6",  "650"],
        ["GAT",      "Baseline", "0.0888", "0.1873", "0.1873", "32.1", "3703"],
        ["GAT",      "GSP",      "0.0911", "0.1925", "0.1925", "32.4", "3703"],
        ["GraphSAGE","Baseline", "0.3548", "0.6500", "0.6500", "18.3",  "650"],
        ["GraphSAGE","GSP",      "0.3506", "0.6465", "0.6465", "18.7",  "650"],
    ]
    yelp_rows = [
        ["LightGCN", "Baseline", "0.0945", "0.1769", "0.1769",  "23.1", "2968"],
        ["LightGCN", "GSP",      "0.0951", "0.1800", "0.1800",  "22.6", "3042"],
        ["GCN",      "Baseline", "0.5950", "0.9125", "0.9125",  "95.4", "1816"],
        ["GCN",      "GSP",      "0.5760", "0.9052", "0.9052",  "96.4", "1814"],
        ["GAT",      "Baseline", "0.4426", "0.8296", "0.8296", "244.6","10049"],
        ["GAT",      "GSP",      "0.5237", "0.8743", "0.8743", "244.8","10044"],
        ["GraphSAGE","Baseline", "0.5887", "0.9096", "0.9096",  "95.7", "1816"],
        ["GraphSAGE","GSP",      "0.5785", "0.9064", "0.9064",  "96.0", "1814"],
    ]

    col_labels = ["Model", "Run", "NDCG@10", "Recall@10", "HitRate@10", "Time(s)", "GPU(MB)"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for ax, rows, title in zip(axes, [ml1m_rows, yelp_rows], ["ML-1M Results", "Yelp Results"]):
        ax.axis("off")
        tbl = ax.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)

        # Header color
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor("#2166ac")
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")

        # Alternate row colors + highlight GSP rows
        for i, row in enumerate(rows, start=1):
            row_color = "#f7f7f7" if i % 2 == 0 else "white"
            if row[1] == "GSP":
                row_color = "#fff3cd"
            for j in range(len(col_labels)):
                tbl[(i, j)].set_facecolor(row_color)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=4)

    fig.suptitle("Full Results Summary (cosine, frac=1.0, ms=1)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "fig11_summary_table")


# ---------------------------------------------------------------------------
# Figure 12: NDCG@10 vs Fraction — sweep line plots by curvature
# (averaged over min_shared, for best model per dataset)
# ---------------------------------------------------------------------------
def fig_ndcg_vs_fraction():
    print("Figure 12: NDCG@10 vs fraction sweep (GCN)...")

    # ML-1M GCN GSP values (average over min_shared 1,3,5)
    ml1m_c_gsp  = {f: np.mean([v[1] for v in ms.values()])
                   for f, ms in {
                       "025": {"1":(0.2392,0.2400),"3":(0.2389,0.2401),"5":(0.2396,0.2411)},
                       "05":  {"1":(0.3823,0.3321),"3":(0.3810,0.3322),"5":(0.3885,0.3290)},
                       "075": {"1":(0.3174,0.3245),"3":(0.3555,0.3553),"5":(0.3562,0.3631)},
                       "10":  {"1":(0.3468,0.3533),"3":(0.3528,0.3141),"5":(0.3337,0.3550)},
                   }.items()}
    ml1m_c_base = {f: np.mean([v[0] for v in ms.values()])
                   for f, ms in {
                       "025": {"1":(0.2392,0.2400),"3":(0.2389,0.2401),"5":(0.2396,0.2411)},
                       "05":  {"1":(0.3823,0.3321),"3":(0.3810,0.3322),"5":(0.3885,0.3290)},
                       "075": {"1":(0.3174,0.3245),"3":(0.3555,0.3553),"5":(0.3562,0.3631)},
                       "10":  {"1":(0.3468,0.3533),"3":(0.3528,0.3141),"5":(0.3337,0.3550)},
                   }.items()}
    ml1m_fr_gsp = {f: np.mean([v[1] for v in ms.values()])
                   for f, ms in {
                       "025": {"1":(0.2384,0.2392),"3":(0.2418,0.2402),"5":(0.2402,0.2445)},
                       "05":  {"1":(0.3933,0.3850),"3":(0.3802,0.3878),"5":(0.3955,0.3294)},
                       "075": {"1":(0.3183,0.3192),"3":(0.3545,0.3529),"5":(0.3544,0.3513)},
                       "10":  {"1":(0.3456,0.3469),"3":(0.3541,0.3014),"5":(0.3441,0.3467)},
                   }.items()}

    frac_nums = [FRAC_VALS[f] for f in FRACS]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.plot(frac_nums, [ml1m_c_base[f] for f in FRACS],
            "o:", color="#888888", linewidth=1.5, markersize=5, label="Baseline (avg)")
    ax.plot(frac_nums, [ml1m_c_gsp[f] for f in FRACS],
            "o-", color=COLORS["cosine"], linewidth=2, markersize=6, label="Cosine (GSP)")
    ax.plot(frac_nums, [ml1m_fr_gsp[f] for f in FRACS],
            "s--", color=COLORS["forman_ricci"], linewidth=2, markersize=6, label="Forman–Ricci (GSP)")
    ax.set_xlabel("Fraction of UU Edges Retained")
    ax.set_ylabel("NDCG@10 (avg over min_shared)")
    ax.set_title("ML-1M — GCN NDCG@10 vs Edge Fraction")
    ax.legend(framealpha=0.8)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # Yelp: average all models
    yelp_cosine_raw = {
        "025": {"1":(0.4853,0.4626),"3":(0.4865,0.4741),"5":(0.4849,0.4656)},
        "05":  {"1":(0.4494,0.4290),"3":(0.4469,0.4409),"5":(0.4497,0.3955)},
        "075": {"1":(0.4466,0.4307),"3":(0.4454,0.4555),"5":(0.4479,0.4713)},
        "10":  {"1":(0.4302,0.4433),"3":(0.4543,0.4482),"5":(0.4504,0.4409)},
    }
    yelp_fr_raw = {
        "025": {"1":(0.4864,0.4875),"3":(0.4951,0.4523),"5":(0.4866,0.4354)},
        "05":  {"1":(0.4480,0.4222),"3":(0.4473,0.4225),"5":(0.4470,0.4194)},
        "075": {"1":(0.4531,0.4382),"3":(0.4474,0.4412),"5":(0.4463,0.4299)},
        "10":  {"1":(0.4447,0.4387),"3":(0.4533,0.4305),"5":(0.4371,0.4526)},
    }
    y_cosine_gsp  = [np.mean([v[1] for v in yelp_cosine_raw[f].values()]) for f in FRACS]
    y_cosine_base = [np.mean([v[0] for v in yelp_cosine_raw[f].values()]) for f in FRACS]
    y_fr_gsp      = [np.mean([v[1] for v in yelp_fr_raw[f].values()]) for f in FRACS]

    ax2 = axes[1]
    ax2.plot(frac_nums, y_cosine_base,
             "o:", color="#888888", linewidth=1.5, markersize=5, label="Baseline (avg)")
    ax2.plot(frac_nums, y_cosine_gsp,
             "o-", color=COLORS["cosine"], linewidth=2, markersize=6, label="Cosine (GSP)")
    ax2.plot(frac_nums, y_fr_gsp,
             "s--", color=COLORS["forman_ricci"], linewidth=2, markersize=6, label="Forman–Ricci (GSP)")
    ax2.set_xlabel("Fraction of UU Edges Retained")
    ax2.set_ylabel("NDCG@10 (avg over models & min_shared)")
    ax2.set_title("Yelp — NDCG@10 vs Edge Fraction")
    ax2.legend(framealpha=0.8)
    ax2.grid(linestyle="--", alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("NDCG@10 vs UU Edge Fraction (averaged over min_shared)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig12_ndcg_vs_fraction")


# ---------------------------------------------------------------------------
# Figure 13: Inference Time Breakdown (Yelp GSP: forward + projection)
# ---------------------------------------------------------------------------
def fig_inference_time():
    print("Figure 13: Inference time breakdown...")

    infer_data = {
        "lightgcn": {"forward": 0.0559, "projection": 0.0508},
        "gcn":       {"forward": 0.0659, "projection": 0.0659},
        "gat":       {"forward": 0.1116, "projection": 0.0669},
        "graphsage": {"forward": 0.0650, "projection": 0.0768},
    }
    baseline_infer = {
        "lightgcn": 0.0551,
        "gcn":       0.0595,
        "gat":       0.0677,
        "graphsage": 0.0643,
    }

    x = np.arange(len(MODELS))
    fwd     = [infer_data[m]["forward"]    for m in MODELS]
    proj    = [infer_data[m]["projection"] for m in MODELS]
    base_i  = [baseline_infer[m]           for m in MODELS]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x, fwd,  label="Forward pass",  color="#4C72B0", alpha=0.9)
    bars2 = ax.bar(x, proj, bottom=fwd, label="Projection",  color="#DD8452", alpha=0.9)

    # Baseline dots
    ax.scatter(x, base_i, marker="D", color="#2ca02c", zorder=5, s=50, label="Baseline inference")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Inference Time (s)")
    ax.set_title("Yelp — GSP Inference Time Breakdown (cosine, frac=1.0, ms=1)")
    ax.legend(framealpha=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, "fig13_inference_time")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Saving figures to: {FIG_DIR}\n")
    fig_pipeline()                  # fig01
    fig_ndcg_comparison()           # fig02
    fig_multi_metric_yelp()         # fig03
    fig_loss_curves_ml1m()          # fig04
    fig_loss_curves_cosine_vs_forman_ml1m()          # fig05b
    fig_loss_curves_cosine_vs_forman_yelp()          # fig05c
    fig_loss_curves_cosine_vs_forman_ml25m()         # fig05d
    fig_resources()                 # fig06  ← stacked time + GPU, data-driven
    print("Figure 7: Per-model sweep heatmaps (NDCG@10)...")
    fig_sweep_heatmaps()            # fig07a–07d  ← per-model, data-driven
    fig_ndcg_at_k()                 # fig08
    print("Figure 9: Speedup heatmaps...")
    fig_speedup_heatmaps()          # fig09a–09d  ← speedup factor, data-driven
    fig_gsp_stats()                 # fig10  ← preprocessing time + super-nodes
    fig_radar_yelp()                # fig11
    fig_summary_table()             # fig12
    fig_ndcg_vs_fraction()          # fig13
    fig_inference_time()            # fig14
    print(f"\nDone — {len(list(FIG_DIR.glob('*.png')))} PNG files in {FIG_DIR}")
