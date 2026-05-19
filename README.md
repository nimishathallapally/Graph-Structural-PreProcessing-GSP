# gsprec — Graph Structural Pre-conditioning Recommender System

`gsprec` is a research framework for GNN-based collaborative filtering that applies **Graph Signal Processing (GSP) pre-conditioning** to compress the user-user interaction graph before training. The key idea is to reduce the number of nodes via spectral coarsening (eigenvector decomposition), guided by either **Forman-Ricci curvature** or **cosine similarity**, so that downstream GNN models train faster on a structurally cleaner graph without sacrificing recommendation quality.

### How it works

1. **Build** a bipartite user-item interaction graph from ratings
2. **Construct** a user-user graph based on co-rated items
3. **Prune** low-curvature or low-similarity edges
4. **Coarsen** user nodes into super-nodes via spectral clustering (eigenvector decomposition)
5. **Train** a GNN (LightGCN, GAT, GraphSAGE, or GCN) on the compressed graph
6. **Evaluate** with Recall, NDCG, and Precision@K against a full-graph baseline

### Supported models

| Model | Description |
|-------|-------------|
| `lightgcn` | LightGCN — simplified linear graph convolution |
| `gat` | Graph Attention Network |
| `graphsage` | GraphSAGE — inductive neighbourhood sampling |
| `gcn` | Vanilla Graph Convolutional Network |

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (GPU optional but recommended for large datasets)
- A [Kaggle](https://www.kaggle.com) account with an API token (for dataset downloads)

---

## Setup

### 1. Create a virtual environment

**Linux / macOS**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install the package

```bash
pip install -e .
```

This installs `gsprec` in editable mode along with all required dependencies.

> **GPU vs CPU note:**
> The default install pulls a CUDA-enabled PyTorch wheel which is several GB.
> For a CPU-only install (much smaller), replace the `torch` install step:
> ```bash
> pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
> pip install -e .
> ```

## Kaggle API token

All datasets are downloaded via `kagglehub`, which requires a Kaggle API token.

### 1. Create the token

1. Log in at [kaggle.com](https://www.kaggle.com)
2. Go to **Account → Settings → API → Create New Token**
3. A file called `kaggle.json` will be downloaded

### 2. Place the token

**Linux / macOS**
```bash
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json   # restrict permissions (required)
```

**Windows (PowerShell)**
```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"
Copy-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

The token is stored at:
- Linux / macOS: `~/.kaggle/kaggle.json`
- Windows: `%USERPROFILE%\.kaggle\kaggle.json`

## Download datasets

Datasets are **not included** in this repository and must be downloaded via Kaggle.

### MovieLens-1M and MovieLens-25M

Downloaded **automatically** on first run via `kagglehub`. You only need to accept the license on Kaggle beforehand:

| Dataset | Kaggle URL |
|---------|-----------|
| MovieLens-1M | https://www.kaggle.com/datasets/shikharg97/movielens-1m |
| MovieLens-25M | https://www.kaggle.com/datasets/garymk/movielens-25m-dataset |

### Yelp Academic Dataset

Requires an explicit download step. Accept the license at https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset, then run:

```bash
python yelp.py
```

This copies the dataset files to `data/yelp/`. Expected files after download:

```
data/yelp/
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_checkin.json
├── yelp_academic_dataset_review.json
├── yelp_academic_dataset_tip.json
└── yelp_academic_dataset_user.json
```

---

## Run

### CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--models` | `lightgcn gat graphsage gcn` | Space-separated list of models, or `all` |
| `--use_gsp` | `false` | Enable GSP pre-conditioning |
| `--epochs` | `10` | Training epochs |
| `--batch_size` | `65536` | BPR training batch size |
| `--debug_mode` | `false` | Small user subset for fast iteration |
| `--alpha` | `0.5` | GSP blend weight α (curvature vs. importance) |
| `--topk` | `0` | Top-k edge selection (`0` = use percentile) |
| `--output_dir` | `outputs` | Directory for checkpoints and results |
| `--config` | — | Path to a JSON config file (overrides CLI flags) |
| `--seed` | `42` | Global random seed |

### Single run

```bash
# MovieLens-1M, all models, defaults
python main.py

# Quick debug run
python main.py --models lightgcn --debug_mode

# With GSP enabled
python main.py --models lightgcn --use_gsp true --epochs 10

# All models
python main.py --models all

# From a config file
python main.py --config configs/default.json
python main.py --config configs/movielens25m.json
python main.py --config configs/yelp.json
```

### Full sweeps

These scripts sweep across curvature modes (`cosine`, `forman_ricci`), graph fractions (`0.25`, `0.5`, `0.75`, `1.0`), and minimum shared interactions (`1`, `3`, `5`) for all four models — 24 conditions per dataset.

```bash
# MovieLens-1M sweep  →  output/sweep_ml1m/
bash scripts/run_sweep_ml1m.sh

# MovieLens-25M sweep  →  output/sweep_ml25m/
bash scripts/run_sweep_ml25m.sh

# MovieLens-25M ordered sweep  →  output/sweep_ml25m_ordered/
bash scripts/run_sweep_ml25m_ordered.sh

# Yelp compressible sweep  →  output/sweep_yelp/
bash scripts/run_sweep_yelp_compressible.sh
```

### Figure generation

```bash
# Thesis figures
python scripts/figures.py

# Paper/publication figures
python scripts/plot_paper_figures.py

# Speedup computation
python scripts/compute_speedup.py
```

---

## Output structure

Each run writes results under `--output_dir` (default: `outputs/`):

```
outputs/
└── <model_name>/
    ├── checkpoints/         # Saved model weights
    ├── metrics.jsonl        # Per-epoch training metrics
    ├── eval_metrics.jsonl   # Final evaluation metrics (Recall, NDCG, Precision@K)
    └── training_log.txt     # Full training log
```

Sweep runs go to `output/sweep_<dataset>/` with one subdirectory per condition, e.g. `cosine_frac050_ms3/`.

---

## Project structure

```
.
├── main.py                  # CLI entry point
├── yelp.py                  # Yelp dataset downloader
├── pyproject.toml
├── requirements.txt
├── configs/                 # JSON experiment configs
│   ├── default.json         # MovieLens-1M defaults
│   ├── movielens25m.json
│   ├── amazon_music.json
│   └── yelp.json
├── src/
│   └── gsprec/
│       ├── config.py        # Dataclass configs (DataConfig, GSPConfig, …)
│       ├── data/            # Dataset loading & bipartite graph building
│       ├── graph/           # GSP pre-conditioning (curvature, ER coarsening)
│       ├── models/          # GNN architectures, BPR trainer, evaluator
│       ├── analytics/       # Post-training analysis & report generation
│       ├── pipeline/        # High-level runner wrappers
│       └── utils/           # Hardware monitoring, metrics export
├── scripts/                 # Experiment, sweep, and figure scripts
└── output/                  # Generated results (git-ignored)
```

