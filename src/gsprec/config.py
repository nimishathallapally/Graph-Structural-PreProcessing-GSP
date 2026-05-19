from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DataConfig:
    implicit_threshold: float = 4.0
    test_seed: int = 42
    debug_mode: bool = False
    max_debug_users: int = 5000
    cache_dir: str = "outputs/cache"
    force_reload: bool = False
    dataset_name: str = "movielens"   # "movielens" | "digital_music"
    dataset_path: str = ""            # required when dataset_name != "movielens"
    min_interactions: int = 0         # drop users with fewer interactions (0 = no filter)


@dataclass
class GSPConfig:
    alpha: float = 0.5
    curvature_percentile: float = 50.0
    curvature_topk: int = 0          # 0 = use percentile
    importance_percentile: float = 50.0
    importance_topk: int = 0         # 0 = use percentile
    er_num_eigenvectors: int = 32
    max_cluster_size: int = 50       # 0 = no cap; splits oversized components
    min_shared_interactions: int = 2 # min co-rated items to retain a UU edge
    er_solver: str = "dwlv"          # "arpack" | "lobpcg" | "jl" | "dwlv"
    er_sketches: int = 32            # number of JL random probes (jl solver only)
    curvature_mode: str = "cosine"   # "cosine" | "forman_ricci"
    er_node_limit: int = 0           # 0 = always run ER; >0 = skip ER when num_users exceeds limit


@dataclass
class TrainRunConfig:
    epochs: int = 100
    lr: float = 5e-3
    weight_decay: float = 1e-5
    batch_size: int = 65536
    neg_ratio: int = 8
    rating_loss_weight: float = 0.0   # kept for legacy compat
    emb_l2_weight: float = 1e-5
    emb_dim: int = 128
    hidden_dim: int = 256
    out_dim: int = 128
    num_layers: int = 4      # LightGCN
    heads: int = 4           # GAT
    dropout: float = 0.1
    use_amp: bool = True
    seed: int = 42


@dataclass
class EvalConfig:
    k: int = 10
    num_negatives: int = 499
    seed: int = 42


@dataclass
class AblationConfig:
    enabled: bool = True
    ablation_model: str = "lightgcn"
    ablation_epochs: int = 30
    importance_percentiles: List[float] = field(
        default_factory=lambda: [20.0, 35.0, 50.0, 65.0, 80.0]
    )


@dataclass
class ProjectConfig:
    output_dir: str = "outputs"
    run_baseline: bool = True
    models: tuple[str, ...] = ("lightgcn", "gat", "graphsage", "gcn")
    data: DataConfig = field(default_factory=DataConfig)
    gsp: GSPConfig = field(default_factory=GSPConfig)
    train: TrainRunConfig = field(default_factory=TrainRunConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    @staticmethod
    def from_json(path: str) -> "ProjectConfig":
        cfg_path = Path(path)
        raw: Dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))

        data_cfg = DataConfig(**raw.get("data", {}))
        gsp_cfg = GSPConfig(**raw.get("gsp", {}))
        train_cfg = TrainRunConfig(**raw.get("train", {}))
        eval_cfg = EvalConfig(**raw.get("eval", {}))

        ablation_raw = raw.get("ablation", {})
        ablation_cfg = AblationConfig(**ablation_raw) if ablation_raw else AblationConfig()

        base_fields = {
            "output_dir": raw.get("output_dir", "outputs"),
            "run_baseline": raw.get("run_baseline", True),
            "models": tuple(raw.get("models", ["lightgcn", "gat", "graphsage", "gcn"])),
        }
        return ProjectConfig(**base_fields, data=data_cfg, gsp=gsp_cfg, train=train_cfg, eval=eval_cfg, ablation=ablation_cfg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
