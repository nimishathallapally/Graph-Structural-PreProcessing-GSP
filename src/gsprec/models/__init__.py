from .gnn import (
    GATRecommender,
    GCNRecommender,
    RankingEvalConfig,
    SAGERecommender,
    TrainConfig,
    evaluate_ranking_from_embeddings,
    rmse_mae,
    train_model,
)
from .architectures import (
    LightGCNRecommender,
    GraphSAGERecommender,
    ModelConfig,
    get_model,
)
from .trainer import (
    TrainConfig as BPRTrainConfig,
    bpr_loss,
    train_model as bpr_train_model,
)
from .evaluator import (
    BatchedEvaluator,
    EvalConfig,
    compute_efficiency_metrics,
    compute_regression_metrics,
    evaluate_ranking_from_embeddings as evaluate_ranking_batched,
)

__all__ = [
    # legacy (gnn.py)
    "GATRecommender",
    "GCNRecommender",
    "RankingEvalConfig",
    "SAGERecommender",
    "TrainConfig",
    "evaluate_ranking_from_embeddings",
    "rmse_mae",
    "train_model",
    # new architectures (architectures.py)
    "LightGCNRecommender",
    "GraphSAGERecommender",
    "ModelConfig",
    "get_model",
    # new trainer (trainer.py)
    "BPRTrainConfig",
    "bpr_loss",
    "bpr_train_model",
    # evaluator (evaluator.py)
    "BatchedEvaluator",
    "EvalConfig",
    "compute_efficiency_metrics",
    "compute_regression_metrics",
    "evaluate_ranking_batched",
]
