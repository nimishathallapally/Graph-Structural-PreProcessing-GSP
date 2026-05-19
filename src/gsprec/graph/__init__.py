from .precondition import (
    adaptive_importance,
    build_bipartite_edge_index,
    build_coarsened_interactions,
    build_train_interaction_matrix,
    coarsen_users,
    compute_user_user_curvature,
)
from .gsp_ops import (
    build_user_item_sparse_torch,
    cluster_users_vectorized,
    compute_curvature_sparse,
    compute_er_fast,
    gsp_preprocess,
    select_high_curvature_edges,
    sparsify_with_adaptive_importance,
)
from .embedding_store import (
    EmbeddingStore,
    open_embedding_store,
    project_embeddings,
)

__all__ = [
    # legacy (precondition.py)
    "adaptive_importance",
    "build_bipartite_edge_index",
    "build_coarsened_interactions",
    "build_train_interaction_matrix",
    "coarsen_users",
    "compute_user_user_curvature",
    # new GSP ops (gsp_ops.py)
    "build_user_item_sparse_torch",
    "cluster_users_vectorized",
    "compute_curvature_sparse",
    "compute_er_fast",
    "gsp_preprocess",
    "select_high_curvature_edges",
    "sparsify_with_adaptive_importance",
    # Stage III embedding storage
    "EmbeddingStore",
    "open_embedding_store",
    "project_embeddings",
]
