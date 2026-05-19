from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_train_interaction_matrix(
    ratings_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    implicit_threshold: float = 4.0,
) -> csr_matrix:
    users = ratings_df["UserID"].to_numpy(dtype=np.int64) - 1
    items = ratings_df["MovieID"].to_numpy(dtype=np.int64) - 1
    ratings = ratings_df["Rating"].to_numpy(dtype=np.float32)
    pos = ratings >= implicit_threshold
    users = users[pos]
    items = items[pos]
    data = np.ones(len(users), dtype=np.float32)
    return csr_matrix((data, (users, items)), shape=(num_users, num_items))


def compute_user_user_curvature(
    adj_user_item: csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    user_deg = np.array(adj_user_item.sum(axis=1)).reshape(-1)
    user_cooc = adj_user_item @ adj_user_item.T
    user_cooc.setdiag(0)
    cooc = user_cooc.tocoo()
    u = cooc.row.astype(np.int64)
    v = cooc.col.astype(np.int64)
    common_movies = cooc.data.astype(np.float32)

    deg_u = user_deg[u]
    deg_v = user_deg[v]
    F = 4.0 - deg_u - deg_v + 3.0 * common_movies
    return u, v, common_movies, F.astype(np.float32), user_deg.astype(np.float32)


def adaptive_importance(
    u: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    user_deg: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    F_min, F_max = float(F.min()), float(F.max())
    F_norm = (F - F_min) / (F_max - F_min + 1e-8)

    ER = 1.0 / (user_deg[u] + user_deg[v] + 1e-8)
    ER_min, ER_max = float(ER.min()), float(ER.max())
    ER_norm = (ER - ER_min) / (ER_max - ER_min + 1e-8)

    I_e = alpha * (1.0 - F_norm) + (1.0 - alpha) * ER_norm
    return I_e.astype(np.float32)


def coarsen_users(
    num_users: int,
    u: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    I_e: np.ndarray,
    curvature_percentile: float = 90.0,
    importance_percentile: float = 50.0,
) -> tuple[np.ndarray, int]:
    if u.size == 0:
        return np.arange(num_users, dtype=np.int64), num_users

    f_thr = float(np.percentile(F, curvature_percentile))
    i_thr = float(np.percentile(I_e, importance_percentile))
    merge_mask = (F >= f_thr) & (I_e >= i_thr)
    u_merge = u[merge_mask]
    v_merge = v[merge_mask]

    parent = np.arange(num_users, dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in zip(u_merge.tolist(), v_merge.tolist()):
        union(int(i), int(j))

    for i in range(num_users):
        parent[i] = find(i)

    roots, inverse = np.unique(parent, return_inverse=True)
    user_to_super = inverse.astype(np.int64)
    return user_to_super, int(roots.size)


def build_coarsened_interactions(
    ratings_df: pd.DataFrame,
    user_to_super: np.ndarray,
) -> pd.DataFrame:
    df = ratings_df[["UserID", "MovieID", "Rating"]].copy()
    df["user_idx"] = df["UserID"].astype(np.int64) - 1
    df["item_idx"] = df["MovieID"].astype(np.int64) - 1
    df["super_idx"] = user_to_super[df["user_idx"].to_numpy()]
    agg = (
        df.groupby(["super_idx", "item_idx"], as_index=False)
        .agg(rating=("Rating", "mean"), count=("Rating", "size"))
        .astype({"super_idx": np.int64, "item_idx": np.int64})
    )
    return agg


def build_bipartite_edge_index(
    interactions: pd.DataFrame,
    num_super: int,
    num_items: int,
    make_undirected: bool = True,
) -> np.ndarray:
    su = interactions["super_idx"].to_numpy(dtype=np.int64)
    it = interactions["item_idx"].to_numpy(dtype=np.int64) + num_super
    if make_undirected:
        src = np.concatenate([su, it])
        dst = np.concatenate([it, su])
    else:
        src, dst = su, it
    return np.stack([src, dst], axis=0)
