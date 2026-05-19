"""
Explainable Recommendation System

Generates per-item explanations based on:
  1. Neighborhood-based reasoning  – similar users who interacted with the item
  2. Graph-based reasoning         – influential neighbors in user-user graph
  3. Path-based explanation        – user → similar users → shared items → item
  4. Feature/interaction overlap   – common items between user and contributors

Outputs: explanation_text, contributing_users, contribution_scores
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ItemExplanation:
    user_id: int
    item_id: int
    model: str
    run_type: str
    rank: int                                    # 1-based rank in top-K list
    explanation_text: str = ""
    reasoning_type: str = ""                     # "neighborhood" | "graph" | "path" | "combined"
    contributing_users: List[int] = field(default_factory=list)
    contribution_scores: List[float] = field(default_factory=list)
    common_items_count: List[int] = field(default_factory=list)    # per contributing user
    path: List[int] = field(default_factory=list)                  # user → ... → item
    path_description: str = ""


@dataclass
class UserExplanation:
    user_id: int
    model: str
    run_type: str
    item_explanations: List[ItemExplanation] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Explainer
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationExplainer:
    """
    Generates human-readable explanations for GNN recommendations.

    Explanations use:
    - User-user similarity graph (from GSP or cosine similarity)
    - User-item interaction history (train set)
    - GNN embeddings for contribution scoring

    Usage
    -----
    ::
        explainer = RecommendationExplainer(
            user_emb=user_emb,
            item_emb=item_emb,
            train_interactions=seen_train,
            output_dir="outputs/explanations",
        )
        explanations = explainer.explain_user(
            user_id=42,
            recommended_items=[5, 17, 83],
            model_name="lightgcn",
            run_type="baseline",
            top_n_neighbors=5,
        )
        explainer.save()
    """

    def __init__(
        self,
        user_emb: np.ndarray,
        item_emb: np.ndarray,
        train_interactions: Dict[int, Set[int]],
        output_dir: str,
        uu_edges: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        num_neighbors: int = 5,
        item_names: Optional[Dict[int, str]] = None,
    ):
        """
        Parameters
        ----------
        user_emb           (U, D) user embeddings
        item_emb           (I, D) item embeddings
        train_interactions user_id -> set of item_ids in training set
        output_dir         directory for saving explanations
        uu_edges           (u_arr, v_arr, weights_arr) user-user graph edges (optional)
        num_neighbors      top-N influential neighbors to include in explanation
        item_names         optional dict item_id -> name for human-readable output
        """
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.train_interactions = train_interactions
        self.output_dir = output_dir
        self.num_neighbors = num_neighbors
        self.item_names = item_names or {}
        os.makedirs(output_dir, exist_ok=True)

        # Build user-user adjacency from provided edges or from embedding cosine similarity
        self._uu_edges = uu_edges
        self._uu_adj: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None
        if uu_edges is not None:
            self._build_uu_adj(*uu_edges)

        # Precompute item popularity (number of training interactions)
        self._item_popularity: np.ndarray = np.zeros(item_emb.shape[0], dtype=np.int32)
        for items in train_interactions.values():
            for it in items:
                if it < self._item_popularity.size:
                    self._item_popularity[it] += 1

        self._explanations: Dict[str, List[UserExplanation]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Graph construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_uu_adj(
        self,
        u_arr: np.ndarray,
        v_arr: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Build adjacency list: user_id -> (neighbor_ids, edge_weights)."""
        adj: Dict[int, Tuple[list, list]] = {}
        for u, v, w in zip(u_arr.tolist(), v_arr.tolist(), weights.tolist()):
            adj.setdefault(u, ([], []))[0].append(v)
            adj.setdefault(u, ([], []))[1].append(w)
            adj.setdefault(v, ([], []))[0].append(u)
            adj.setdefault(v, ([], []))[1].append(w)
        self._uu_adj = {
            k: (np.array(nb, dtype=np.int64), np.array(wt, dtype=np.float32))
            for k, (nb, wt) in adj.items()
        }

    def _get_similar_users(self, user_id: int, top_n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (neighbor_ids, scores) – uses UU graph if available,
        otherwise falls back to embedding cosine similarity.
        """
        if self._uu_adj and user_id in self._uu_adj:
            nb_ids, nb_wts = self._uu_adj[user_id]
            if nb_ids.size > top_n:
                top_idx = np.argpartition(nb_wts, -top_n)[-top_n:]
                top_idx = top_idx[np.argsort(nb_wts[top_idx])[::-1]]
                return nb_ids[top_idx], nb_wts[top_idx]
            order = np.argsort(nb_wts)[::-1]
            return nb_ids[order], nb_wts[order]

        # Fallback: cosine similarity from embeddings
        if user_id >= self.user_emb.shape[0]:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        u_vec = self.user_emb[user_id]
        u_norm = np.linalg.norm(u_vec) + 1e-9
        norms = np.linalg.norm(self.user_emb, axis=1) + 1e-9
        sims = (self.user_emb @ u_vec) / (norms * u_norm)
        sims[user_id] = -1.0  # exclude self
        top_idx = np.argpartition(sims, -(top_n + 1))[-(top_n + 1):]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]][:top_n]
        return top_idx.astype(np.int64), sims[top_idx]

    # ─────────────────────────────────────────────────────────────────────────
    # Explanation generation
    # ─────────────────────────────────────────────────────────────────────────

    def explain_item(
        self,
        user_id: int,
        item_id: int,
        rank: int,
        model_name: str,
        run_type: str,
    ) -> ItemExplanation:
        """Generate a single item explanation for one user-item pair."""
        neighbors, neighbor_scores = self._get_similar_users(user_id, self.num_neighbors)

        user_items: Set[int] = self.train_interactions.get(user_id, set())
        contributing: List[int] = []
        contrib_scores: List[float] = []
        common_counts: List[int] = []
        neighbor_item_evidence: List[int] = []  # neighbors that interacted with item_id

        for nb, score in zip(neighbors.tolist(), neighbor_scores.tolist()):
            nb_items: Set[int] = self.train_interactions.get(nb, set())
            if item_id in nb_items:
                neighbor_item_evidence.append(nb)
                contributing.append(nb)
                contrib_scores.append(float(score))
                common = len(user_items & nb_items)
                common_counts.append(common)

        # Build path: user → most-similar contributing neighbor → item
        path: List[int] = []
        path_desc = ""
        if contributing:
            # Pick highest-scoring contributing neighbor
            best_nb = contributing[0]
            nb_items = self.train_interactions.get(best_nb, set())
            shared = list(user_items & nb_items)[:3]  # up to 3 shared items for path
            path = [user_id, best_nb] + (shared if shared else []) + [item_id]
            shared_str = (
                f"via shared items {shared}" if shared else "without shared training items"
            )
            path_desc = (
                f"User {user_id} → neighbor {best_nb} {shared_str} → item {item_id}"
            )

        # Determine item name if available
        item_label = self.item_names.get(item_id, f"item_{item_id}")
        popularity = int(self._item_popularity[item_id]) if item_id < len(self._item_popularity) else 0

        # Generate explanation text
        if neighbor_item_evidence:
            nb_str = ", ".join(str(n) for n in neighbor_item_evidence[:3])
            text = (
                f"Recommended because {len(neighbor_item_evidence)} similar user(s) "
                f"(e.g. users {nb_str}) interacted with {item_label}. "
                f"This item has been interacted with by {popularity} users in training."
            )
            if common_counts:
                text += (
                    f" The most similar contributing neighbor shared "
                    f"{common_counts[0]} item(s) with you."
                )
            reasoning = "neighborhood+graph"
        else:
            # Pure embedding similarity – no direct neighbor evidence
            text = (
                f"Recommended based on your embedding profile similarity to "
                f"{item_label} (rank {rank}). "
                f"Popularity: {popularity} training interactions."
            )
            reasoning = "embedding"

        return ItemExplanation(
            user_id=user_id,
            item_id=item_id,
            model=model_name,
            run_type=run_type,
            rank=rank,
            explanation_text=text,
            reasoning_type=reasoning,
            contributing_users=contributing,
            contribution_scores=contrib_scores,
            common_items_count=common_counts,
            path=path,
            path_description=path_desc,
        )

    def explain_user(
        self,
        user_id: int,
        recommended_items: List[int],
        model_name: str,
        run_type: str,
        top_k: int = 10,
    ) -> UserExplanation:
        """Generate explanations for all top-K items for a single user."""
        item_exps: List[ItemExplanation] = []
        for rank, item_id in enumerate(recommended_items[:top_k], start=1):
            exp = self.explain_item(user_id, item_id, rank, model_name, run_type)
            item_exps.append(exp)

        ue = UserExplanation(
            user_id=user_id,
            model=model_name,
            run_type=run_type,
            item_explanations=item_exps,
        )
        key = f"{model_name}_{run_type}"
        self._explanations.setdefault(key, []).append(ue)
        return ue

    def explain_batch(
        self,
        recommendations: Dict[int, List[int]],   # user_id -> ranked item list
        model_name: str,
        run_type: str,
        top_k: int = 10,
        max_users: int = 500,
    ) -> List[UserExplanation]:
        """
        Generate explanations for a batch of users.

        Parameters
        ----------
        recommendations  user_id -> ranked list of item_ids
        max_users        cap to avoid excessive computation on large datasets
        """
        results: List[UserExplanation] = []
        for i, (uid, items) in enumerate(recommendations.items()):
            if i >= max_users:
                break
            ue = self.explain_user(uid, items, model_name, run_type, top_k)
            results.append(ue)
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # I/O
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, model_name: Optional[str] = None) -> None:
        """Persist explanations to JSON and CSV."""
        keys = (
            [k for k in self._explanations if not model_name or k.startswith(model_name)]
        )
        for key in keys:
            user_exps = self._explanations[key]
            if not user_exps:
                continue

            # --- JSON (full structured explanations) ---
            json_path = os.path.join(self.output_dir, f"explanations_{key}.json")
            payload = []
            for ue in user_exps:
                payload.append({
                    "user_id": ue.user_id,
                    "model": ue.model,
                    "run_type": ue.run_type,
                    "items": [asdict(ie) for ie in ue.item_explanations],
                })
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)

            # --- CSV (one row per user-item pair) ---
            csv_path = os.path.join(self.output_dir, f"explanations_{key}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "user_id", "item_id", "model", "run_type", "rank",
                    "reasoning_type", "num_contributing_users",
                    "max_contribution_score", "max_common_items",
                    "path_length", "explanation_text",
                ])
                for ue in user_exps:
                    for ie in ue.item_explanations:
                        writer.writerow([
                            ie.user_id, ie.item_id, ie.model, ie.run_type, ie.rank,
                            ie.reasoning_type,
                            len(ie.contributing_users),
                            f"{max(ie.contribution_scores, default=0.0):.4f}",
                            max(ie.common_items_count, default=0),
                            len(ie.path),
                            ie.explanation_text[:200],
                        ])

    def get_sample_explanations(
        self,
        key: str,
        n_users: int = 5,
        items_per_user: int = 3,
    ) -> List[Dict]:
        """Return a small human-readable sample for reporting."""
        user_exps = self._explanations.get(key, [])[:n_users]
        samples = []
        for ue in user_exps:
            samples.append({
                "user_id": ue.user_id,
                "model": ue.model,
                "run_type": ue.run_type,
                "top_items": [
                    {
                        "item_id": ie.item_id,
                        "rank": ie.rank,
                        "explanation": ie.explanation_text,
                        "contributing_users": ie.contributing_users[:3],
                        "path": ie.path_description,
                    }
                    for ie in ue.item_explanations[:items_per_user]
                ],
            })
        return samples
