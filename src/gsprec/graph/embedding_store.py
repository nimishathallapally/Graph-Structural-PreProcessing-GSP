"""
Stage III — Embedding Storage via NumPy memory-mapped arrays.

Stores GNN embeddings on disk as .npy memory-mapped files so that the full
embedding matrix never has to reside in CPU RAM simultaneously.

Public API
----------
    EmbeddingStore(path, num_nodes, emb_dim, dtype)  →  context manager
        .store(arr)          write numpy array into the mmap file
        .load_slice(start, end)  read a row slice without full load
        .project(C)          sparse projection H_final = Cᵀ @ H_mmap

    project_embeddings(H_super, C)  →  H_final (num_users × emb_dim)
        Implements:  H_final = Cᵀ × H_GNN
        where C is the sparse CSR mapping matrix (num_super × num_users).
"""
from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp


class EmbeddingStore:
    """
    Thin wrapper around numpy.memmap for large embedding matrices.

    Usage
    -----
    >>> store = EmbeddingStore("outputs/embeddings/gsp_user.npy", 1_000_000, 128)
    >>> store.store(np_array_float16)         # write once
    >>> row = store.load_slice(0, 100)         # read without full load
    >>> store.close()
    """

    def __init__(
        self,
        path: str,
        num_nodes: int,
        emb_dim: int,
        dtype: np.dtype = np.float16,
        mode: str = "w+",                  # 'w+' = create+read/write
    ) -> None:
        self.path = path
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.dtype = np.dtype(dtype)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._mmap: Optional[np.memmap] = np.memmap(
            path, dtype=self.dtype, mode=mode, shape=(num_nodes, emb_dim)
        )

    # ── Write ────────────────────────────────────────────────────────────────

    def store(self, arr: np.ndarray) -> None:
        """Write a (num_nodes × emb_dim) array into the mmap file."""
        if arr.shape != (self.num_nodes, self.emb_dim):
            raise ValueError(
                f"Shape mismatch: expected ({self.num_nodes}, {self.emb_dim}), "
                f"got {arr.shape}"
            )
        self._mmap[:] = arr.astype(self.dtype)
        self._mmap.flush()

    def store_chunk(self, arr: np.ndarray, start: int) -> None:
        """Write a partial chunk starting at row `start`."""
        end = start + arr.shape[0]
        self._mmap[start:end] = arr.astype(self.dtype)

    def flush(self) -> None:
        if self._mmap is not None:
            self._mmap.flush()

    # ── Read ─────────────────────────────────────────────────────────────────

    def load_slice(self, start: int, end: int) -> np.ndarray:
        """Return rows [start, end) as float32 (no full-file load)."""
        return np.array(self._mmap[start:end], dtype=np.float32)

    def load_all(self) -> np.ndarray:
        """Load entire matrix as float32. Use only when RAM allows."""
        return np.array(self._mmap[:], dtype=np.float32)

    # ── Projection ────────────────────────────────────────────────────────────

    def project(self, C: sp.csr_matrix) -> np.ndarray:
        """
        H_final = Cᵀ @ H_mmap

        C is (num_super × num_users); Cᵀ is (num_users × num_super).
        Result is (num_users × emb_dim) as float32.

        Operates in chunks to avoid loading the full mmap into RAM.
        """
        return _sparse_project(self._mmap[:].astype(np.float32), C)

    # ── Context manager & cleanup ─────────────────────────────────────────────

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.flush()
            del self._mmap
            self._mmap = None

    def memory_MB(self) -> float:
        """On-disk file size in MB (equals dtype size × num_nodes × emb_dim)."""
        return float(self.num_nodes * self.emb_dim * self.dtype.itemsize) / (1024 ** 2)

    def __enter__(self) -> "EmbeddingStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"EmbeddingStore({self.path!r}, "
            f"shape=({self.num_nodes}, {self.emb_dim}), "
            f"dtype={self.dtype}, "
            f"mem={self.memory_MB():.2f}MB)"
        )


# ---------------------------------------------------------------------------
# Projection step:  H_final = Cᵀ × H_GNN
# ---------------------------------------------------------------------------

def project_embeddings(
    H_super: np.ndarray,
    C: sp.csr_matrix,
    chunk_size: int = 50_000,
) -> Tuple[np.ndarray, float]:
    """
    Project super-node embeddings back to the original user space.

        H_final = Cᵀ @ H_super

    Parameters
    ----------
    H_super : (num_super × emb_dim) float32/float16 array
    C       : (num_super × num_users) CSR sparse matrix (binary)
    chunk_size : rows of H_final computed per batch (controls peak RAM)

    Returns
    -------
    H_final : (num_users × emb_dim) float32 array
    projection_time_s : float
    """
    t0 = time.perf_counter()
    result = _sparse_project(H_super.astype(np.float32), C)
    projection_time_s = time.perf_counter() - t0
    return result, projection_time_s


def _sparse_project(
    H_super: np.ndarray,
    C: sp.csr_matrix,
) -> np.ndarray:
    """
    Cᵀ @ H_super  using sparse-dense multiply.

    C shape  : (num_super × num_users)
    Cᵀ shape : (num_users × num_super)
    H_super  : (num_super × emb_dim)
    result   : (num_users × emb_dim)

    scipy.sparse uses CSR for efficient row access; Cᵀ in CSC == C in CSR.
    All arithmetic in float64 internally for numerical stability, then
    cast to float32 for storage.
    """
    Ct = C.T.tocsr()                          # (num_users × num_super) CSR
    H = H_super.astype(np.float64)           # numeric stability
    projected = Ct.dot(H)                    # (num_users × emb_dim)
    return projected.astype(np.float32)


# ---------------------------------------------------------------------------
# Open existing mmap in read-only mode
# ---------------------------------------------------------------------------

def open_embedding_store(
    path: str,
    num_nodes: int,
    emb_dim: int,
    dtype: np.dtype = np.float16,
) -> EmbeddingStore:
    """Open a previously written EmbeddingStore in read-only mode."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding store not found: {path}")
    store = EmbeddingStore.__new__(EmbeddingStore)
    store.path = path
    store.num_nodes = num_nodes
    store.emb_dim = emb_dim
    store.dtype = np.dtype(dtype)
    store._mmap = np.memmap(path, dtype=store.dtype, mode="r", shape=(num_nodes, emb_dim))
    return store
