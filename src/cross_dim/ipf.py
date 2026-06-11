from __future__ import annotations
import numpy as np


def run_ipf(
    seed: np.ndarray,
    row_marginals: np.ndarray,
    col_marginals: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-9,
) -> tuple[np.ndarray, int]:
    """Iterative Proportional Fitting (RAS algorithm).

    Scales a seed matrix iteratively until row and column sums match the
    given marginals. Convergence is guaranteed when seed is non-negative
    and marginals share the same total.

    Args:
        seed: (R, C) non-negative prior matrix.
        row_marginals: (R,) target row sums.
        col_marginals: (C,) target col sums — must sum to same total as row_marginals.
        max_iter: iteration cap.
        tol: max absolute deviation from marginals to declare convergence.

    Returns:
        (balanced_matrix, iterations_used)
    """
    M = seed.copy().astype(float)

    if M.sum() == 0:
        # Degenerate seed: use outer product of marginals as flat prior.
        M = np.outer(row_marginals, col_marginals)

    for i in range(1, max_iter + 1):
        row_sums = M.sum(axis=1)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        M *= (row_marginals / row_sums)[:, None]

        col_sums = M.sum(axis=0)
        col_sums = np.where(col_sums > 0, col_sums, 1.0)
        M *= (col_marginals / col_sums)[None, :]

        err = max(
            np.abs(M.sum(axis=1) - row_marginals).max(),
            np.abs(M.sum(axis=0) - col_marginals).max(),
        )
        if err < tol:
            return M, i

    return M, max_iter
