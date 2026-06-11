from __future__ import annotations
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .ipf import run_ipf


def extract_dim_totals(results, dim: str) -> pd.Series:
    """Total contribution per variable slug for a Deep Dive dimension.

    Returns Series[slug → float] for real slugs only (skips __outros__, anchor_).
    """
    contrib_df = results.contribs[dim]
    slugs = [c for c in contrib_df.columns if c.startswith("$")]
    return contrib_df[slugs].sum()


def _slug_value(slug: str, category: str) -> str | None:
    """Extract the breakdown value from a slug for a given category."""
    m = re.search(rf'\$category:{re.escape(category)}:([^$]+)', slug)
    return m.group(1) if m else None


def _solve_elastic_net(
    S: np.ndarray,
    row_marginals: np.ndarray,
    col_marginals: np.ndarray,
    l1_ratio: float = 0.5,
    huber_eps: float = 1.0,
) -> np.ndarray:
    """Minimize elastic net objective subject to row/col marginals and X >= 0.

    Objective (normalized by spend scale to keep gradients well-conditioned):
        (1 - l1_ratio) * ||X - S||_F^2  +  l1_ratio * sum(pseudo_huber(X - S, eps))

    pseudo_huber(r, eps) = eps^2 * (sqrt(1 + (r/eps)^2) - 1)  →  |r| as eps→0,
    r^2/2 near zero. Smooth everywhere, so SLSQP can use exact gradients.

    l1_ratio=0  →  pure L2 (Ridge): minimise distance to spend
    l1_ratio=1  →  pure pseudo-Huber (robust L1): less sensitive to spend outliers
    l1_ratio=0.5 →  balanced elastic net (default)

    Falls back to IPF result if SLSQP fails to converge.
    """
    K, M = S.shape
    n = K * M
    S_flat = S.flatten()

    # Normalise so objective values are O(1) regardless of spend magnitude
    scale = S_flat[S_flat > 0].mean() if (S_flat > 0).any() else 1.0

    def objective(x):
        d = (x - S_flat) / scale
        l2 = np.sum(d ** 2)
        huber = np.sum(huber_eps ** 2 * (np.sqrt(1.0 + (d / huber_eps) ** 2) - 1.0))
        return (1.0 - l1_ratio) * l2 + l1_ratio * huber

    def gradient(x):
        d = (x - S_flat) / scale
        g_l2 = 2.0 * d / scale
        g_huber = d / (scale * np.sqrt(1.0 + (d / huber_eps) ** 2))
        return (1.0 - l1_ratio) * g_l2 + l1_ratio * g_huber

    # Equality constraints: row and col marginals
    constraints = []
    for i in range(K):
        def _row(x, _i=i):
            return x[_i * M:(_i + 1) * M].sum() - row_marginals[_i]
        def _row_jac(x, _i=i):
            g = np.zeros(n)
            g[_i * M:(_i + 1) * M] = 1.0
            return g
        constraints.append({"type": "eq", "fun": _row, "jac": _row_jac})

    for j in range(M):
        def _col(x, _j=j):
            return x[_j::M].sum() - col_marginals[_j]
        def _col_jac(x, _j=j):
            g = np.zeros(n)
            g[_j::M] = 1.0
            return g
        constraints.append({"type": "eq", "fun": _col, "jac": _col_jac})

    bounds = [(0.0, None)] * n

    # Warm-start from IPF (5 iterations) — already feasible
    x0, _ = run_ipf(S, row_marginals, col_marginals, max_iter=5)

    result = minimize(
        objective, x0.flatten(), jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not result.success:
        fallback, _ = run_ipf(S, row_marginals, col_marginals)
        return fallback

    return result.x.reshape(K, M)


def compute_cross_attribution(
    results,
    nested_dim: str,
    cross_dim: str,
    cross_spend_df: pd.DataFrame,
    slug_to_pair: dict[str, tuple[str, str]],
    method: str = "ipf",
    l1_ratio: float = 0.5,
    huber_eps: float = 1.0,
) -> pd.DataFrame:
    """Compute nested × cross-dim contribution matrix.

    Two solvers available via `method`:

    ``"ipf"`` (default)
        Iterative Proportional Fitting. Minimises KL(X ‖ spend) subject to
        marginal constraints. Fast, parameter-free, recommended for most cases.

    ``"elastic_net"``
        Minimises (1-l1_ratio)·‖X-S‖² + l1_ratio·pseudo_huber(X-S) subject
        to the same constraints. Use when spend has extreme per-cell outliers
        (high l1_ratio → more robust). Solved via SLSQP.

        Parameters:
            l1_ratio (float, 0–1): 0 = pure L2, 1 = pure pseudo-Huber/L1.
            huber_eps (float): smoothing radius for pseudo-Huber (same units
                as spend; default 1.0 means cells within ±1 behave quadratically).

    Both solvers respect:
      - Row marginals: total DD contribution per nested variable slug.
      - Col marginals: total DD contribution per cross-dim value, normalised
        to the nested total (two models may have slightly different proxy_ratios).
      - Zero spend cells → zero contribution (never purchased = never attributed).

    Args:
        results: DDResult from run_deep_dive_e1.
        nested_dim: e.g. "Ambiente" or "Praca".
        cross_dim: e.g. "Midia".
        cross_spend_df: DataFrame returned by load_breakdown_spend on the cross slugs.
        slug_to_pair: mapping cross_slug → (base_nested_slug, cross_value),
            from build_cross_slugs.
        method: ``"ipf"`` or ``"elastic_net"``.
        l1_ratio: elastic net mixing parameter (ignored for IPF).
        huber_eps: pseudo-Huber smoothing radius (ignored for IPF).

    Returns:
        DataFrame with index=nested_slugs, columns=observed_cross_values,
        values=contributions. Only cross values that appear in slug_to_pair
        are included (spec values with no spend data are dropped).
    """
    if method not in ("ipf", "elastic_net"):
        raise ValueError(f"method must be 'ipf' or 'elastic_net', got {method!r}")

    nested_totals = extract_dim_totals(results, nested_dim)
    cross_totals = extract_dim_totals(results, cross_dim)

    vspec = results.config.vehicle_spec
    cross_cat = vspec["breakdowns"][cross_dim]["category"]

    nested_slugs = list(nested_totals.index)
    cross_values = sorted({cv for _, cv in slug_to_pair.values()})

    # ── Spend matrix ──────────────────────────────────────────────────────────
    total_spend = cross_spend_df.sum()
    ns_idx = {s: i for i, s in enumerate(nested_slugs)}
    cv_idx = {v: i for i, v in enumerate(cross_values)}

    S = np.zeros((len(nested_slugs), len(cross_values)))
    for cross_slug, (base_slug, cv) in slug_to_pair.items():
        if cross_slug in total_spend.index and base_slug in ns_idx and cv in cv_idx:
            S[ns_idx[base_slug], cv_idx[cv]] = total_spend[cross_slug]

    nested_contrib = np.array([nested_totals.get(s, 0.0) for s in nested_slugs])

    cross_contrib_raw: dict[str, float] = {}
    for slug, val in cross_totals.items():
        v = _slug_value(slug, cross_cat)
        if v:
            cross_contrib_raw[v] = float(val)
    cross_contrib = np.array([cross_contrib_raw.get(v, 0.0) for v in cross_values])

    # Normalise col marginals to match row total
    nested_total = nested_contrib.sum()
    cross_total = cross_contrib.sum()
    if cross_total > 0:
        cross_contrib = cross_contrib * (nested_total / cross_total)

    # ── Solve ─────────────────────────────────────────────────────────────────
    if method == "ipf":
        result, n_iter = run_ipf(S, nested_contrib, cross_contrib)
        print(f"IPF [{nested_dim} × {cross_dim}]: converged in {n_iter} iterations")
    else:
        result = _solve_elastic_net(S, nested_contrib, cross_contrib, l1_ratio, huber_eps)
        print(f"ElasticNet [{nested_dim} × {cross_dim}]: l1_ratio={l1_ratio}")

    return pd.DataFrame(result, index=nested_slugs, columns=cross_values)


def compute_cross_roi(
    attr_df: pd.DataFrame,
    cross_spend_df: pd.DataFrame,
    slug_to_pair: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """ROI matrix for a nested × cross-dim attribution.

    Args:
        attr_df: contribution DataFrame from compute_cross_attribution
            (index=nested_slugs, columns=cross_values).
        cross_spend_df: raw spend DataFrame returned by load_breakdown_spend.
        slug_to_pair: mapping cross_slug → (base_nested_slug, cross_value).

    Returns:
        DataFrame with same shape as attr_df; values = contribution / spend.
        Cells where spend == 0 are NaN.
    """
    nested_slugs = list(attr_df.index)
    cross_values = list(attr_df.columns)

    ns_idx = {s: i for i, s in enumerate(nested_slugs)}
    cv_idx = {v: i for i, v in enumerate(cross_values)}

    total_spend = cross_spend_df.sum()
    S = np.zeros((len(nested_slugs), len(cross_values)))
    for cross_slug, (base_slug, cv) in slug_to_pair.items():
        if cross_slug in total_spend.index and base_slug in ns_idx and cv in cv_idx:
            S[ns_idx[base_slug], cv_idx[cv]] = total_spend[cross_slug]

    with np.errstate(invalid="ignore", divide="ignore"):
        roi = np.where(S > 0, attr_df.values / S, np.nan)

    return pd.DataFrame(roi, index=nested_slugs, columns=cross_values)


def friendly_index(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Replace slug index with the breakdown value for readability."""
    new_index = [_slug_value(s, category) or s for s in df.index]
    return df.set_axis(new_index, axis=0)
