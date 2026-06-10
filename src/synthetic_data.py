"""Semi-synthetic data generator for Deep Dive E1 validation.

Uses real spend patterns with synthetic Hill parameters to create
ground-truth contribution breakdowns for controlled methodology evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SyntheticDimension:
    dim_name: str
    variaveis: list[str]
    spend_df: pd.DataFrame          # raw spend (T x K), original index
    hill_params: pd.DataFrame       # known θ_k: columns me/hm/sl, index=variaveis
    contributions: pd.DataFrame     # synthetic c_kt (T x K), same index as spend_df
    eletro_contrib: pd.Series       # synthetic C_t (T,), same index as spend_df
    true_shares: pd.Series          # σ_k^true (K,), index=variaveis, sums to 1
    col_maxes: pd.Series            # max spend per channel (K,), index=variaveis


def hill(x: np.ndarray, max_effect: float, half_max: float, slope: float) -> np.ndarray:
    """Hill saturation: me * x^sl / (hm^sl + x^sl + 1e-12). Clips x >= 0."""
    x = np.clip(x, 0, None)
    denom = half_max ** slope + x ** slope + 1e-12
    return max_effect * x ** slope / denom


def generate_synthetic_dim(
    dim_name: str,
    spend_df: pd.DataFrame,
    hill_params: dict | None = None,
    noise_sigma: float = 0.02,
    rng_seed: int = 42,
) -> SyntheticDimension:
    """
    Generate synthetic contributions using real spend patterns.

    spend_df: (T, K) DataFrame of real spend values per sub-channel.
    hill_params: optional {var: {me, hm, sl}} dict. If None, samples from:
      me_k ~ Uniform(0.05/K, 0.35/K)
      hm_k ~ Uniform(0.20, 0.65)
      sl_k ~ Uniform(0.8, 2.2)
    noise_sigma: std of multiplicative noise on C_t (0.02 = 2%)
    """
    rng = np.random.default_rng(rng_seed)
    variaveis = list(spend_df.columns)
    K = len(variaveis)

    # col_maxes: replace 0 with 1 to avoid division by zero
    col_maxes = spend_df.max(axis=0).replace(0, 1.0)

    # x_kt: normalized spend in [0, 1]
    x_kt = spend_df.div(col_maxes)

    # Build hill_params DataFrame, sampling if not provided
    if hill_params is None:
        records = {}
        for v in variaveis:
            records[v] = {
                "me": float(rng.uniform(0.05 / K, 0.35 / K)),
                "hm": float(rng.uniform(0.20, 0.65)),
                "sl": float(rng.uniform(0.8, 2.2)),
            }
        hill_params_dict = records
    else:
        hill_params_dict = {v: dict(hill_params[v]) for v in variaveis}

    params_df = pd.DataFrame(hill_params_dict).T[["me", "hm", "sl"]]
    params_df.index.name = None

    # c_kt = hill(x_kt, me_k, hm_k, sl_k) applied column-by-column
    contribs_data = {}
    for v in variaveis:
        me = params_df.loc[v, "me"]
        hm = params_df.loc[v, "hm"]
        sl = params_df.loc[v, "sl"]
        contribs_data[v] = hill(x_kt[v].values, me, hm, sl)

    contributions = pd.DataFrame(contribs_data, index=spend_df.index)

    # eletro_contrib = sum_k c_kt * (1 + N(0, noise_sigma)) clipped >= 0
    noise = rng.normal(0.0, noise_sigma, size=len(spend_df))
    raw_sum = contributions.sum(axis=1).values
    eletro_vals = np.clip(raw_sum * (1.0 + noise), 0, None)
    eletro_contrib = pd.Series(eletro_vals, index=spend_df.index, name="eletro")

    # true_shares = contributions.sum(axis=0) / contributions.sum().sum()
    total_per_channel = contributions.sum(axis=0)
    grand_total = total_per_channel.sum()
    if grand_total > 0:
        true_shares = total_per_channel / grand_total
    else:
        true_shares = pd.Series(1.0 / K, index=variaveis)

    return SyntheticDimension(
        dim_name=dim_name,
        variaveis=variaveis,
        spend_df=spend_df.copy(),
        hill_params=params_df,
        contributions=contributions,
        eletro_contrib=eletro_contrib,
        true_shares=true_shares,
        col_maxes=col_maxes,
    )


def simulate_measurement_prior(
    true_shares: pd.Series,
    n_obs: int,
    sigma: float = 0.05,
    rng_seed: int = 0,
    index=None,
) -> pd.DataFrame:
    """
    Simulate noisy measurement data (brand study/GRP) as CSL metric_df.

    Returns (n_obs, K) DataFrame where column k has constant value ŝ_k.
    ŝ = softmax(log(true_shares) + N(0, sigma)).
    With sigma=0: ŝ = true_shares (deterministic).

    index: optional DatetimeIndex — MUST match the y2 index used in _run_raven2_eletro
           so that auxiliary_metric_df.reindex(y2.index) aligns correctly.
           If None, uses RangeIndex — caller is responsible for alignment.
    """
    rng = np.random.default_rng(rng_seed)
    variaveis = list(true_shares.index)
    K = len(variaveis)

    # log-probability space: softmax(log(p)) = p, so sigma=0 recovers true_shares exactly.
    # Binary logit log(p/(1-p)) is NOT the inverse of softmax for K>2 — use log(p) instead.
    p = np.clip(true_shares.values.astype(float), 1e-9, None)
    logits = np.log(p)

    if sigma > 0:
        logits = logits + rng.normal(0.0, sigma, size=K)

    # softmax
    logits_shifted = logits - logits.max()
    exp_logits = np.exp(logits_shifted)
    s_hat = exp_logits / exp_logits.sum()

    # Return (n_obs, K) DataFrame with constant columns
    data = np.tile(s_hat, (n_obs, 1))
    return pd.DataFrame(data, columns=variaveis, index=index)
