"""Tests for semi-synthetic deep dive validation framework.

Tests 1-4: pure numpy/pandas, fast (no model fitting).
Test 5: @pytest.mark.slow — actual MAP optimization (500 steps, ~minutes).
"""
import sys
import os
import inspect
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from synthetic_data import (
    SyntheticDimension,
    hill,
    generate_synthetic_dim,
    simulate_measurement_prior,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _small_spend_df(T=26, K=3, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=T, freq="W-MON")
    cols = [f"v{i+1}" for i in range(K)]
    return pd.DataFrame(rng.uniform(100, 1000, (T, K)), index=idx, columns=cols)


# ── Test 1: hill() basic properties ──────────────────────────────────────────

def test_hill_basic():
    # hill(0) == 0 regardless of params
    assert hill(np.array([0.0]), 1.0, 0.5, 1.0)[0] == pytest.approx(0.0, abs=1e-9)

    # half-max property: hill(hm) = me/2
    val = hill(np.array([0.5]), max_effect=1.0, half_max=0.5, slope=1.0)[0]
    assert val == pytest.approx(0.5, abs=1e-6)

    # strictly increasing on (0, 1]
    x = np.linspace(0, 1, 100)
    y = hill(x, max_effect=2.0, half_max=0.4, slope=1.5)
    assert (np.diff(y) >= -1e-10).all(), "hill must be non-decreasing"

    # asymptote approaches max_effect
    y_large = hill(np.array([1e6]), max_effect=3.0, half_max=0.5, slope=2.0)[0]
    assert abs(y_large - 3.0) < 0.001

    # negative x clipped to 0 → hill == 0
    assert hill(np.array([-5.0]), 1.0, 0.5, 1.0)[0] == pytest.approx(0.0, abs=1e-9)


# ── Test 2: generate_synthetic_dim ───────────────────────────────────────────

def test_generate_synthetic_dim():
    spend_df = _small_spend_df(T=26, K=3)
    syn = generate_synthetic_dim("TestDim", spend_df, rng_seed=0)

    assert isinstance(syn, SyntheticDimension)
    assert syn.contributions.shape == (26, 3)
    assert syn.eletro_contrib.shape == (26,)
    assert list(syn.contributions.columns) == ["v1", "v2", "v3"]
    assert list(syn.hill_params.index) == ["v1", "v2", "v3"]
    assert all(c in syn.hill_params.columns for c in ["me", "hm", "sl"])

    # true shares sum to 1
    assert syn.true_shares.sum() == pytest.approx(1.0, abs=1e-6)
    assert (syn.true_shares > 0).all()

    # eletro_contrib ≈ contributions.sum(axis=1) within noise tolerance
    rel_err = (
        (syn.eletro_contrib - syn.contributions.sum(axis=1)).abs()
        / (syn.eletro_contrib.abs() + 1e-12)
    )
    assert rel_err.max() < 0.15, f"noise too large: {rel_err.max():.4f}"

    # eletro_contrib non-negative
    assert (syn.eletro_contrib >= 0).all()

    # custom hill_params respected
    params = {"v1": {"me": 0.1, "hm": 0.3, "sl": 1.0},
              "v2": {"me": 0.05, "hm": 0.5, "sl": 1.5},
              "v3": {"me": 0.08, "hm": 0.4, "sl": 0.9}}
    syn2 = generate_synthetic_dim("TestDim2", spend_df, hill_params=params, rng_seed=1)
    assert syn2.hill_params.loc["v1", "me"] == pytest.approx(0.1)
    assert syn2.hill_params.loc["v2", "hm"] == pytest.approx(0.5)


# ── Test 3: simulate_measurement_prior ───────────────────────────────────────

def test_simulate_measurement_prior():
    true_shares = pd.Series({"a": 0.5, "b": 0.3, "c": 0.2})

    # sigma=0 → perfect recovery
    prior = simulate_measurement_prior(true_shares, n_obs=26, sigma=0.0, rng_seed=0)
    assert prior.shape == (26, 3)
    assert list(prior.columns) == ["a", "b", "c"]

    # all rows identical (constant over time)
    assert (prior.diff().dropna().abs() < 1e-10).all().all()

    # each row sums to 1
    assert (prior.sum(axis=1) - 1.0).abs().max() < 1e-6

    # sigma=0 recovers true_shares
    assert (prior.iloc[0] - true_shares).abs().max() < 1e-6

    # sigma>0: still normalized, non-negative
    noisy = simulate_measurement_prior(true_shares, n_obs=26, sigma=5.0, rng_seed=42)
    assert (noisy.sum(axis=1) - 1.0).abs().max() < 1e-6
    assert (noisy >= 0).all().all()
    assert noisy.shape == (26, 3)

    # deterministic with same seed
    p1 = simulate_measurement_prior(true_shares, n_obs=10, sigma=0.1, rng_seed=7)
    p2 = simulate_measurement_prior(true_shares, n_obs=10, sigma=0.1, rng_seed=7)
    assert (p1 - p2).abs().max().max() < 1e-12


# ── Test 4: pipeline accepts auxiliary_metric_df params ──────────────────────

def test_pipeline_accepts_auxiliary_df():
    from pipeline import _run_raven2_eletro, run_deep_dive_e1

    sig_run = inspect.signature(_run_raven2_eletro)
    assert "auxiliary_metric_df" in sig_run.parameters, (
        "_run_raven2_eletro missing auxiliary_metric_df param"
    )
    assert sig_run.parameters["auxiliary_metric_df"].default is None

    sig_e1 = inspect.signature(run_deep_dive_e1)
    assert "auxiliary_metric_dfs" in sig_e1.parameters, (
        "run_deep_dive_e1 missing auxiliary_metric_dfs param"
    )
    assert sig_e1.parameters["auxiliary_metric_dfs"].default is None


# ── Test 5: integration — share recovery with auxiliary prior (slow) ──────────

@pytest.mark.slow
def test_share_recovery_integration():
    """Verify deep dive recovers synthetic true shares.

    Runs actual MAP optimization (500 steps). Expects both baseline and
    perfect-prior runs to produce shares within 0.40 MAE of ground truth.
    """
    import jax
    from pipeline import _run_raven2_eletro

    spend_df = _small_spend_df(T=26, K=3, seed=99)
    syn = generate_synthetic_dim("Praca", spend_df, rng_seed=42)

    common_kwargs = dict(
        dim_name="Praca",
        features_df=syn.spend_df,
        eletro_contrib=syn.eletro_contrib,
        share_prior_scale=0.05,
        proxy_ct_tolerance=0.15,
        num_steps=500,
        verbose=False,
    )

    # baseline: spend-based CSL (no auxiliary prior)
    r_baseline = _run_raven2_eletro(**common_kwargs)

    # perfect measurement prior (sigma=0 → exact true shares)
    aux_perfect = simulate_measurement_prior(
        syn.true_shares, n_obs=len(spend_df), sigma=0.0, index=spend_df.index
    )
    r_perfect = _run_raven2_eletro(**common_kwargs, auxiliary_metric_df=aux_perfect)

    mae_baseline = float((r_baseline["shares_model"] - syn.true_shares).abs().mean())
    mae_perfect = float((r_perfect["shares_model"] - syn.true_shares).abs().mean())

    print(f"\nTrue shares:    {syn.true_shares.round(3).to_dict()}")
    print(f"Baseline shares:{r_baseline['shares_model'].round(3).to_dict()}")
    print(f"Perfect shares: {r_perfect['shares_model'].round(3).to_dict()}")
    print(f"MAE baseline={mae_baseline:.3f}  perfect={mae_perfect:.3f}  Δ={mae_baseline-mae_perfect:.3f}")

    assert mae_baseline < 0.40, f"Baseline too far from truth: MAE={mae_baseline:.3f}"
    assert mae_perfect < 0.40, f"Perfect prior too far from truth: MAE={mae_perfect:.3f}"