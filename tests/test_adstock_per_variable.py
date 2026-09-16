"""Tests for upper_funnel_adstock_effect_per_dim (per-variable adstock,
mmmverse >= "feat: customize adstock per variable" (#137)).

Fast tests: signature/wiring, no model fitting.
Slow test: @pytest.mark.slow — actual MAP optimization on synthetic data,
confirms two channels with different configured memory actually decay
differently after their fitted contribution.
"""
import inspect
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
import pandas as pd
import pytest

from config import DeepDiveConfig


# ── fast: signature/wiring ───────────────────────────────────────────────────

def test_deepdive_config_has_upper_funnel_adstock_effect_per_dim():
    sig = inspect.signature(DeepDiveConfig)
    assert "upper_funnel_adstock_effect_per_dim" in sig.parameters
    cfg = DeepDiveConfig(dims=["d"], vars_per_dim={"d": ["v1"]}, media_var="m")
    assert cfg.upper_funnel_adstock_effect_per_dim == {}


def test_run_raven_dim_accepts_upper_funnel_adstock_effect():
    from pipeline import _run_raven_dim

    sig = inspect.signature(_run_raven_dim)
    assert "upper_funnel_adstock_effect" in sig.parameters
    assert sig.parameters["upper_funnel_adstock_effect"].default is None


def test_run_diagnostics_preserves_upper_funnel_adstock_effect_per_dim():
    from diagnostics import run_diagnostics
    from extraction import UpgradeResult

    idx = pd.date_range("2023-01-02", periods=52, freq="W-MON")
    rng = np.random.default_rng(0)
    slug_a = "$metric:m$category:cat:a"
    slug_b = "$metric:m$category:cat:b"
    spend = pd.DataFrame({slug_a: rng.random(52) * 1000, slug_b: rng.random(52) * 1000}, index=idx)
    marker = object()
    cfg = DeepDiveConfig(
        dims=["dim1"],
        vars_per_dim={"dim1": [slug_a, slug_b]},
        media_var="total",
        share_likelihood_metric="m",
        auxiliary_metric="m",
        upper_funnel_adstock_effect_per_dim={"dim1": marker},
    )
    contrib_df = spend.copy()
    contrib_df["total"] = rng.random(52) * 100
    upgrade = UpgradeResult(model=None, contrib_df=contrib_df, spend_df=spend, mmm_config={}, y_hat=None)

    new_cfg, _ = run_diagnostics(cfg, upgrade)
    assert new_cfg.upper_funnel_adstock_effect_per_dim == {"dim1": marker}


# ── slow: actual fit, real Raven object ──────────────────────────────────────

@pytest.mark.slow
def test_per_variable_adstock_end_to_end_produces_different_decay():
    """v1 gets almost no memory (max_lag=1), v2 gets long memory (max_lag=30).
    Both fed the same spend series that stops halfway through -- v2 must keep
    contributing well after v1 has gone to zero."""
    from prophetverse.effects import WeibullAdstockEffect
    from pipeline import run_deep_dive
    from diagnostics import run_diagnostics
    from extraction import UpgradeResult

    T = 60
    idx = pd.date_range("2023-01-02", periods=T, freq="W-MON")
    rng = np.random.default_rng(0)
    base = rng.uniform(100, 1000, T)
    base[30:] = 0  # spend stops halfway

    slug_v1 = "$metric:invest$category:cat:v1"
    slug_v2 = "$metric:invest$category:cat:v2"
    spend = pd.DataFrame({slug_v1: base, slug_v2: base}, index=idx)
    media_total = spend.sum(axis=1) * 0.3
    contrib_df = spend.copy()
    contrib_df["media_total"] = media_total

    upgrade = UpgradeResult(
        model=None, contrib_df=contrib_df, spend_df=spend, mmm_config={}, y_hat=None,
    )
    config = DeepDiveConfig(
        dims=["TestDim"],
        vars_per_dim={"TestDim": [slug_v1, slug_v2]},
        media_var="media_total",
        share_likelihood_metric="invest",
        auxiliary_metric="invest",
        num_steps=300,
        upper_funnel_adstock_effect_per_dim={
            "TestDim": {
                slug_v1: WeibullAdstockEffect(max_lag=1),
                slug_v2: WeibullAdstockEffect(max_lag=30),
            },
        },
    )
    config, diag = run_diagnostics(config, upgrade, min_spend_share=0.0)
    result = run_deep_dive(config, upgrade, diag.auxiliary_metric_dfs, verbose=False)

    contribs = result.contribs["TestDim"]
    tail_v1 = float(contribs[slug_v1].iloc[31:].sum())
    tail_v2 = float(contribs[slug_v2].iloc[31:].sum())
    assert tail_v2 > tail_v1
