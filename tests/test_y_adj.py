"""Tests for y_adj strategy in run_deep_dive_e1."""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from extraction import UpgradeResult
from config import DeepDiveConfig
from pipeline import run_deep_dive_e1


def _make_upgrade(with_y_actual: bool = True, wape: float = 0.0):
    """Fake UpgradeResult with optional y_actual."""
    idx = pd.date_range("2023-01-02", periods=26, freq="W-MON")
    rng = np.random.default_rng(0)

    eletro = pd.Series(rng.uniform(50, 200, 26), index=idx, name="eletro")
    other = pd.Series(rng.uniform(100, 400, 26), index=idx)
    y_hat = eletro + other  # fitted = eletro + outros canais

    # y_actual = y_hat + wape-level noise
    noise = rng.normal(0, wape * float(y_hat.mean()), 26)
    y_actual = (y_hat + noise).clip(lower=0)

    contrib_df = pd.DataFrame({"eletro": eletro, "other": other}, index=idx)
    spend_df = pd.DataFrame({
        "v1": rng.uniform(100, 500, 26),
        "v2": rng.uniform(50, 300, 26),
    }, index=idx)

    return UpgradeResult(
        model=None,
        contrib_df=contrib_df,
        spend_df=spend_df,
        mmm_config={},
        y_hat=y_hat,
        y_actual=y_actual if with_y_actual else None,
    )


def _make_config():
    return DeepDiveConfig(
        dims=["Praca"],
        vars_per_dim={"Praca": ["v1", "v2"]},
        media_var="eletro",
        share_prior_scale=0.05,
        proxy_ct_tolerance=0.15,
        num_steps=10,
    )


# ── unit: y_adj formula ───────────────────────────────────────────────────────

def test_y_adj_formula_zero_wape():
    """When WAPE=0 (y_actual == y_hat), y_adj == eletro_contrib exactly."""
    upgrade = _make_upgrade(with_y_actual=True, wape=0.0)
    eletro = upgrade.contrib_df["eletro"]
    y_adj = (upgrade.y_actual - upgrade.y_hat + eletro).clip(lower=0)
    assert (y_adj - eletro).abs().max() < 1e-6


def test_y_adj_formula_with_residual():
    """When y_actual != y_hat, y_adj carries the residual."""
    idx = pd.date_range("2023-01-02", periods=5, freq="W-MON")
    y_hat = pd.Series([100.0, 200.0, 150.0, 180.0, 120.0], index=idx)
    y_actual = pd.Series([110.0, 190.0, 160.0, 175.0, 130.0], index=idx)
    eletro = pd.Series([30.0, 60.0, 45.0, 55.0, 35.0], index=idx)

    y_adj = (y_actual - y_hat + eletro).clip(lower=0)
    expected = pd.Series([40.0, 50.0, 55.0, 50.0, 45.0], index=idx)
    assert (y_adj - expected).abs().max() < 1e-6


def test_y_adj_clips_negatives():
    """Negative y_adj values (large negative residual) clipped to 0."""
    idx = pd.date_range("2023-01-02", periods=3, freq="W-MON")
    y_hat = pd.Series([500.0, 500.0, 500.0], index=idx)
    y_actual = pd.Series([400.0, 500.0, 600.0], index=idx)  # -100 residual in week 0
    eletro = pd.Series([50.0, 50.0, 50.0], index=idx)

    y_adj = (y_actual - y_hat + eletro).clip(lower=0)
    assert float(y_adj.iloc[0]) == pytest.approx(0.0)   # -50 clipped to 0
    assert float(y_adj.iloc[1]) == pytest.approx(50.0)  # no residual
    assert float(y_adj.iloc[2]) == pytest.approx(150.0) # +100 residual


# ── unit: run_deep_dive_e1 strategy param ────────────────────────────────────

def test_run_deep_dive_e1_raises_without_y_actual():
    """strategy='y_adj' must raise ValueError if y_actual is None."""
    upgrade = _make_upgrade(with_y_actual=False)
    config = _make_config()
    with pytest.raises(ValueError, match="y_actual"):
        run_deep_dive_e1(config, upgrade, strategy="y_adj")


def test_run_deep_dive_e1_invalid_strategy():
    """Unknown strategy string raises immediately."""
    upgrade = _make_upgrade(with_y_actual=True)
    config = _make_config()
    with pytest.raises((ValueError, TypeError)):
        run_deep_dive_e1(config, upgrade, strategy="invalid_strategy")
