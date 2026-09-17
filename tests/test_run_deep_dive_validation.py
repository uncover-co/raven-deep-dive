"""Tests for run_deep_dive()'s auxiliary_metric_dfs validation -- both raises
happen before _run_raven_dim (no MAP fit), so these are fast, no mocking needed."""
import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from config import DeepDiveConfig
from extraction import UpgradeResult
from pipeline import run_deep_dive


def _fixtures():
    idx = pd.date_range("2023-01-02", periods=26, freq="W-MON")
    rng = np.random.default_rng(0)
    spend = pd.DataFrame(
        {"v1": rng.uniform(100, 1000, 26), "v2": rng.uniform(100, 1000, 26)}, index=idx
    )
    contrib = (spend.sum(axis=1) * 0.3).rename("total")
    cfg = DeepDiveConfig(
        dims=["Dim"],
        vars_per_dim={"Dim": ["v1", "v2"]},
        media_var="total",
    )
    upgrade = UpgradeResult(
        model=None,
        contrib_df=contrib.to_frame(),
        spend_df=spend,
        mmm_config={},
        y_hat=contrib,
    )
    return cfg, upgrade


def test_run_deep_dive_raises_when_auxiliary_metric_dfs_missing_dim():
    cfg, upgrade = _fixtures()
    with pytest.raises(ValueError, match=r"\[Dim\] auxiliary_metric_dfs has no entry"):
        run_deep_dive(cfg, upgrade, auxiliary_metric_dfs={}, verbose=False)


def test_run_deep_dive_raises_when_auxiliary_metric_dfs_missing_column():
    cfg, upgrade = _fixtures()
    aux = pd.DataFrame({"v1": np.ones(len(upgrade.spend_df))}, index=upgrade.spend_df.index)  # v2 missing
    with pytest.raises(ValueError, match=r"\[Dim\] auxiliary_metric_dfs is missing column"):
        run_deep_dive(cfg, upgrade, auxiliary_metric_dfs={"Dim": aux}, verbose=False)
