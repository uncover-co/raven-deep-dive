import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from config import DeepDiveConfig
from extraction import UpgradeResult
from diagnostics import DiagnosisResult, run_diagnostics


def _make_fixtures():
    idx = pd.date_range("2023-01-02", periods=52, freq="W-MON")
    rng = np.random.default_rng(42)
    spend = pd.DataFrame({
        "invest:sp":  rng.random(52) * 1000,
        "invest:rj":  rng.random(52) * 200,
        "invest:rec": rng.random(52) * 5,   # < 2% → outros
    }, index=idx)
    cfg = DeepDiveConfig(
        dims=["Praca"],
        vars_per_dim={"Praca": ["invest:sp", "invest:rj", "invest:rec"]},
        media_var="eletro_total",
    )
    eletro = pd.Series(rng.random(52) * 100, index=idx, name="eletro_total")
    contrib_df = spend.copy()
    contrib_df["eletro_total"] = eletro
    upgrade = UpgradeResult(
        model=None,
        contrib_df=contrib_df,
        spend_df=spend,
        mmm_config={},
        y_hat=eletro,
    )
    return cfg, upgrade


def test_run_diagnostics_returns_types():
    cfg, upgrade = _make_fixtures()
    new_cfg, diag = run_diagnostics(cfg, upgrade, min_spend_share=0.02)
    assert isinstance(diag, DiagnosisResult)
    assert isinstance(new_cfg, DeepDiveConfig)


def test_tiny_var_bucketed_into_outros():
    cfg, upgrade = _make_fixtures()
    new_cfg, diag = run_diagnostics(cfg, upgrade, min_spend_share=0.02)
    praca_vars = new_cfg.vars_per_dim.get("Praca", [])
    # invest:rec < 2% → should NOT be in kept vars
    assert "invest:rec" not in praca_vars
    # __outros__ column should be added
    assert any("__outros__" in v for v in praca_vars)


def test_spend_report_columns():
    cfg, upgrade = _make_fixtures()
    _, diag = run_diagnostics(cfg, upgrade, min_spend_share=0.02)
    expected_cols = {"dim", "slug", "spend_total", "pct_dim", "semanas_ativas", "hhi", "keep"}
    assert expected_cols.issubset(set(diag.spend_report.columns))
