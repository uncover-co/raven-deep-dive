import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from config import DeepDiveConfig
from extraction import UpgradeResult
from diagnostics import DiagnosisResult, run_diagnostics


def _make_fixtures():
    idx = pd.date_range("2023-01-02", periods=52, freq="W-MON")
    rng = np.random.default_rng(42)
    spend = pd.DataFrame({
        "$metric:invest$category:praca:sp":  rng.random(52) * 1000,
        "$metric:invest$category:praca:rj":  rng.random(52) * 200,
        "$metric:invest$category:praca:rec": rng.random(52) * 5,   # < 2% → excluded
        "$metric:invest$category:praca:go":  rng.random(52) * 5,   # < 2% → excluded
    }, index=idx)
    cfg = DeepDiveConfig(
        dims=["Praca"],
        vars_per_dim={"Praca": list(spend.columns)},
        media_var="eletro_total",
        share_likelihood_metric="invest",
        auxiliary_metric="invest",
    )
    eletro = pd.Series(rng.random(52) * 100, index=idx, name="eletro_total")
    contrib_df = spend.copy()
    contrib_df["eletro_total"] = eletro
    upgrade = UpgradeResult(
        model=None,
        contrib_df=contrib_df,
        spend_df=spend,
        mmm_config={},
    )
    return cfg, upgrade


def test_run_diagnostics_returns_types():
    cfg, upgrade = _make_fixtures()
    new_cfg, diag = run_diagnostics(cfg, upgrade, min_spend_share=0.02)
    assert isinstance(diag, DiagnosisResult)
    assert isinstance(new_cfg, DeepDiveConfig)


def test_tiny_vars_bucketed_into_others():
    cfg, upgrade = _make_fixtures()
    new_cfg, diag = run_diagnostics(cfg, upgrade, min_spend_share=0.02)
    praca_vars = new_cfg.vars_per_dim.get("Praca", [])
    # rec, go < 2% → should NOT be in kept vars
    assert "$metric:invest$category:praca:rec" not in praca_vars
    assert "$metric:invest$category:praca:go" not in praca_vars
    # 2+ excluded vars → __others__ column should be added
    assert any("__others__" in v for v in praca_vars)
    assert diag.bucketed.get("Praca") == [
        "$metric:invest$category:praca:rec", "$metric:invest$category:praca:go",
    ]


def test_bucketed_raw_has_original_per_member_series():
    """diag.bucketed_raw exposes the original (pre-aggregation) investment +
    auxiliary_metric series for each __others__ member, for audit purposes."""
    cfg, upgrade = _make_fixtures()
    _, diag = run_diagnostics(cfg, upgrade, min_spend_share=0.02)
    df = diag.bucketed_raw["Praca"]
    assert set(df.columns) == {"date", "variable", "investment", "auxiliary_metric"}
    assert set(df["variable"]) == {
        "$metric:invest$category:praca:rec", "$metric:invest$category:praca:go",
    }
    # 52 weeks x 2 bucketed members
    assert len(df) == 104


def test_others_inherits_lower_funnel_when_bucket_fully_lower():
    """Both bucketed members (rec, go) are configured lower funnel -> the
    __others__ aggregate is unambiguous, inherits lower funnel too."""
    cfg, upgrade = _make_fixtures()
    cfg.lower_funnel_vars_per_dim = {
        "Praca": ["$metric:invest$category:praca:rec", "$metric:invest$category:praca:go"],
    }
    new_cfg, _ = run_diagnostics(cfg, upgrade, min_spend_share=0.02)
    others_col = next(v for v in new_cfg.vars_per_dim["Praca"] if v.startswith("__others__"))
    assert new_cfg.lower_funnel_vars_per_dim["Praca"] == [others_col]


def test_others_stays_upper_when_bucket_mixed():
    """Only one of the two bucketed members (rec) is configured lower funnel
    -> no unambiguous classification, __others__ defaults to upper funnel
    (i.e. does NOT get added to lower_funnel_vars_per_dim)."""
    cfg, upgrade = _make_fixtures()
    cfg.lower_funnel_vars_per_dim = {
        "Praca": ["$metric:invest$category:praca:rec"],
    }
    new_cfg, _ = run_diagnostics(cfg, upgrade, min_spend_share=0.02)
    others_col = next(v for v in new_cfg.vars_per_dim["Praca"] if v.startswith("__others__"))
    assert others_col not in new_cfg.lower_funnel_vars_per_dim.get("Praca", [])
    # rec was bucketed away (no longer a standalone slug) -> stale entry dropped
    assert new_cfg.lower_funnel_vars_per_dim.get("Praca", []) == []


def test_single_excluded_var_not_bucketed():
    idx = pd.date_range("2023-01-02", periods=52, freq="W-MON")
    rng = np.random.default_rng(42)
    spend = pd.DataFrame({
        "$metric:invest$category:praca:sp":  rng.random(52) * 1000,
        "$metric:invest$category:praca:rj":  rng.random(52) * 200,
        "$metric:invest$category:praca:rec": rng.random(52) * 5,   # < 2%, sole exclusion → just skipped
    }, index=idx)
    cfg = DeepDiveConfig(
        dims=["Praca"],
        vars_per_dim={"Praca": list(spend.columns)},
        media_var="eletro_total",
        share_likelihood_metric="invest",
        auxiliary_metric="invest",
    )
    eletro = pd.Series(rng.random(52) * 100, index=idx, name="eletro_total")
    contrib_df = spend.copy()
    contrib_df["eletro_total"] = eletro
    upgrade = UpgradeResult(
        model=None,
        contrib_df=contrib_df,
        spend_df=spend,
        mmm_config={},
    )
    new_cfg, diag = run_diagnostics(cfg, upgrade, min_spend_share=0.02)
    praca_vars = new_cfg.vars_per_dim.get("Praca", [])
    # a single excluded var isn't a "group" → no __others__, just dropped
    assert "$metric:invest$category:praca:rec" not in praca_vars
    assert not any("__others__" in v for v in praca_vars)
    assert "Praca" not in diag.bucketed


def test_slug_without_primary_column_excluded_even_if_aux_available():
    """Regression: a slug can pass the exposure-based gate (real impressions)
    while its investment column doesn't exist at all (dropped upstream as
    all-zero) -- must not end up "kept", or auxiliary_metric_dfs ends up with
    a column that pipeline.py's real feature set (filtered to what actually
    exists in spend_df) doesn't have, desyncing metric_df from
    target_effect_names during CSL fitting."""
    idx = pd.date_range("2023-01-02", periods=52, freq="W-MON")
    rng = np.random.default_rng(7)
    spend = pd.DataFrame({
        "$metric:invest$category:praca:sp": rng.random(52) * 1000,
        "$metric:invest$category:praca:rj": rng.random(52) * 1000,
        "$metric:impr$category:praca:sp":   rng.random(52) * 500,
        "$metric:impr$category:praca:rj":   rng.random(52) * 500,
        # "ghost": real impressions but no investment column at all --
        # mirrors load_breakdown_spend dropping an all-zero primary column.
        "$metric:impr$category:praca:ghost": rng.random(52) * 500,
    }, index=idx)
    cfg = DeepDiveConfig(
        dims=["Praca"],
        vars_per_dim={"Praca": [
            "$metric:invest$category:praca:sp",
            "$metric:invest$category:praca:rj",
            "$metric:invest$category:praca:ghost",
        ]},
        media_var="eletro_total",
        share_likelihood_metric="invest",
        auxiliary_metric="impr",
    )
    eletro = pd.Series(rng.random(52) * 100, index=idx, name="eletro_total")
    contrib_df = spend.copy()
    contrib_df["eletro_total"] = eletro
    upgrade = UpgradeResult(model=None, contrib_df=contrib_df, spend_df=spend, mmm_config={})

    new_cfg, diag = run_diagnostics(cfg, upgrade, min_spend_share=0.02)
    ghost_slug = "$metric:invest$category:praca:ghost"
    assert ghost_slug not in new_cfg.vars_per_dim["Praca"]
    aux_df = diag.auxiliary_metric_dfs["Praca"]
    assert ghost_slug not in aux_df.columns


def test_raises_when_auxiliary_metric_has_no_real_data_for_dim():
    cfg, upgrade = _make_fixtures()
    cfg.auxiliary_metric = "impr"  # no $metric:impr$... columns exist in spend
    with pytest.raises(ValueError, match="auxiliary_metric 'impr' has no real data"):
        run_diagnostics(cfg, upgrade, min_spend_share=0.02)


def test_raises_when_auxiliary_metric_not_set_on_config():
    cfg, upgrade = _make_fixtures()
    cfg.auxiliary_metric = ""
    with pytest.raises(ValueError, match="config.auxiliary_metric is not set"):
        run_diagnostics(cfg, upgrade, min_spend_share=0.02)


def test_spend_report_columns():
    cfg, upgrade = _make_fixtures()
    _, diag = run_diagnostics(cfg, upgrade, min_spend_share=0.02)
    expected_cols = {"dim", "slug", "gate_total", "pct_gate_dim", "active_weeks", "hhi", "keep"}
    assert expected_cols.issubset(set(diag.spend_report.columns))
