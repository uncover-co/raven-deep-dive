"""Tests for check_spend_coverage / plot_spend_coverage.

The check reconciles the sum of the Deep Dive's breakdown spend against the
vehicle's total, from either the workspace (queried without the breakdown
segment) or the main model's own investment column.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
import pandas as pd
import pytest

from config import DeepDiveConfig
from diagnostics import check_spend_coverage, _vehicle_level_filter, _to_weekly
from extraction import UpgradeResult

P = "$metric:m$category:cat:"


def _cfg(vars_=None):
    return DeepDiveConfig(
        dims=["d"], vars_per_dim={"d": vars_ or [P + "a", P + "b"]},
        media_var="t", share_likelihood_metric="m", auxiliary_metric="m",
    )


def _upgrade(index, values, col="spend"):
    inp = pd.DataFrame({"timestamp": index, col: values})
    return UpgradeResult(
        model=None, contrib_df=pd.DataFrame(), spend_df=pd.DataFrame(),
        mmm_config={}, input_df=inp,
    )


# ── pure helpers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("template,expected", [
    # {value} last
    ("$metric:{metric}$vehicle:{vehicle}$category:brand:{brand}$category:{category}:{value}",
     "$metric:{metric}$vehicle:{vehicle}$category:brand:{brand}"),
    # {value} in the middle (state_template)
    ("$metric:{metric}$state:{value}$vehicle:{vehicle}$category:brand:{brand}",
     "$metric:{metric}$vehicle:{vehicle}$category:brand:{brand}"),
    # {value} in the middle (tiktok campaign_category)
    ("$metric:{metric}$category:brand:{brand}$category:{category}:{value}$category:product-level-1:tiktok",
     "$metric:{metric}$category:brand:{brand}$category:product-level-1:tiktok"),
])
def test_vehicle_level_filter_drops_the_value_segment_wherever_it_sits(template, expected):
    assert _vehicle_level_filter(template) == expected


def test_to_weekly_buckets_by_week_end_not_week_start():
    """W-MON periods END on Monday, so a Monday closes its own week and the
    days after it roll into the next one. Two sources anchored differently
    therefore line up only if both go through this."""
    idx = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-09"])
    out = _to_weekly(pd.Series([1.0, 2.0, 4.0], index=idx))

    assert list(out.index.date.astype(str)) == ["2023-01-02", "2023-01-09"]
    assert out.iloc[0] == 1.0          # the Monday closes its own week
    assert out.iloc[1] == 6.0          # 2.0 (Jan 3) + 4.0 (Jan 9)


# ── windowing ────────────────────────────────────────────────────────────────

def test_totals_are_compared_only_on_the_shared_weeks():
    """The main model is usually trained on a wider window than the Deep Dive.
    Summing each series whole reported that difference as missing spend."""
    full = pd.date_range("2022-01-03", periods=200, freq="W-MON")
    up = _upgrade(full, [100.0] * 200)
    dd = full[-100:]
    spend = pd.DataFrame({P + "a": [100.0] * 100}, index=dd)

    r = check_spend_coverage(
        _cfg([P + "a"]), spend, against="upgrade",
        upgrade=up, upgrade_spend_col="spend", verbose=False,
    )
    assert r["weeks_compared"].iloc[0] == 100
    assert r["gap_pct"].iloc[0] == pytest.approx(0.0)
    assert r["ref_total_full"].iloc[0] == pytest.approx(20000.0)  # full sum kept


def test_gap_is_reported_when_breakdowns_really_are_short():
    idx = pd.date_range("2023-01-02", periods=10, freq="W-MON")
    up = _upgrade(idx, [100.0] * 10)
    spend = pd.DataFrame({P + "a": [80.0] * 10}, index=idx)

    r = check_spend_coverage(
        _cfg([P + "a"]), spend, against="upgrade",
        upgrade=up, upgrade_spend_col="spend", verbose=False,
    )
    assert r["gap_pct"].iloc[0] == pytest.approx(0.2)


def test_zero_reference_with_real_spend_is_not_full_coverage(capsys):
    """gap_pct was 0.0 and corr NaN, so an empty reference passed as perfect
    coverage with no warning at all."""
    idx = pd.date_range("2023-01-02", periods=10, freq="W-MON")
    up = _upgrade(idx, [0.0] * 10)
    spend = pd.DataFrame({P + "a": [80.0] * 10}, index=idx)

    r = check_spend_coverage(
        _cfg([P + "a"]), spend, against="upgrade",
        upgrade=up, upgrade_spend_col="spend", verbose=True,
    )
    assert pd.isna(r["gap_pct"].iloc[0])
    assert "no reference spend" in capsys.readouterr().out


def test_zero_on_both_sides_stays_zero():
    idx = pd.date_range("2023-01-02", periods=10, freq="W-MON")
    up = _upgrade(idx, [0.0] * 10)
    spend = pd.DataFrame({P + "a": [0.0] * 10}, index=idx)

    r = check_spend_coverage(
        _cfg([P + "a"]), spend, against="upgrade",
        upgrade=up, upgrade_spend_col="spend", verbose=False,
    )
    assert r["gap_pct"].iloc[0] == pytest.approx(0.0)


def test_others_bucket_counts_as_breakdown_spend():
    """Running after run_diagnostics, the bucket carries real spend; leaving it
    out understated the breakdown sum by the whole bucket."""
    idx = pd.date_range("2023-01-02", periods=10, freq="W-MON")
    up = _upgrade(idx, [100.0] * 10)
    spend = pd.DataFrame({P + "a": [60.0] * 10, "__others__d": [40.0] * 10}, index=idx)

    r = check_spend_coverage(
        _cfg([P + "a", "__others__d"]), spend, against="upgrade",
        upgrade=up, upgrade_spend_col="spend", verbose=False,
    )
    assert r["gap_pct"].iloc[0] == pytest.approx(0.0)


# ── input validation ─────────────────────────────────────────────────────────

def test_rejects_unknown_reference():
    with pytest.raises(ValueError, match="against must be one of"):
        check_spend_coverage(_cfg(), pd.DataFrame(), against="upgrades", verbose=False)


def test_workspace_reference_requires_workspace_and_dates():
    """Without the dates the reference would cover a different window than the
    breakdown, which is exactly the bug the windowing fix addresses."""
    with pytest.raises(ValueError, match="start_date"):
        check_spend_coverage(_cfg(), pd.DataFrame(), against="workspace",
                             workspace="ws", verbose=False)


def test_upgrade_reference_requires_the_column_name():
    with pytest.raises(ValueError, match="upgrade_spend_col"):
        check_spend_coverage(_cfg(), pd.DataFrame(), against="upgrade",
                             upgrade=_upgrade(pd.date_range("2023-01-02", periods=2, freq="W-MON"),
                                              [1.0, 2.0]), verbose=False)


def test_unknown_upgrade_column_lists_candidates():
    idx = pd.date_range("2023-01-02", periods=2, freq="W-MON")
    up = _upgrade(idx, [1.0, 2.0], col="investments_foo")
    with pytest.raises(ValueError, match="not a column of input_data"):
        check_spend_coverage(_cfg(), pd.DataFrame(), against="upgrade",
                             upgrade=up, upgrade_spend_col="nope", verbose=False)

# ── plot ─────────────────────────────────────────────────────────────────────

def test_plot_spend_coverage_needs_the_series():
    from plots import plot_spend_coverage

    with pytest.raises(ValueError, match="no series"):
        plot_spend_coverage(pd.DataFrame({"dim": ["d"], "reference": ["upgrade"]}))


def test_plot_spend_coverage_styles_every_subplot():
    """_styled() used to hardcode xaxis/xaxis2, leaving subplots 3+ on plotly's
    default white grid."""
    from plots import plot_spend_coverage

    idx = pd.date_range("2023-01-02", periods=10, freq="W-MON")
    ref = pd.Series(np.arange(10, dtype=float), index=idx)
    series, rows = {}, []
    for dim in ["a", "b", "c", "d"]:
        series[(dim, "upgrade")] = (ref, ref * 0.9)
        rows.append({"dim": dim, "reference": "upgrade"})
    report = pd.DataFrame(rows)
    report.attrs["series"] = series

    fig = plot_spend_coverage(report)
    grids = {fig.layout[k].gridcolor for k in fig.layout
             if k.startswith(("xaxis", "yaxis"))}
    assert grids == {"#2A2A2A"}


# ── silent-display regressions found by the sweep ────────────────────────────

def test_sunburst_keeps_a_leaf_that_belongs_to_no_group():
    """`__others__<dim>` is never in the spec's `grupos:`, so the tree dropped
    it -- shrinking the root and renormalising every percent and ROAS colour."""
    from plots import _groups_sunburst

    groups = {
        "g1": {"vertical": "ruas", "tipo": "outdoor", "ambientes": ["a1", "a2"]},
        "g2": {"vertical": "transportes", "tipo": "indoor", "ambientes": ["a3"]},
    }
    pfx = "$metric:m$category:ambiente:"
    sm = pd.Series({pfx + "a1": 0.25, pfx + "a2": 0.25, pfx + "a3": 0.20,
                    "__others__ambiente": 0.30})
    ss = pd.Series({pfx + "a1": 0.20, pfx + "a2": 0.20, pfx + "a3": 0.20,
                    "__others__ambiente": 0.40})

    out = _groups_sunburst(sm, ss, "ambiente", groups, "ambientes")
    root_total = sum(v for p, v in zip(out["parents"], out["values_m"]) if p == "")

    assert "__others__ambiente" in out["labels"]
    assert root_total == pytest.approx(1.0)
    i = out["labels"].index("a1")
    assert out["values_m"][i] / out["values_s"][i] == pytest.approx(1.25)


def test_diagnosis_table_prints_the_bucket_line(capsys):
    """The `→ others` line read spend_report, which never gains an __others__
    row, so it never printed."""
    from config import DeepDiveConfig
    from diagnostics import run_diagnostics
    from extraction import UpgradeResult

    idx = pd.date_range("2023-01-02", periods=52, freq="W-MON")
    rng = np.random.default_rng(0)
    pfx = "$metric:m$category:cat:"
    v = [pfx + "big1", pfx + "big2", pfx + "small1", pfx + "small2"]
    spend = pd.DataFrame(
        {v[0]: rng.random(52) * 10000, v[1]: rng.random(52) * 10000,
         v[2]: rng.random(52) * 40, v[3]: rng.random(52) * 40},
        index=idx,
    )
    contrib = spend.copy()
    contrib["total"] = rng.random(52) * 100
    up = UpgradeResult(model=None, contrib_df=contrib, spend_df=spend,
                       mmm_config={})

    run_diagnostics(
        DeepDiveConfig(dims=["dim1"], vars_per_dim={"dim1": v}, media_var="total",
                       share_likelihood_metric="m", auxiliary_metric="m"),
        up,
    )
    out = capsys.readouterr().out
    assert "→ __others__" in out
    assert "'Others' groups: 1" in out      # the summary counter was always 0


def test_others_active_weeks_is_the_union_not_the_max():
    """Members are bucketed for being sparse, so their active weeks are usually
    disjoint; a max reads as 'this bucket barely ran'."""
    from report import _others_active_weeks

    class _Diag:
        bucketed_raw = {
            "Praca": pd.DataFrame({
                "date": pd.to_datetime(
                    ["2023-01-02"] * 2 + ["2023-01-09"] * 2 + ["2023-01-16"] * 2
                ),
                "variable": ["a", "b"] * 3,
                "investment": [10.0, 0.0, 0.0, 10.0, 10.0, 0.0],
            })
        }

    base = pd.DataFrame({"active_weeks": [2, 1]})
    assert _others_active_weeks(_Diag(), "Praca", base) == 3   # union, not max=2
