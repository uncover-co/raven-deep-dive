"""Tests for lower_funnel_vars_per_dim (Track 3 Fase B — generic, vehicle-agnostic:
the pipeline only ever sees dimension names and variable slugs, never a
funnel/branding/TikTok concept; each client YAML supplies its own real
classification via its own vehicle_specs.yaml hierarchy, or leaves it empty).

Fast tests: pure signature/wiring checks, no model fitting.
Slow test: @pytest.mark.slow — actual MAP optimization on synthetic data.
"""
import inspect
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
import pandas as pd
import pytest

from config import DeepDiveConfig


def _small_spend_df(T=26, K=3, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=T, freq="W-MON")
    cols = [f"v{i+1}" for i in range(K)]
    return pd.DataFrame(rng.uniform(100, 1000, (T, K)), index=idx, columns=cols)


# ── fast: signature/wiring ───────────────────────────────────────────────────

def test_deepdive_config_has_lower_funnel_vars_per_dim():
    sig = inspect.signature(DeepDiveConfig)
    assert "lower_funnel_vars_per_dim" in sig.parameters
    assert sig.parameters["lower_funnel_vars_per_dim"].default is not inspect.Parameter.empty
    cfg = DeepDiveConfig(dims=["d"], vars_per_dim={"d": ["v1"]}, media_var="m")
    assert cfg.lower_funnel_vars_per_dim == {}


def test_run_raven_dim_accepts_lower_funnel_variables():
    from pipeline import _run_raven_dim

    sig = inspect.signature(_run_raven_dim)
    assert "lower_funnel_variables" in sig.parameters
    assert sig.parameters["lower_funnel_variables"].default is None


def test_run_diagnostics_preserves_lower_funnel_vars_per_dim():
    """Regression: run_diagnostics rebuilds DeepDiveConfig from scratch (to
    filter vars_per_dim) — must carry lower_funnel_vars_per_dim through too,
    or any pipeline call going through diagnostics (i.e. every real run)
    silently drops the funnel-split config. Caught by hand: an end-to-end
    run through the real pipeline had lower_funnel_variables == [] despite
    the client YAML setting it, because this field wasn't threaded here."""
    from diagnostics import run_diagnostics
    from extraction import UpgradeResult

    idx = pd.date_range("2023-01-02", periods=52, freq="W-MON")
    rng = np.random.default_rng(0)
    slug_a = "$metric:m$category:cat:a"
    slug_b = "$metric:m$category:cat:b"
    spend = pd.DataFrame({slug_a: rng.random(52) * 1000, slug_b: rng.random(52) * 1000}, index=idx)
    cfg = DeepDiveConfig(
        dims=["dim1"],
        vars_per_dim={"dim1": [slug_a, slug_b]},
        media_var="total",
        share_likelihood_metric="m",
        auxiliary_metric="m",
        lower_funnel_vars_per_dim={"dim1": [slug_b]},
    )
    contrib_df = spend.copy()
    contrib_df["total"] = rng.random(52) * 100
    upgrade = UpgradeResult(model=None, contrib_df=contrib_df, spend_df=spend, mmm_config={}, y_hat=None)

    new_cfg, _ = run_diagnostics(cfg, upgrade)
    assert new_cfg.lower_funnel_vars_per_dim == {"dim1": [slug_b]}


def test_build_config_reads_lower_funnel_vars_per_dim(tmp_path):
    from config import build_config

    vehicle_specs = tmp_path / "vehicle_specs.yaml"
    vehicle_specs.write_text(
        "vehicles:\n"
        "  fake_vehicle:\n"
        "    vehicle_slug: fake\n"
        "    default_metric: investments\n"
        "    models:\n"
        "      default_template: \"$metric:{metric}$category:{category}:{value}\"\n"
        "    breakdowns:\n"
        "      dim1:\n"
        "        category: cat1\n"
        "        values: [a, b]\n"
    )
    client_yaml = tmp_path / "client.yaml"
    client_yaml.write_text(
        "brand: acme\n"
        "vehicle: fake_vehicle\n"
        "vehicle_specs_path: \"vehicle_specs.yaml\"\n"
        "model_type: stan\n"
        "media_var: \"total\"\n"
        "auxiliary_metric: \"investments\"\n"
        "lower_funnel_vars_per_dim:\n"
        "  dim1: [\"$metric:investments$category:cat1:a\"]\n"
    )

    class _FakeUpgrade:
        contrib_df = pd.DataFrame({"total": [1.0, 2.0]})

    config = build_config(_FakeUpgrade(), str(client_yaml))
    assert config.lower_funnel_vars_per_dim == {
        "dim1": ["$metric:investments$category:cat1:a"]
    }


# ── slow: actual fit, real Raven object ──────────────────────────────────────

@pytest.mark.slow
def test_lower_funnel_split_changes_bucket_assignment():
    from pipeline import _run_raven_dim

    spend_df = _small_spend_df(T=26, K=3, seed=1)
    media_dd_contrib = spend_df.sum(axis=1) * 0.3

    r = _run_raven_dim(
        dim_name="TestDim",
        features_df=spend_df,
        media_dd_contrib=media_dd_contrib,
        num_steps=200,
        lower_funnel_variables=["v2"],
        verbose=False,
    )
    model = r["model"]
    assert model.lower_funnel_variables == ["v2"]
    assert set(model.upper_funnel_variables) == {"v1", "v3"}


@pytest.mark.slow
def test_no_lower_funnel_config_matches_current_behavior():
    """Regression: omitting lower_funnel_variables must reproduce the exact
    pre-existing behavior (everything upper funnel/adstocked). MAP is
    deterministic (fixed rng_key=0), so results must match exactly, not just
    approximately."""
    from pipeline import _run_raven_dim

    spend_df = _small_spend_df(T=26, K=3, seed=2)
    media_dd_contrib = spend_df.sum(axis=1) * 0.3
    common = dict(
        dim_name="TestDim", features_df=spend_df, media_dd_contrib=media_dd_contrib,
        num_steps=200, verbose=False,
    )

    r_omitted = _run_raven_dim(**common)
    r_explicit_empty = _run_raven_dim(**common, lower_funnel_variables=[])

    assert r_omitted["model"].lower_funnel_variables == []
    assert set(r_omitted["model"].upper_funnel_variables) == {"v1", "v2", "v3"}
    assert r_omitted["proxy_ratio"] == pytest.approx(r_explicit_empty["proxy_ratio"])
    assert r_omitted["r2"] == pytest.approx(r_explicit_empty["r2"])
    assert (r_omitted["shares_model"] - r_explicit_empty["shares_model"]).abs().max() < 1e-9


@pytest.mark.slow
def test_run_deep_dive_lower_funnel_vars_per_dim_end_to_end():
    from pipeline import run_deep_dive
    from extraction import UpgradeResult

    spend_df = _small_spend_df(T=26, K=3, seed=3)
    media_dd_contrib = spend_df.sum(axis=1) * 0.3
    contrib_df = media_dd_contrib.to_frame(name="media_total")

    upgrade = UpgradeResult(
        model=None, contrib_df=contrib_df, spend_df=spend_df, mmm_config={},
        y_hat=None, model_type="stan",
    )

    slugs = list(spend_df.columns)
    # No real diagnostics run here -- spend itself stands in as its own
    # auxiliary metric, same convention as a client with no exposure data.
    aux_dfs = {"TestDim": spend_df}

    config_split = DeepDiveConfig(
        dims=["TestDim"], vars_per_dim={"TestDim": slugs}, media_var="media_total",
        num_steps=200, lower_funnel_vars_per_dim={"TestDim": ["v2"]},
    )
    result = run_deep_dive(config_split, upgrade, aux_dfs, verbose=False)
    assert result.models["TestDim"].lower_funnel_variables == ["v2"]

    config_default = DeepDiveConfig(
        dims=["TestDim"], vars_per_dim={"TestDim": slugs}, media_var="media_total",
        num_steps=200,
    )
    result_default = run_deep_dive(config_default, upgrade, aux_dfs, verbose=False)
    assert result_default.models["TestDim"].lower_funnel_variables == []
