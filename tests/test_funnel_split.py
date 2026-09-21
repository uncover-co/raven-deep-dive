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


# ── __others__ bucket funnel classification ──────────────────────────────────

def _bucket_fixture():
    """4 sub-channels: 2 survive the gate, 2 fall under 2% -> __others__ bucket."""
    from extraction import UpgradeResult

    idx = pd.date_range("2023-01-02", periods=52, freq="W-MON")
    rng = np.random.default_rng(0)
    P = "$metric:m$category:cat:"
    big1, big2, sm1, sm2 = P + "big1", P + "big2", P + "small1", P + "small2"
    spend = pd.DataFrame(
        {
            big1: rng.random(52) * 10000, big2: rng.random(52) * 10000,
            sm1: rng.random(52) * 40, sm2: rng.random(52) * 40,
        },
        index=idx,
    )
    contrib = spend.copy()
    contrib["total"] = rng.random(52) * 100
    upgrade = UpgradeResult(
        model=None, contrib_df=contrib, spend_df=spend, mmm_config={}, y_hat=None
    )
    return upgrade, (big1, big2, sm1, sm2)


def _run(upgrade, slugs, lower):
    from diagnostics import run_diagnostics

    cfg = DeepDiveConfig(
        dims=["dim1"], vars_per_dim={"dim1": list(slugs)}, media_var="total",
        share_likelihood_metric="m", auxiliary_metric="m",
        lower_funnel_vars_per_dim=lower,
    )
    return run_diagnostics(cfg, upgrade)


def test_mixed_bucket_defaults_to_upper_funnel():
    """A bucket whose members are a mix of lower and upper funnel has no
    unambiguous classification -> stays upper (adstocked)."""
    upgrade, (big1, big2, sm1, sm2) = _bucket_fixture()
    new_cfg, diag = _run(upgrade, (big1, big2, sm1, sm2), {"dim1": [sm1]})

    assert diag.bucketed["dim1"] == [sm1, sm2]
    assert "__others__dim1" in new_cfg.vars_per_dim["dim1"]
    assert new_cfg.lower_funnel_vars_per_dim.get("dim1", []) == []


def test_homogeneous_lower_bucket_inherits_lower_funnel():
    upgrade, (big1, big2, sm1, sm2) = _bucket_fixture()
    new_cfg, _ = _run(upgrade, (big1, big2, sm1, sm2), {"dim1": [sm1, sm2]})

    assert new_cfg.lower_funnel_vars_per_dim["dim1"] == ["__others__dim1"]


def test_bucket_declared_in_yaml_survives_diagnostics():
    """The bucket name is deterministic, so a client YAML may declare it up
    front. That declaration must survive -- it is the DS's override of the
    mixed-bucket default, and the only way to set it before the fit."""
    upgrade, (big1, big2, sm1, sm2) = _bucket_fixture()
    new_cfg, _ = _run(
        upgrade, (big1, big2, sm1, sm2), {"dim1": [sm1, "__others__dim1"]}
    )

    assert new_cfg.lower_funnel_vars_per_dim["dim1"] == ["__others__dim1"]


def test_declared_bucket_that_never_forms_is_a_silent_noop():
    """Only one sub-channel fails the gate -> no bucket (a one-member bucket
    would be a rename, not an aggregation). A YAML that declared the bucket
    anyway must not break the run; the pipeline filters it out at fit time."""
    from extraction import UpgradeResult

    idx = pd.date_range("2023-01-02", periods=52, freq="W-MON")
    rng = np.random.default_rng(0)
    P = "$metric:m$category:cat:"
    big1, big2, sm1 = P + "big1", P + "big2", P + "small1"
    spend = pd.DataFrame(
        {big1: rng.random(52) * 10000, big2: rng.random(52) * 10000,
         sm1: rng.random(52) * 40},
        index=idx,
    )
    contrib = spend.copy()
    contrib["total"] = rng.random(52) * 100
    upgrade = UpgradeResult(
        model=None, contrib_df=contrib, spend_df=spend, mmm_config={}, y_hat=None
    )
    new_cfg, diag = _run(upgrade, (big1, big2, sm1), {"dim1": ["__others__dim1"]})

    assert "dim1" not in diag.bucketed
    assert new_cfg.vars_per_dim["dim1"] == [big1, big2]   # sm1 dropped entirely
    assert new_cfg.lower_funnel_vars_per_dim["dim1"] == ["__others__dim1"]


# ── override_funnel: the post-diagnostics declaration API ────────────────────

P = "$metric:m$category:cat:"


def _cfg(variables=None):
    variables = variables or [P + "a", P + "b", P + "c", "__others__dim1"]
    return DeepDiveConfig(
        dims=["dim1"], vars_per_dim={"dim1": list(variables)}, media_var="total",
        share_likelihood_metric="m", auxiliary_metric="m",
    )


def test_override_funnel_accepts_short_label_and_full_slug():
    from config import override_funnel

    cfg = _cfg()
    override_funnel(cfg, "dim1", lower=["a", P + "b"], verbose=False)

    assert cfg.lower_funnel_vars_per_dim["dim1"] == [P + "a", P + "b"]


def test_override_funnel_fills_unlisted_upper_with_default():
    """Raven needs the dict to cover every upper-funnel variable, so the ones
    the DS didn't mention get the same shape Raven would have built."""
    from config import override_funnel, DEFAULT_MAX_LAG
    from prophetverse.effects import WeibullAdstockEffect

    cfg = _cfg()
    override_funnel(
        cfg, "dim1",
        lower=["__others__dim1"],
        adstock={"a": WeibullAdstockEffect(max_lag=4)},
        verbose=False,
    )

    eff = cfg.upper_funnel_adstock_effect_per_dim["dim1"]
    assert set(eff) == {P + "a", P + "b", P + "c"}        # covers every upper
    assert eff[P + "a"].max_lag == 4                       # explicit one kept
    assert eff[P + "b"].max_lag == DEFAULT_MAX_LAG         # rest filled in


def test_override_funnel_keeps_a_non_weibull_effect_as_given():
    from config import override_funnel
    from prophetverse.effects import GeometricAdstockEffect

    cfg = _cfg()
    override_funnel(cfg, "dim1", adstock={"a": GeometricAdstockEffect()}, verbose=False)

    assert isinstance(
        cfg.upper_funnel_adstock_effect_per_dim["dim1"][P + "a"], GeometricAdstockEffect
    )


def test_override_funnel_without_adstock_leaves_it_untouched():
    from config import override_funnel

    cfg = _cfg()
    override_funnel(cfg, "dim1", lower=["a"], verbose=False)

    assert cfg.upper_funnel_adstock_effect_per_dim == {}


def test_override_funnel_rejects_a_raw_max_lag():
    """An int used to be accepted as shorthand; it hid which distribution you
    were picking, so the API now asks for the effect itself."""
    from config import override_funnel

    with pytest.raises(TypeError, match="effect instances"):
        override_funnel(_cfg(), "dim1", adstock={"a": 4}, verbose=False)


def test_override_funnel_rejects_unknown_variable():
    from config import override_funnel

    with pytest.raises(ValueError, match="not a variable of this dimension"):
        override_funnel(_cfg(), "dim1", lower=["nope"], verbose=False)


def test_override_funnel_rejects_adstock_on_a_lower_funnel_variable():
    from config import override_funnel
    from prophetverse.effects import WeibullAdstockEffect

    with pytest.raises(ValueError, match="lower funnel has no adstock"):
        override_funnel(
            _cfg(), "dim1", lower=["a"],
            adstock={"a": WeibullAdstockEffect(max_lag=4)}, verbose=False,
        )


def test_override_funnel_rejects_unknown_dimension():
    from config import override_funnel

    with pytest.raises(ValueError, match="not a modelled dimension"):
        override_funnel(_cfg(), "nao_existe", lower=[], verbose=False)


def test_override_funnel_rejects_ambiguous_label():
    from config import override_funnel

    cfg = _cfg([P + "a", "$metric:m$category:outra:a", P + "b"])
    with pytest.raises(ValueError, match="ambiguous"):
        override_funnel(cfg, "dim1", lower=["a"], verbose=False)


def test_override_funnel_without_lower_keeps_the_current_classification():
    """Calling it only to tweak adstock used to wipe the funnel split the
    diagnostics had just established -- silently, changing the fit."""
    from config import override_funnel
    from prophetverse.effects import WeibullAdstockEffect

    cfg = _cfg()
    cfg.lower_funnel_vars_per_dim["dim1"] = ["__others__dim1"]
    override_funnel(cfg, "dim1", adstock={"a": WeibullAdstockEffect(max_lag=4)},
                    verbose=False)

    assert cfg.lower_funnel_vars_per_dim["dim1"] == ["__others__dim1"]


def test_override_funnel_with_empty_lower_clears_the_classification():
    from config import override_funnel

    cfg = _cfg()
    cfg.lower_funnel_vars_per_dim["dim1"] = ["__others__dim1"]
    override_funnel(cfg, "dim1", lower=[], verbose=False)

    assert cfg.lower_funnel_vars_per_dim["dim1"] == []


def test_override_funnel_prunes_a_stale_adstock_dict():
    """Moving a variable to lower funnel leaves a dict adstock covering it;
    Raven only notices at fit time, and blames diagnostics for it."""
    from config import override_funnel
    from prophetverse.effects import WeibullAdstockEffect

    cfg = _cfg()
    cfg.upper_funnel_adstock_effect_per_dim["dim1"] = {
        v: WeibullAdstockEffect(max_lag=4) for v in cfg.vars_per_dim["dim1"]
    }
    override_funnel(cfg, "dim1", lower=["__others__dim1"], verbose=False)

    upper = {v for v in cfg.vars_per_dim["dim1"] if v != "__others__dim1"}
    assert set(cfg.upper_funnel_adstock_effect_per_dim["dim1"]) == upper


def test_override_funnel_rejects_a_bare_string_for_lower():
    """lower="x" would iterate characters and fail with '_' is not a variable."""
    from config import override_funnel

    with pytest.raises(TypeError, match="not a string"):
        override_funnel(_cfg(), "dim1", lower="__others__dim1", verbose=False)
