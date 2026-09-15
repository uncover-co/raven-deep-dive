"""Smoke tests: does build_config/_build_vars_per_dim correctly construct slugs
for arbitrary vehicle_specs.yaml patterns — not just the 3 real vehicles
(eletromidia, tiktok, tiktok_stellantis) currently in data/vehicle_specs.yaml?

No network, no MLflow, no model fitting — pure slug-construction logic
against fictitious vehicle specs. Complements the real-data batch runs
(Track 2/3 spike) that already proved this for the 3 real vehicles; this
file is the fast, offline, always-runnable version covering patterns those
runs didn't happen to exercise (e.g. 3+ metrics, breakdown-level template
override combined with a custom placeholder, missing-placeholder failure).

Templates are a single value per vehicle/breakdown, independent of
model_type -- confirmed with real data (see docstring in config._get_template)
that the Uncover Webserver API parses slug segments as an unordered
key:value set, not a literal string match, so a stan/meridian split in
models: or a breakdown template only ever reordered the same segments.
metrics: (which metrics get fetched) is likewise independent of model_type
-- the vehicle's default_metric plus the client's auxiliary_metric, when
set; model_type only picks which upstream MLflow artifacts to load.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pandas as pd
import pytest
import yaml

from config import build_config, _build_vars_per_dim


class _FakeUpgrade:
    contrib_df = pd.DataFrame({"total": [1.0, 2.0]})


def _write_specs(tmp_path, vehicle_key: str, vehicle_spec: dict) -> str:
    path = tmp_path / "vehicle_specs.yaml"
    path.write_text(yaml.dump({"vehicles": {vehicle_key: vehicle_spec}}))
    return str(path)


def _write_client(tmp_path, filename="client.yaml", **fields) -> str:
    path = tmp_path / filename
    path.write_text(yaml.dump(fields))
    return str(path)


# ── Case A: single metric, single template, only {brand} placeholder ────────

def test_basic_single_metric_single_template(tmp_path):
    vehicle_spec = {
        "vehicle_slug": "fake",
        "default_metric": "spend",
        "models": {"default_template": "$metric:{metric}$category:brand:{brand}$category:{category}:{value}"},
        "breakdowns": {"Region": {"category": "region", "values": ["north", "south", "east"]}},
    }
    specs_path = _write_specs(tmp_path, "fake_basic", vehicle_spec)
    client_path = _write_client(
        tmp_path, brand="acme", vehicle="fake_basic", vehicle_specs_path=os.path.basename(specs_path),
        model_type="stan", media_var="total",
    )
    config = build_config(_FakeUpgrade(), client_path)
    assert config.dims == ["Region"]
    assert config.vars_per_dim["Region"] == [
        "$metric:spend$category:brand:acme$category:region:north",
        "$metric:spend$category:brand:acme$category:region:south",
        "$metric:spend$category:brand:acme$category:region:east",
    ]


# ── Case B: default_metric + client's auxiliary_metric cross product ────────

def test_default_metric_plus_auxiliary_metric_cross_product(tmp_path):
    vehicle_spec = {
        "vehicle_slug": "fake",
        "default_metric": "spend",
        "models": {"default_template": "$metric:{metric}$category:{category}:{value}"},
        "breakdowns": {"Channel": {"category": "channel", "values": ["a", "b"]}},
    }
    specs_path = _write_specs(tmp_path, "fake_multi", vehicle_spec)
    client_path = _write_client(
        tmp_path, vehicle="fake_multi", vehicle_specs_path=os.path.basename(specs_path),
        model_type="stan", media_var="total", auxiliary_metric="reach",
    )
    config = build_config(_FakeUpgrade(), client_path)
    # 2 metrics (default + auxiliary) x 2 values, order = outer loop over metrics
    assert config.vars_per_dim["Channel"] == [
        "$metric:spend$category:channel:a",
        "$metric:spend$category:channel:b",
        "$metric:reach$category:channel:a",
        "$metric:reach$category:channel:b",
    ]


# ── Case C: custom template placeholder beyond metric/vehicle/brand/category/value ──

def test_custom_placeholder_from_client_cfg(tmp_path):
    vehicle_spec = {
        "vehicle_slug": "fake",
        "default_metric": "spend",
        "models": {"default_template": "$metric:{metric}$_connection:{conn_id}$nameplate:{nameplate}$category:{category}:{value}"},
        "breakdowns": {"Seg": {"category": "seg", "values": ["x", "y"]}},
    }
    specs_path = _write_specs(tmp_path, "fake_custom", vehicle_spec)
    client_path = _write_client(
        tmp_path, vehicle="fake_custom", vehicle_specs_path=os.path.basename(specs_path),
        model_type="stan", media_var="total",
        conn_id="abc-123", nameplate="model,trim",
    )
    config = build_config(_FakeUpgrade(), client_path)
    assert config.vars_per_dim["Seg"] == [
        "$metric:spend$_connection:abc-123$nameplate:model,trim$category:seg:x",
        "$metric:spend$_connection:abc-123$nameplate:model,trim$category:seg:y",
    ]


def test_missing_custom_placeholder_raises_clear_error(tmp_path):
    """If the client YAML doesn't supply a field the template needs, this must
    fail loudly (KeyError from str.format), not silently produce a wrong slug."""
    vehicle_spec = {
        "vehicle_slug": "fake",
        "default_metric": "spend",
        "models": {"default_template": "$metric:{metric}$nameplate:{nameplate}$category:{category}:{value}"},
        "breakdowns": {"Seg": {"category": "seg", "values": ["x"]}},
    }
    specs_path = _write_specs(tmp_path, "fake_missing", vehicle_spec)
    client_path = _write_client(
        tmp_path, vehicle="fake_missing", vehicle_specs_path=os.path.basename(specs_path),
        model_type="stan", media_var="total",
        # nameplate deliberately omitted
    )
    with pytest.raises(KeyError):
        build_config(_FakeUpgrade(), client_path)


# ── Case D: breakdown-level template override + state_template priority ─────

def test_breakdown_level_template_override(tmp_path):
    vehicle_spec = {
        "vehicle_slug": "fake",
        "default_metric": "spend",
        "models": {
            "default_template": "$metric:{metric}$category:brand:{brand}$category:{category}:{value}",
            "state_template": "$metric:{metric}$state:{value}$category:brand:{brand}",
        },
        "breakdowns": {
            "State": {"category": "state", "values": ["sp", "rj"]},
            "Custom": {
                "category": "custom",
                "template": "$metric:{metric}$custom-slug:{value}",
                "values": ["p"],
            },
        },
    }
    specs_path = _write_specs(tmp_path, "fake_override", vehicle_spec)
    client_path = _write_client(
        tmp_path, brand="acme", vehicle="fake_override", vehicle_specs_path=os.path.basename(specs_path),
        model_type="stan", media_var="total",
    )
    config = build_config(_FakeUpgrade(), client_path)
    # State breakdown (category=="state") uses state_template, not default_template
    assert config.vars_per_dim["State"] == [
        "$metric:spend$state:sp$category:brand:acme",
        "$metric:spend$state:rj$category:brand:acme",
    ]
    # Custom breakdown's own `template` wins over both vehicle templates
    assert config.vars_per_dim["Custom"] == ["$metric:spend$custom-slug:p"]


# ── Case E: model_dims default selection vs explicit `dimensions:` override ─

def test_model_dims_default_vs_explicit_dimensions_override(tmp_path):
    vehicle_spec = {
        "vehicle_slug": "fake",
        "default_metric": "spend",
        "model_dims": ["A", "B"],
        "models": {"default_template": "$metric:{metric}$category:{category}:{value}"},
        "breakdowns": {
            "A": {"category": "a", "values": ["1"]},
            "B": {"category": "b", "values": ["1"]},
            "C": {"category": "c", "values": ["1"]},
        },
    }
    specs_path = _write_specs(tmp_path, "fake_dims", vehicle_spec)

    # No `dimensions:` in client cfg -> falls back to vehicle's model_dims (A, B), not C
    client_default = _write_client(
        tmp_path, vehicle="fake_dims", vehicle_specs_path=os.path.basename(specs_path),
        model_type="stan", media_var="total",
    )
    config_default = build_config(_FakeUpgrade(), client_default)
    assert set(config_default.dims) == {"A", "B"}

    # Explicit `dimensions: [C]` in client cfg overrides model_dims entirely
    client_override_path = _write_client(
        tmp_path, filename="client_override.yaml",
        vehicle="fake_dims", vehicle_specs_path=os.path.basename(specs_path),
        model_type="stan", media_var="total", dimensions=["C"],
    )
    config_override = build_config(_FakeUpgrade(), client_override_path)
    assert config_override.dims == ["C"]


# ── Case F: template and metrics are the same regardless of model_type ──────

def test_template_and_metrics_independent_of_model_type(tmp_path):
    """models: is a single flat template, not keyed by stan/meridian, and
    metrics fetched = default_metric + auxiliary_metric (client cfg) — same
    slugs come out no matter what model_type the client declares, since
    model_type only picks which upstream MLflow artifacts to load."""
    vehicle_spec = {
        "vehicle_slug": "fake",
        "default_metric": "spend",
        "models": {"default_template": "$metric:{metric}$marker$category:{category}:{value}"},
        "breakdowns": {"Ch": {"category": "ch", "values": ["a"]}},
    }
    specs_path = _write_specs(tmp_path, "fake_dual", vehicle_spec)

    stan_client = _write_client(
        tmp_path, filename="client_stan.yaml",
        vehicle="fake_dual", vehicle_specs_path=os.path.basename(specs_path),
        model_type="stan", media_var="total", auxiliary_metric="reach",
    )
    meridian_client = _write_client(
        tmp_path, filename="client_meridian.yaml",
        vehicle="fake_dual", vehicle_specs_path=os.path.basename(specs_path),
        model_type="meridian", media_var="total", auxiliary_metric="reach",
    )

    config_stan = build_config(_FakeUpgrade(), stan_client)
    config_meridian = build_config(_FakeUpgrade(), meridian_client)
    assert config_stan.vars_per_dim["Ch"] == config_meridian.vars_per_dim["Ch"] == [
        "$metric:spend$marker$category:ch:a",
        "$metric:reach$marker$category:ch:a",
    ]


# ── Case G: missing required breakdown fields raise clear errors ────────────

def test_breakdown_missing_category_raises():
    vehicle_spec = {
        "vehicle_slug": "fake",
        "default_metric": "spend",
        "models": {"default_template": "$metric:{metric}$category:{category}:{value}"},
        "breakdowns": {"Bad": {"values": ["a"]}},  # no "category"
    }
    with pytest.raises(ValueError, match="category"):
        _build_vars_per_dim(vehicle_spec, {}, None)


def test_breakdown_missing_values_raises():
    vehicle_spec = {
        "vehicle_slug": "fake",
        "default_metric": "spend",
        "models": {"default_template": "$metric:{metric}$category:{category}:{value}"},
        "breakdowns": {"Bad": {"category": "bad"}},  # no "values"
    }
    with pytest.raises(ValueError, match="values"):
        _build_vars_per_dim(vehicle_spec, {}, None)
