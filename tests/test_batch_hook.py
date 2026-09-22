import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))


def _patched(monkeypatch, recorder):
    """Stub the pipeline around run_single_client, keeping only the hook wiring."""
    import batch

    from types import SimpleNamespace
    cfg_in = SimpleNamespace(vars_per_dim={"dim1": ["a"]}, name="in")
    cfg_out = SimpleNamespace(vars_per_dim={"dim1": ["a"]}, name="out")
    diag = SimpleNamespace(auxiliary_metric_dfs={})
    upgrade = SimpleNamespace(spend_df=None)
    result = SimpleNamespace(name="result")
    monkeypatch.setattr(batch, "_resolve_specs_path", lambda *a, **k: "client.yaml")
    specs = {"workspace_dd": "w", "upgrade_run_id": "r", "media_var": "m",
             "model_type": "stan", "start_date": "2023-01-02", "end_date": "2024-01-01"}
    monkeypatch.setattr(batch, "load_yaml", lambda *a, **k: specs)
    monkeypatch.setattr(batch, "load_upgrade_auto", lambda *a, **k: upgrade)
    monkeypatch.setattr(batch, "load_breakdown_spend", lambda *a, **k: None)
    monkeypatch.setattr(batch, "build_config", lambda *a, **k: cfg_in)
    monkeypatch.setattr(batch, "run_diagnostics", lambda c, u: (cfg_in, diag))
    monkeypatch.setattr(batch, "generate_report", lambda *a, **k: None)

    def _fit(config, up, **kw):
        recorder["config_used"] = config
        return result

    monkeypatch.setattr(batch, "run_deep_dive", _fit)
    return batch, cfg_in, cfg_out, diag, upgrade


def _run(batch, **kw):
    return batch.run_single_client(
        "bradesco_eletro",
        {"specs_path": "x.yaml"},
        registry_path="reg.yaml", output_base_dir="out", verbose=False, **kw
    )


def test_hook_runs_between_diagnosis_and_fit(monkeypatch):
    """The bucket only exists after run_diagnostics, so that is the one moment
    override_funnel can name it."""
    rec = {}
    batch, cfg_in, _, diag, upgrade = _patched(monkeypatch, rec)
    seen = {}

    def hook(name, config, d, up):
        seen.update(name=name, config=config, diag=d, upgrade=up)

    _, _, err = _run(batch, after_diagnostics=hook)

    assert err is None
    assert seen == {"name": "bradesco_eletro", "config": cfg_in,
                    "diag": diag, "upgrade": upgrade}
    assert rec["config_used"] is cfg_in


def test_hook_can_replace_the_config(monkeypatch):
    rec = {}
    batch, _, cfg_out, _, _ = _patched(monkeypatch, rec)

    _run(batch, after_diagnostics=lambda *a: cfg_out)

    assert rec["config_used"] is cfg_out


def test_hook_returning_none_keeps_the_config(monkeypatch):
    """override_funnel mutates in place and the DS may not return anything."""
    rec = {}
    batch, cfg_in, _, _, _ = _patched(monkeypatch, rec)

    _run(batch, after_diagnostics=lambda *a: None)

    assert rec["config_used"] is cfg_in


def test_without_hook_nothing_changes(monkeypatch):
    rec = {}
    batch, cfg_in, _, _, _ = _patched(monkeypatch, rec)

    _, _, err = _run(batch)

    assert err is None
    assert rec["config_used"] is cfg_in


def test_hook_failure_is_reported_as_that_client_error(monkeypatch):
    """A bad override must not take the whole batch down."""
    rec = {}
    batch, _, _, _, _ = _patched(monkeypatch, rec)

    def boom(*a):
        raise ValueError("slug errado")

    result, diag, err = _run(batch, after_diagnostics=boom)

    assert result is None and "slug errado" in err
