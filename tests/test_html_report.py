import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from config import DeepDiveConfig
from pipeline import DDResult
from report import generate_report
from report_gen import generate_html_report


def _fake_result(vehicle="veic"):
    idx = pd.date_range("2023-01-02", periods=10, freq="W-MON")
    cols = {"Praca": ["sp", "rj"], "Midia": ["tv", "digital", "radio"]}
    rng = np.random.default_rng(0)
    contribs = {d: pd.DataFrame({c: rng.uniform(0.5, 2, 10) for c in cs}, index=idx) for d, cs in cols.items()}
    inv = {d: pd.DataFrame({c: rng.uniform(5, 20, 10) for c in cs}, index=idx) for d, cs in cols.items()}
    shares = {d: contribs[d].sum() / contribs[d].sum().sum() for d in cols}
    spend = {d: inv[d].sum() / inv[d].sum().sum() for d in cols}
    return DDResult(
        models={},
        contribs=contribs,
        shares_model=shares,
        shares_spend=spend,
        proxy_ratios={"Praca": 0.98, "Midia": 1.4},
        csl_devs={"Praca": 0.04, "Midia": 0.05},
        r2={"Praca": 0.9, "Midia": 0.8},
        wape={"Praca": 0.1, "Midia": 0.2},
        media_dd_contrib=pd.Series(np.ones(10) * 150, index=idx),
        config=DeepDiveConfig(dims=list(cols), vars_per_dim=cols, media_var="eletro", vehicle=vehicle),
        features_raw=inv,
        auxiliary_metric_raw=inv,
    )


def _run_dir(tmp, vehicle="veic", hill=True, diag=True):
    generate_report(_fake_result(vehicle), output_dir=tmp, client_name="Cli")
    pasta = os.path.join(tmp, f"Cli_{vehicle}")
    if hill:
        pd.DataFrame({
            "variable": ["sp", "rj", "tv"], "dim": ["Praca", "Praca", "Midia"],
            "max_effect": [0.5, 0.3, 0.9], "half_max_norm": [0.4, 0.5, 0.3],
            "half_max_spend": [8.0, 10.0, 6.0], "slope": [2.0, 9.0, 1.5],
            "mean_spend": [10.0, 5.0, 12.0], "saturation_at_mean": [0.95, 0.4, 0.5],
            "max_effect_kpi": [75.0, 45.0, 135.0],
        }).to_csv(os.path.join(pasta, "hill_params.csv"), index=False)
    if diag:
        pd.DataFrame({
            "dim": ["Praca", "Praca"], "slug": ["sp", "rj"], "status": ["kept", "discarded_low_weeks"],
            "reason": ["ok", "poucas semanas"], "active_weeks": [10, 2], "gate_total": [100.0, 3.0],
            "pct_gate_dim": [0.97, 0.03], "contrib_total": [10.0, None], "pct_contrib_dim": [0.7, None],
        }).to_csv(os.path.join(pasta, "diagnostics.csv"), index=False)
    return pasta


def test_html_report_has_blocks_and_figures():
    with tempfile.TemporaryDirectory() as tmp:
        pasta = _run_dir(tmp)
        out = generate_html_report([pasta], os.path.join(tmp, "dd.html"))
        html = out.read_text(encoding="utf-8")
    for bloco in ("b00", "b01", "b02", "b03", "b04", "b05", "b06"):
        assert f'data-bloco="{bloco}"' in html
    assert "Quebra por Praca" in html and "Quebra por Midia" in html
    assert 'class="pfig"' in html and "data-fz-figs" in html


def test_html_report_flags_gate_failure_and_alert():
    with tempfile.TemporaryDirectory() as tmp:
        html = generate_html_report([_run_dir(tmp)], os.path.join(tmp, "dd.html")).read_text(encoding="utf-8")
    assert "fz-status-fail" in html      # Midia: proxy_ratio 1.4 fora dos gates
    assert "fz-status-warn" in html      # sp saturada, rj com slope de degrau


def test_html_report_without_optional_csvs_drops_those_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        pasta = _run_dir(tmp, hill=False, diag=False)
        html = generate_html_report([pasta], os.path.join(tmp, "dd.html")).read_text(encoding="utf-8")
    assert 'data-bloco="b05"' not in html and 'data-bloco="b06"' not in html
    assert 'data-bloco="b04"' in html


def test_html_report_several_vehicles_get_a_filter():
    with tempfile.TemporaryDirectory() as tmp:
        pastas = [_run_dir(tmp, "a"), _run_dir(tmp, "b")]
        html = generate_html_report(pastas, os.path.join(tmp, "dd.html")).read_text(encoding="utf-8")
    assert html.count('data-fz-filtro-painel="veiculo"') == 2


def test_html_report_missing_contributions_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            generate_html_report([tmp])
