import pandas as pd
import numpy as np
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from report import generate_report
from pipeline import DDResult
from config import DeepDiveConfig


def _fake_result():
    idx = pd.date_range("2023-01-02", periods=10, freq="W-MON")
    return DDResult(
        models={},
        contribs={"Praca": pd.DataFrame({"sp": np.ones(10), "rj": np.ones(10) * 0.5}, index=idx)},
        shares_model={"Praca": pd.Series({"sp": 0.67, "rj": 0.33})},
        shares_spend={"Praca": pd.Series({"sp": 0.60, "rj": 0.40})},
        proxy_ratios={"Praca": 0.98},
        csl_devs={"Praca": 0.04},
        r2={"Praca": 0.9},
        wape={"Praca": 0.1},
        media_dd_contrib=pd.Series(np.ones(10) * 150, index=idx),
        config=DeepDiveConfig(
            dims=["Praca"],
            vars_per_dim={"Praca": ["sp", "rj"]},
            media_var="eletro",
        ),
    )


def test_generate_report_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_report(_fake_result(), output_dir=tmpdir, client_name="Test")
        assert os.path.exists(paths["csv_contributions"])
        assert os.path.exists(paths["html_contributions"])
        assert os.path.exists(paths["html_roas"])


def test_generate_report_returns_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_report(_fake_result(), output_dir=tmpdir)
        assert isinstance(paths, dict)
        assert "csv_contributions" in paths
        assert "html_roas" in paths


def test_shares_csv_has_expected_columns():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_report(_fake_result(), output_dir=tmpdir, client_name="Test")
        df = pd.read_csv(paths["csv_contributions"])
        assert "dim" in df.columns
        assert "item" in df.columns
        assert "contrib_share" in df.columns
        assert "spend_share" in df.columns


def test_diagnostics_status_maps_no_gate_signal_and_no_primary_col():
    """Regression: reason_code values produced by diagnostics.py must have a
    status mapping here, or an excluded row (keep=False) silently reads as
    "kept" in the report (the fallback for any unmapped code)."""
    from report import _build_diagnostics_df
    from diagnostics import DiagnosisResult

    class _FakeResult:
        contribs = {"Praca": pd.DataFrame({"sp": [1.0]})}

    diag = DiagnosisResult(
        spend_report=pd.DataFrame([
            {"dim": "Praca", "slug": "sp", "reason": "no signal in impr", "reason_code": "no_gate_signal",
             "active_weeks": 0, "gate_total": 0.0, "pct_gate_dim": 0.0},
            {"dim": "Praca", "slug": "rj", "reason": "no investment (missing column)", "reason_code": "no_primary_col",
             "active_weeks": 10, "gate_total": 0.0, "pct_gate_dim": 0.0},
        ]),
        bucketed={},
        skipped_dims=[],
    )
    df = _build_diagnostics_df(_FakeResult(), diag)
    status_by_slug = dict(zip(df["slug"], df["status"]))
    assert status_by_slug["sp"] == "discarded_no_signal"
    assert status_by_slug["rj"] == "discarded_no_investment"
