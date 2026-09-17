import pandas as pd
import numpy as np
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from report import generate_report
from pipeline import DDResult
from config import DeepDiveConfig


def _fake_result():
    idx = pd.date_range("2023-01-02", periods=10, freq="W-MON")
    inv = pd.DataFrame({"sp": np.ones(10) * 10, "rj": np.ones(10) * 5}, index=idx)
    aux = pd.DataFrame({"sp": np.ones(10) * 100, "rj": np.ones(10) * 50}, index=idx)
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
        features_raw={"Praca": inv},
        auxiliary_metric_raw={"Praca": aux},
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


def test_model_inputs_csv_has_investment_and_auxiliary_metric_per_variable():
    """model_inputs.csv reflects exactly what fed each dim's Raven fit:
    investment + auxiliary metric, same week, same variable (already the
    post-diagnostic variable set, e.g. __others__ instead of its members)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_report(_fake_result(), output_dir=tmpdir, client_name="Test")
        df = pd.read_csv(paths["csv_model_inputs"])
        assert set(df.columns) == {"dim", "variable", "date", "investment", "auxiliary_metric"}
        assert set(df["variable"]) == {"sp", "rj"}
        sp_row = df[(df["variable"] == "sp") & (df["date"] == "2023-01-02")].iloc[0]
        assert sp_row["investment"] == 10.0
        assert sp_row["auxiliary_metric"] == 100.0


def _empty_spend_report() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "dim", "slug", "reason", "reason_code", "active_weeks", "gate_total", "pct_gate_dim",
    ])


def test_bucketed_detail_csv_present_only_when_diag_has_bucketing():
    from diagnostics import DiagnosisResult

    idx = pd.date_range("2023-01-02", periods=10, freq="W-MON")
    bucketed_raw_df = pd.DataFrame({
        "date": idx, "variable": ["rec"] * 10,
        "investment": np.ones(10), "auxiliary_metric": np.ones(10) * 2,
    })
    diag = DiagnosisResult(
        spend_report=_empty_spend_report(), bucketed={"Praca": ["rec"]}, skipped_dims=[],
        bucketed_raw={"Praca": bucketed_raw_df},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_report(_fake_result(), diag=diag, output_dir=tmpdir, client_name="Test")
        assert "csv_model_inputs_bucketed_detail" in paths
        df = pd.read_csv(paths["csv_model_inputs_bucketed_detail"])
        assert set(df.columns) == {"dim", "date", "variable", "investment", "auxiliary_metric"}
        assert df["dim"].unique().tolist() == ["Praca"]

    # No bucketing at all -> file not generated.
    diag_no_bucket = DiagnosisResult(spend_report=_empty_spend_report(), bucketed={}, skipped_dims=[])
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_report(_fake_result(), diag=diag_no_bucket, output_dir=tmpdir, client_name="Test")
        assert "csv_model_inputs_bucketed_detail" not in paths


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
