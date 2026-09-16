import os
import sys
import tempfile
from datetime import datetime
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from extraction import UpgradeResult, load_upgrade_stan, load_raven_upgrade


def test_upgrade_result_fields():
    ur = UpgradeResult(
        model=None,
        contrib_df=pd.DataFrame({"a": [1, 2]}),
        spend_df=pd.DataFrame({"x": [10, 20]}),
        mmm_config={"media_features": ["a"]},
        y_hat=pd.Series([100.0, 200.0]),
    )
    assert ur.contrib_df.shape == (2, 1)
    assert ur.spend_df.shape == (2, 1)
    assert ur.y_hat.sum() == 300.0
    assert ur.mmm_config["media_features"] == ["a"]
    assert ur.model is None


def test_load_upgrade_stan_with_mocks():
    idx = pd.date_range("2023-01-02", periods=2, freq="W-MON")

    # Build fake export_data.parquet (Contribution Unadstocked rows)
    export_rows = []
    for ts, (a_val, b_val) in zip(idx, [(10.0, 5.0), (20.0, 5.0)]):
        export_rows.append({"timestamp": ts, "variable_name": "chan_a", "value": a_val, "metric_type": "Contribution Unadstocked"})
        export_rows.append({"timestamp": ts, "variable_name": "chan_b", "value": b_val, "metric_type": "Contribution Unadstocked"})
    export_df = pd.DataFrame(export_rows)

    # Build fake input_data.parquet (last column = KPI)
    input_df = pd.DataFrame({"timestamp": idx, "other_col": [1.0, 2.0], "kpi": [100.0, 200.0]})

    with tempfile.TemporaryDirectory() as tmp:
        export_path = os.path.join(tmp, "export_data.parquet")
        input_path = os.path.join(tmp, "input_data.parquet")
        export_df.to_parquet(export_path, index=False)
        input_df.to_parquet(input_path, index=False)

        fake_run = MagicMock()
        fake_run.data.params = {"media_features": "chan_a,chan_b", "target": "kpi"}
        mock_client = MagicMock()
        mock_client.return_value.download_artifacts.side_effect = [export_path, input_path]
        mock_client.return_value.get_run.return_value = fake_run

        with patch("mlflow.tracking.MlflowClient", mock_client):
            result = load_upgrade_stan("fake-run-id", tracking_uri="http://fake")

    assert list(result.contrib_df.columns) == ["chan_a", "chan_b"]
    assert result.spend_df.empty
    assert abs(float(result.y_hat.iloc[0]) - 15.0) < 1e-6   # 10 + 5
    assert abs(float(result.y_hat.iloc[1]) - 25.0) < 1e-6   # 20 + 5
    assert result.y_actual is not None
    assert abs(float(result.y_actual.iloc[0]) - 100.0) < 1e-6
    assert abs(float(result.y_actual.iloc[1]) - 200.0) < 1e-6


def test_load_raven_upgrade_with_mocks():
    """Raven export_data.parquet/input_data.parquet follow the same convention
    as Stan/Meridian, but need 2 adjustments (see
    docs/deepdive-raven-upgrade-extraction-findings.md):
      - y_actual resolved by NAME (via the 'Target Prediction' metric_type row),
        not positionally (Raven's input_data.parquet doesn't put the KPI last).
      - Contribution Unadstocked needs the weekly 'Efficiency Scaler' applied
        to match the model's actual fitted values.
    """
    idx = pd.date_range("2023-01-02", periods=2, freq="W-MON")

    export_rows = []
    for ts, (a_val, b_val, eff) in zip(idx, [(10.0, 5.0, 2.0), (20.0, 5.0, 1.5)]):
        export_rows.append({"timestamp": ts, "variable_name": "chan_a", "value": a_val, "metric_type": "Contribution Unadstocked"})
        export_rows.append({"timestamp": ts, "variable_name": "chan_b", "value": b_val, "metric_type": "Contribution Unadstocked"})
        export_rows.append({"timestamp": ts, "variable_name": "efficiency_scaler", "value": eff, "metric_type": "Efficiency Scaler"})
        export_rows.append({"timestamp": ts, "variable_name": "kpi", "value": 0.0, "metric_type": "Target Prediction"})
    export_df = pd.DataFrame(export_rows)

    # KPI is NOT the last column — mirrors real Raven input_data.parquet layout.
    input_df = pd.DataFrame({"timestamp": idx, "kpi": [100.0, 200.0], "other_col": [1.0, 2.0]})

    with tempfile.TemporaryDirectory() as tmp:
        export_path = os.path.join(tmp, "export_data.parquet")
        input_path = os.path.join(tmp, "input_data.parquet")
        export_df.to_parquet(export_path, index=False)
        input_df.to_parquet(input_path, index=False)

        fake_run = MagicMock()
        fake_run.data.params = {"media_features": "chan_a,chan_b", "target": "kpi"}
        mock_client = MagicMock()
        mock_client.return_value.download_artifacts.side_effect = [export_path, input_path]
        mock_client.return_value.get_run.return_value = fake_run

        with patch("mlflow.tracking.MlflowClient", mock_client):
            result = load_raven_upgrade("fake-run-id", tracking_uri="http://fake")

    assert result.model_type == "raven"
    assert list(result.contrib_df.columns) == ["chan_a", "chan_b"]
    # y_hat = (chan_a + chan_b) * efficiency_scaler, per week
    assert abs(float(result.y_hat.iloc[0]) - 15.0 * 2.0) < 1e-6   # (10+5)*2.0
    assert abs(float(result.y_hat.iloc[1]) - 25.0 * 1.5) < 1e-6   # (20+5)*1.5
    assert result.y_actual is not None
    assert abs(float(result.y_actual.iloc[0]) - 100.0) < 1e-6
    assert abs(float(result.y_actual.iloc[1]) - 200.0) < 1e-6


def test_load_breakdown_spend_drops_all_zero_columns_and_pins_data_version():
    """Also covers a real gap in ducks' zero_fill="media": it matches metric
    names by substring ("$metric:investments"/"$metric:impressions"), but our
    actual slugs (e.g. "$metric:w:investments---tiktok-mmm$...") never match
    that, so we don't rely on it -- fillna(0) is applied unconditionally
    here, matching the pre-ducks behavior regardless of naming."""
    from extraction import load_breakdown_spend
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=3, freq="W-MON")
    fake_df = pd.DataFrame({
        "$metric:w:investments---tiktok-mmm$category:a": [10.0, np.nan, 30.0],
        "$metric:w:investments---tiktok-mmm$category:b": [0.0, 0.0, 0.0],
    }, index=idx)

    mock_ws = MagicMock()
    mock_ws.build_modelling_dataset.return_value = fake_df

    with patch("ducks.workspace", return_value=mock_ws) as mock_workspace:
        result = load_breakdown_spend(
            "some-workspace",
            ["$metric:w:investments---tiktok-mmm$category:a", "$metric:w:investments---tiktok-mmm$category:b"],
            datetime(2024, 1, 1), datetime(2024, 1, 21),
            data_version="2026-01-01_000000",
        )

    mock_workspace.assert_called_once_with("some-workspace")
    kwargs = mock_ws.build_modelling_dataset.call_args.kwargs
    assert "zero_fill" not in kwargs
    assert kwargs["data_version"] == "2026-01-01_000000"
    assert list(result.columns) == ["$metric:w:investments---tiktok-mmm$category:a"]
    assert not result.isna().any().any()
