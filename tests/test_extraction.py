import os
import sys
import tempfile
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from extraction import UpgradeResult, load_upgrade_stan


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