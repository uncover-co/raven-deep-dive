import os
import sys
import tempfile
from datetime import datetime
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from extraction import UpgradeResult, load_upgrade_stan, load_raven_upgrade, _download_export_input_parquets


def test_upgrade_result_fields():
    ur = UpgradeResult(
        model=None,
        contrib_df=pd.DataFrame({"a": [1, 2]}),
        spend_df=pd.DataFrame({"x": [10, 20]}),
        mmm_config={"media_features": ["a"]},
    )
    assert ur.contrib_df.shape == (2, 1)
    assert ur.spend_df.shape == (2, 1)
    assert ur.contrib_df.sum().sum() == 3
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
    assert result.input_df is not None      # check_spend_coverage depends on it
    weekly = result.contrib_df.sum(axis=1)
    assert abs(float(weekly.iloc[0]) - 15.0) < 1e-6   # 10 + 5
    assert abs(float(weekly.iloc[1]) - 25.0) < 1e-6   # 20 + 5


def test_load_raven_upgrade_reads_unadstocked_and_ignores_the_scaler():
    """Raven logs an `Efficiency Scaler` alongside the contributions, but that
    converts a non-financial target into a financial one to produce ROI. The
    Deep Dive decomposes the model's own target, so the scaler must not be
    applied -- the DS does that transformation afterwards if they want it.
    """
    idx = pd.date_range("2023-01-02", periods=2, freq="W-MON")

    export_rows = []
    for ts, (a_val, b_val, eff) in zip(idx, [(10.0, 5.0, 2.0), (20.0, 5.0, 1.5)]):
        export_rows.append({"timestamp": ts, "variable_name": "chan_a", "value": a_val, "metric_type": "Contribution Unadstocked"})
        export_rows.append({"timestamp": ts, "variable_name": "chan_b", "value": b_val, "metric_type": "Contribution Unadstocked"})
        export_rows.append({"timestamp": ts, "variable_name": "efficiency_scaler", "value": eff, "metric_type": "Efficiency Scaler"})
    export_df = pd.DataFrame(export_rows)
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
    assert result.input_df is not None      # check_spend_coverage depends on it
    assert list(result.contrib_df.columns) == ["chan_a", "chan_b"]
    # raw Unadstocked, NOT multiplied by the 2.0 / 1.5 scaler
    weekly = result.contrib_df.sum(axis=1)
    assert abs(float(weekly.iloc[0]) - 15.0) < 1e-6
    assert abs(float(weekly.iloc[1]) - 25.0) < 1e-6


def test_load_breakdown_spend_drops_all_zero_columns_and_pins_data_version():
    """zero_fill="media" (ducks) matches metric names by substring
    ("$metric:investments"/"$metric:impressions"); our actual slugs (e.g.
    "$metric:w:investments---tiktok-mmm$...") never match that, so we pass
    zero_fill=True instead -- fills every column regardless of naming,
    matching the pre-ducks behavior."""
    from extraction import load_breakdown_spend

    idx = pd.date_range("2024-01-01", periods=3, freq="W-MON")
    fake_df = pd.DataFrame({
        "$metric:w:investments---tiktok-mmm$category:a": [10.0, 20.0, 30.0],
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
    assert kwargs["zero_fill"] is True
    assert kwargs["data_version"] == "2026-01-01_000000"
    assert list(result.columns) == ["$metric:w:investments---tiktok-mmm$category:a"]


def test_download_export_input_parquets_full_cache_hit_skips_download():
    """Both files already cached -> reused as-is, no download_artifacts call."""
    with tempfile.TemporaryDirectory() as cache_dir:
        run_dir = os.path.join(cache_dir, "fake-run-id")
        os.makedirs(run_dir)
        export_cached = os.path.join(run_dir, "export_data.parquet")
        input_cached = os.path.join(run_dir, "input_data.parquet")
        open(export_cached, "w").close()
        open(input_cached, "w").close()

        mock_client = MagicMock()
        with patch("mlflow.tracking.MlflowClient", mock_client):
            export_path, input_path, _ = _download_export_input_parquets(
                "fake-run-id", tracking_uri=None, cache_dir=cache_dir
            )

        assert export_path == export_cached
        assert input_path == input_cached
        mock_client.return_value.download_artifacts.assert_not_called()


def test_download_export_input_parquets_partial_cache_redownloads_both():
    """Only export_data.parquet cached (e.g. an interrupted prior download) ->
    must NOT be treated as a cache hit; both files are re-downloaded."""
    with tempfile.TemporaryDirectory() as cache_dir, tempfile.TemporaryDirectory() as dl_dir:
        run_dir = os.path.join(cache_dir, "fake-run-id")
        os.makedirs(run_dir)
        export_cached = os.path.join(run_dir, "export_data.parquet")
        open(export_cached, "w").close()
        # input_data.parquet deliberately missing.

        dl_export = os.path.join(dl_dir, "export_data.parquet")
        dl_input = os.path.join(dl_dir, "input_data.parquet")
        open(dl_export, "w").close()
        open(dl_input, "w").close()

        mock_client = MagicMock()
        mock_client.return_value.download_artifacts.side_effect = [dl_export, dl_input]
        with patch("mlflow.tracking.MlflowClient", mock_client):
            export_path, input_path, _ = _download_export_input_parquets(
                "fake-run-id", tracking_uri=None, cache_dir=cache_dir
            )

        assert mock_client.return_value.download_artifacts.call_count == 2
        # Re-downloaded files get copied back into the cache dir.
        assert export_path == os.path.join(run_dir, "export_data.parquet")
        assert input_path == os.path.join(run_dir, "input_data.parquet")
        assert os.path.exists(input_path)


def _run_loader(export_df, input_df, loader):
    with tempfile.TemporaryDirectory() as tmp:
        export_path = os.path.join(tmp, "export_data.parquet")
        input_path = os.path.join(tmp, "input_data.parquet")
        export_df.to_parquet(export_path, index=False)
        input_df.to_parquet(input_path, index=False)

        fake_run = MagicMock()
        fake_run.data.params = {"media_features": "chan_a", "target": "kpi"}
        mock_client = MagicMock()
        mock_client.return_value.download_artifacts.side_effect = [export_path, input_path]
        mock_client.return_value.get_run.return_value = fake_run

        with patch("mlflow.tracking.MlflowClient", mock_client):
            return loader("fake-run-id", tracking_uri="http://fake")


def _export(weeks, values):
    return pd.DataFrame([
        {"timestamp": ts, "variable_name": "chan_a", "value": v,
         "metric_type": "Contribution Unadstocked"}
        for ts, v in zip(weeks, values)
    ])


def test_extra_leading_week_is_dropped_not_the_last_real_one():
    """Raven's stray week leads. A positional trim kept it and dropped the last
    real week, shifting contribution against spend by a whole week."""
    from extraction import load_raven_upgrade

    weeks = pd.date_range("2022-12-26", periods=4, freq="W-MON")
    result = _run_loader(
        _export(weeks, [999.0, 10.0, 20.0, 30.0]),
        pd.DataFrame({"timestamp": weeks[1:], "kpi": [1.0, 2.0, 3.0]}),
        load_raven_upgrade,
    )

    assert list(result.contrib_df["chan_a"]) == [10.0, 20.0, 30.0]
    kept = pd.DatetimeIndex(weeks[1:]).to_period("W-MON").start_time.normalize()
    assert list(result.contrib_df.index) == list(kept)


def test_extra_trailing_week_is_still_dropped():
    from extraction import load_meridian_upgrade

    weeks = pd.date_range("2023-01-02", periods=4, freq="W-MON")
    export = pd.DataFrame([
        {"timestamp": ts, "variable_name": "chan_a", "value": v,
         "metric_type": "Contribution"}
        for ts, v in zip(weeks, [10.0, 20.0, 30.0, 999.0])
    ])
    result = _run_loader(
        export,
        pd.DataFrame({"timestamp": weeks[:3], "kpi": [1.0, 2.0, 3.0]}),
        load_meridian_upgrade,
    )

    assert list(result.contrib_df["chan_a"]) == [10.0, 20.0, 30.0]


def test_ambiguous_extra_week_raises_instead_of_guessing():
    from extraction import load_raven_upgrade

    weeks = pd.date_range("2023-01-02", periods=4, freq="W-MON")
    off = pd.Timestamp("2024-06-03")
    with pytest.raises(ValueError, match="could not be identified by date"):
        _run_loader(
            _export(weeks, [10.0, 20.0, 30.0, 40.0]),
            pd.DataFrame({"timestamp": [weeks[0], weeks[1], off],
                          "kpi": [1.0, 2.0, 3.0]}),
            load_raven_upgrade,
        )
