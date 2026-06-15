from __future__ import annotations
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import mlflow
import pandas as pd


ModelType = Literal["stan", "meridian", "raven"]


@dataclass
class UpgradeResult:
    model: Any
    contrib_df: pd.DataFrame        # all channel contributions, index=timestamp
    spend_df: pd.DataFrame          # breakdown-level spend (populated by load_breakdown_spend)
    mmm_config: dict                # {media_features, control_features, target, ...}
    y_hat: pd.Series                # fitted KPI values (sum of contribs)
    model_type: ModelType = "stan"  # "stan" | "meridian" | "raven"
    y_actual: pd.Series | None = None  # observed KPI


def _load_from_parquets(
    run_id: str,
    tracking_uri: str | None = None,
    cache_dir: str | None = None,
    contribution_metric_type: str = "Contribution Unadstocked",
    model_type: ModelType = "stan",
) -> UpgradeResult:
    """Load UpgradeResult from export_data.parquet + input_data.parquet.

    cache_dir: if set, parquets are persisted under <cache_dir>/<run_id>/ and
               reused on subsequent calls.
    contribution_metric_type: metric_type row to use for contrib_df.
        Stan: 'Contribution Unadstocked'  Meridian: 'Contribution'
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    if cache_dir:
        dst = os.path.join(cache_dir, run_id)
        os.makedirs(dst, exist_ok=True)
        export_cached = os.path.join(dst, "export_data.parquet")
        input_cached = os.path.join(dst, "input_data.parquet")
        if os.path.exists(export_cached):
            print(f"[cache] {dst}")
            export_path, input_path = export_cached, input_cached
        else:
            import shutil, tempfile
            _tmp = tempfile.mkdtemp()
            export_path = client.download_artifacts(run_id, "export_data.parquet", _tmp)
            input_path = client.download_artifacts(run_id, "input_data.parquet", _tmp)
            shutil.copy(export_path, export_cached)
            shutil.copy(input_path, input_cached)
    else:
        import tempfile
        _tmp = tempfile.mkdtemp()
        export_path = client.download_artifacts(run_id, "export_data.parquet", _tmp)
        input_path = client.download_artifacts(run_id, "input_data.parquet", _tmp)

    export = pd.read_parquet(export_path)

    cu = export[export["metric_type"] == contribution_metric_type].copy()
    cu["timestamp"] = pd.to_datetime(cu["timestamp"]).dt.to_period("W-MON").dt.start_time
    contrib_df = (
        cu.groupby(["timestamp", "variable_name"])["value"]
        .sum()
        .unstack("variable_name")
        .fillna(0.0)
    )
    contrib_df.index = pd.DatetimeIndex(contrib_df.index).normalize()
    contrib_df.index.name = None

    # y_hat = sum of all contributions, including $metric:intercept
    y_hat = contrib_df.sum(axis=1).rename(None)

    # y_actual: positional alignment avoids timezone bucketing mismatches.
    # Meridian export_data may include 1 extra forecast week at the end — trim it.
    inp = pd.read_parquet(input_path)
    if "timestamp" in inp.columns:
        inp = inp.sort_values("timestamp")
    kpi_values = inp.iloc[:, -1].values
    n_inp, n_contrib = len(kpi_values), len(contrib_df)
    if n_contrib == n_inp + 1:
        contrib_df = contrib_df.iloc[:n_inp]
        y_hat = contrib_df.sum(axis=1).rename(None)
    elif n_contrib != n_inp:
        raise ValueError(
            f"input_data has {n_inp} rows but contrib_df has {n_contrib}. "
            "Positional alignment requires the same number of weeks (tolerance: +1)."
        )
    y_actual = pd.Series(kpi_values, index=contrib_df.index, name=None)

    mmm_config = dict(client.get_run(run_id).data.params)

    return UpgradeResult(
        model=None,
        contrib_df=contrib_df,
        spend_df=pd.DataFrame(),
        mmm_config=mmm_config,
        y_hat=y_hat,
        model_type=model_type,
        y_actual=y_actual,
    )


def load_upgrade_stan(
    run_id: str,
    tracking_uri: str | None = None,
    cache_dir: str | None = None,
) -> UpgradeResult:
    """Load a Stan upgrade run from export_data.parquet + input_data.parquet."""
    return _load_from_parquets(
        run_id,
        tracking_uri=tracking_uri,
        cache_dir=cache_dir,
        contribution_metric_type="Contribution Unadstocked",
        model_type="stan",
    )


def load_meridian_upgrade(
    run_id: str,
    tracking_uri: str | None = None,
    cache_dir: str | None = None,
) -> UpgradeResult:
    """Load a Meridian upgrade run from export_data.parquet + input_data.parquet.

    Meridian logs 'Contribution' (not 'Contribution Unadstocked') in export_data.
    cache_dir: if set, parquets are persisted under <cache_dir>/<run_id>/ and
               reused on subsequent calls.
    """
    return _load_from_parquets(
        run_id,
        tracking_uri=tracking_uri,
        cache_dir=cache_dir,
        contribution_metric_type="Contribution",
        model_type="meridian",
    )


def load_raven_upgrade(run_id: str, tracking_uri: str | None = None) -> UpgradeResult:
    """Load a Raven (mmmverse/prophetverse) upgrade run from MLflow.

    Not yet implemented — no standardized MLflow artifact path for Raven models.
    Once a Raven run_id is available, implement contrib extraction via predict_components.
    """
    raise NotImplementedError(
        "Raven upgrade extraction not yet implemented. "
        "Raven (mmmverse) models don't have a standardized MLflow artifact path yet. "
        "To implement: load the fitted Raven model, call predict_components(), "
        "and extract the target channel contribution series."
    )


def load_upgrade_auto(
    run_id: str,
    model_type: ModelType = "stan",
    tracking_uri: str | None = None,
) -> UpgradeResult:
    """Dispatch to the correct loader based on model_type."""
    if model_type == "stan":
        return load_upgrade_stan(run_id, tracking_uri=tracking_uri)
    if model_type == "meridian":
        return load_meridian_upgrade(run_id, tracking_uri=tracking_uri)
    if model_type == "raven":
        return load_raven_upgrade(run_id, tracking_uri=tracking_uri)
    raise ValueError(f"Unknown model_type='{model_type}'. Use 'stan', 'meridian', or 'raven'.")


def load_breakdown_spend(
    workspace: str,
    all_vars: list[str],
    start_date: datetime,
    end_date: datetime,
    time_interval: str = "week",
    timezone: str = "America/Sao_Paulo",
    output_path: str = "/tmp/dd_spend.parquet",
) -> pd.DataFrame:
    """Load breakdown-level spend data for all Deep Dive variables.

    Wraps preprocessing_dd from the mammoth BuildDefaultDataset pipeline.
    Returns DataFrame with timestamp index and one column per variable.
    """
    from uncover.deploy.pipelines.preprocessing import BuildDefaultDataset

    ds = BuildDefaultDataset(
        workspace=workspace,
        filters=all_vars,
        time_interval=time_interval,
        timezone=timezone,
        start_date=start_date,
        end_date=end_date,
    )
    ds.data = ds.data.fillna(0)
    ds.zero_fill_investments()
    zero_cols = [c for c in ds.data.columns if (ds.data[c] == 0).all()]
    if zero_cols:
        print(f"Dropping {len(zero_cols)} all-zero columns: {zero_cols}")
        ds.data = ds.data.drop(columns=zero_cols)
    ds.validate_output_dataset()
    ds.save_modelling_inputs(output_path=output_path)

    df = pd.read_parquet(output_path).fillna(0)
    if "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"])).drop(columns=["timestamp"])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df
