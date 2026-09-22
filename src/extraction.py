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
    input_df: pd.DataFrame | None = None  # raw input_data.parquet (main model's own features)


def _download_export_input_parquets(
    run_id: str,
    tracking_uri: str | None,
    cache_dir: str | None,
) -> tuple[str, str, "mlflow.tracking.MlflowClient"]:
    """Download (or reuse cached) export_data.parquet + input_data.parquet for a run.

    cache_dir: if set, parquets are persisted under <cache_dir>/<run_id>/ and
               reused on subsequent calls. Both files must be present in the
               cache to count as a hit -- a partial cache (e.g. an interrupted
               previous download) re-downloads rather than failing later on a
               missing input_data.parquet.

    Returns (export_path, input_path, client) -- client is also needed by
    callers for client.get_run(run_id).data.params.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    if cache_dir:
        dst = os.path.join(cache_dir, run_id)
        os.makedirs(dst, exist_ok=True)
        export_cached = os.path.join(dst, "export_data.parquet")
        input_cached = os.path.join(dst, "input_data.parquet")
        if os.path.exists(export_cached) and os.path.exists(input_cached):
            print(f"[cache] {dst}")
            return export_cached, input_cached, client
        import shutil, tempfile
        _tmp = tempfile.mkdtemp()
        export_path = client.download_artifacts(run_id, "export_data.parquet", _tmp)
        input_path = client.download_artifacts(run_id, "input_data.parquet", _tmp)
        shutil.copy(export_path, export_cached)
        shutil.copy(input_path, input_cached)
        return export_cached, input_cached, client

    import tempfile
    _tmp = tempfile.mkdtemp()
    export_path = client.download_artifacts(run_id, "export_data.parquet", _tmp)
    input_path = client.download_artifacts(run_id, "input_data.parquet", _tmp)
    return export_path, input_path, client


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
    export_path, input_path, client = _download_export_input_parquets(
        run_id, tracking_uri, cache_dir
    )

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
        input_df=inp,
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


def load_raven_upgrade(
    run_id: str,
    tracking_uri: str | None = None,
    cache_dir: str | None = None,
) -> UpgradeResult:
    """Load a Raven (mmmverse/prophetverse) upgrade run from MLflow.

    Note: the artifact store backing some Raven runs may require AWS SSO
    (`aws sso login`) rather than the static keys used for Stan/Meridian.
    """
    export_path, input_path, client = _download_export_input_parquets(
        run_id, tracking_uri, cache_dir
    )

    export = pd.read_parquet(export_path)
    inp = pd.read_parquet(input_path)
    if "timestamp" in inp.columns:
        inp = inp.sort_values("timestamp")

    def _weekly(metric_type: str, agg: str) -> pd.DataFrame:
        rows = export[export["metric_type"] == metric_type].copy()
        rows["timestamp"] = pd.to_datetime(rows["timestamp"]).dt.to_period("W-MON").dt.start_time
        df = rows.groupby(["timestamp", "variable_name"])["value"].agg(agg).unstack("variable_name")
        df.index = pd.DatetimeIndex(df.index).normalize()
        df.index.name = None
        return df

    contrib_raw = _weekly("Contribution Unadstocked", "sum").fillna(0.0)
    eff_scaler = _weekly("Efficiency Scaler", "mean")["efficiency_scaler"]

    # `Contribution Unadstocked` is logged daily and can include a stray
    # leading day that buckets into an extra partial week not present in
    # `Efficiency Scaler` (logged already weekly) — e.g. one December day
    # rolling into a "W-MON" period bucket before the real data starts.
    # Align the two to each other (both went through the same `_weekly`
    # bucketing, so their index labels are mutually consistent even though
    # — like `_load_from_parquets` — that bucketing lands 1 day off from
    # input_data.parquet's own timestamps; don't compare labels against
    # `inp` directly).
    extra_weeks = contrib_raw.index.difference(eff_scaler.index)
    if len(extra_weeks) > 0:
        contrib_raw = contrib_raw.drop(index=extra_weeks)
    missing_weeks = eff_scaler.index.difference(contrib_raw.index)
    if len(missing_weeks) > 0:
        raise ValueError(
            f"run {run_id}: 'Efficiency Scaler' has weeks {list(missing_weeks)} "
            "with no matching 'Contribution Unadstocked' data."
        )

    contrib_df = contrib_raw.mul(eff_scaler.reindex(contrib_raw.index), axis=0)
    y_hat = contrib_df.sum(axis=1).rename(None)

    n_inp, n_contrib = len(inp), len(contrib_df)
    if n_contrib == n_inp + 1:
        contrib_df = contrib_df.iloc[1:]
        y_hat = contrib_df.sum(axis=1).rename(None)
    elif n_contrib != n_inp:
        raise ValueError(
            f"input_data has {n_inp} rows but contrib_df has {n_contrib} after aligning "
            "to Efficiency Scaler. Positional alignment requires the same number of "
            "weeks (tolerance: +1)."
        )

    target_rows = export[export["metric_type"] == "Target Prediction"]
    if target_rows.empty:
        raise ValueError(
            f"No 'Target Prediction' rows in export_data.parquet for run {run_id} — "
            "cannot resolve the KPI column name in input_data.parquet."
        )
    target_var = target_rows["variable_name"].iloc[0]
    if target_var not in inp.columns:
        raise ValueError(
            f"Target variable '{target_var}' (from 'Target Prediction' in export_data.parquet) "
            f"not found in input_data.parquet columns for run {run_id}."
        )
    y_actual = pd.Series(inp[target_var].values, index=contrib_df.index, name=None)

    mmm_config = dict(client.get_run(run_id).data.params)

    return UpgradeResult(
        model=None,
        contrib_df=contrib_df,
        spend_df=pd.DataFrame(),
        mmm_config=mmm_config,
        y_hat=y_hat,
        model_type="raven",
        y_actual=y_actual,
        input_df=inp,
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
    data_version: str | None = None,
) -> pd.DataFrame:
    """Load breakdown-level spend data for all Deep Dive variables.

    Wraps ducks' build_modelling_dataset (the maintained replacement for the
    legacy mammoth BuildDefaultDataset). Returns DataFrame with timestamp
    index and one column per variable that has real data -- an all-zero
    column (no signal at all) is dropped, same as the pre-ducks behavior.

    data_version pins the read to a completed workspace snapshot (see
    ducks' data-version docs); omit for the latest live data.
    """
    import ducks

    ws = ducks.workspace(workspace)
    df = ws.build_modelling_dataset(
        all_vars,
        start_date=start_date,
        end_date=end_date,
        time_interval=time_interval,
        timezone=timezone,
        # zero_fill="media" matches metric names by substring
        # ("$metric:investments"/"$metric:impressions"); our real slugs (e.g.
        # "$metric:w:investments---tiktok-mmm$...") don't match it, so fill
        # every column instead -- matches the pre-ducks behavior regardless
        # of naming.
        zero_fill=True,
        data_version=data_version,
    )
    zero_cols = [c for c in df.columns if (df[c] == 0).all()]
    if zero_cols:
        print(f"Dropping {len(zero_cols)} all-zero columns: {zero_cols}")
        df = df.drop(columns=zero_cols)
    df.index = df.index.normalize()
    return df
