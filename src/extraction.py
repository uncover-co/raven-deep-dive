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
    spend_df: pd.DataFrame          # breakdown-level spend + aux metric (populated by load_breakdown_data)
    mmm_config: dict                # {media_features, control_features, target, ...}
    model_type: ModelType = "stan"  # "stan" | "meridian" | "raven"
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

    # export_data may carry 1 week input_data doesn't have -- a Meridian
    # forecast week, or a partial week from daily-logged contributions. Which
    # end it lands on is not fixed: raven run 0fad79f9 has it trailing, while
    # the old raven loader was written against a stray leading day. Drop it by
    # date; a positional trim shifts the whole series when it guesses wrong.
    inp = pd.read_parquet(input_path)
    if "timestamp" in inp.columns:
        inp = inp.sort_values("timestamp")
    n_inp, n_contrib = len(inp), len(contrib_df)
    if n_contrib == n_inp + 1:
        extra = None
        if "timestamp" in inp.columns:
            inp_weeks = pd.DatetimeIndex(
                pd.to_datetime(inp["timestamp"]).dt.to_period("W-MON").dt.start_time
            ).normalize().unique()
            extra = contrib_df.index.difference(inp_weeks)
        if extra is None or len(extra) != 1:
            found = [] if extra is None else [str(d.date()) for d in extra]
            raise ValueError(
                f"input_data has {n_inp} rows and contrib_df has {n_contrib}, but the "
                f"extra week could not be identified by date (got {found}). "
                "Trimming by position would shift the series."
            )
        contrib_df = contrib_df.drop(index=extra)
    elif n_contrib != n_inp:
        raise ValueError(
            f"input_data has {n_inp} rows but contrib_df has {n_contrib}. "
            "Positional alignment requires the same number of weeks (tolerance: +1)."
        )
    mmm_config = dict(client.get_run(run_id).data.params)

    return UpgradeResult(
        model=None,
        contrib_df=contrib_df,
        spend_df=pd.DataFrame(),
        mmm_config=mmm_config,
        model_type=model_type,
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

    Same contract as the Stan loader, and the same metric_type: the Deep Dive
    decomposes the model's own target, so the `Efficiency Scaler` that Raven
    also logs must NOT be applied here -- it converts a non-financial target
    into a financial one to produce ROI, which is a separate transformation
    the DS applies afterwards if they want it.

    Note: the artifact store backing some Raven runs may require AWS SSO
    (`aws sso login`) rather than the static keys used for Stan/Meridian.
    """
    return _load_from_parquets(
        run_id,
        tracking_uri=tracking_uri,
        cache_dir=cache_dir,
        contribution_metric_type="Contribution Unadstocked",
        model_type="raven",
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


def load_breakdown_data(
    workspace: str,
    all_vars: list[str],
    start_date: datetime,
    end_date: datetime,
    time_interval: str = "week",
    timezone: str = "America/Sao_Paulo",
    data_version: str | None = None,
) -> pd.DataFrame:
    """Load breakdown-level spend + auxiliary metric data for all Deep Dive variables.

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
