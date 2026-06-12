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
    y_actual: pd.Series | None = None  # observed KPI — required for y_adj strategy


def load_upgrade(run_id: str, tracking_uri: str | None = None) -> UpgradeResult:
    """Load a Stan upgrade run from MLflow.

    Artifact path is 'mmm' (mammoth convention) — unwraps pyfunc → base model.
    Contributions computed via mammoth Contribution class.
    spend_df is initially empty; call load_breakdown_spend() afterward.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Load the model: mlflow pyfunc wrapper → unwrap → base StanMMM
    pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/mmm")
    stan_model = pyfunc_model.unwrap_python_model().model

    # Build contributions via mammoth
    from mammoth.mmm.contribution.contribution import Contribution
    contrib_df = Contribution(stan_model).get_contribution(unadstocked=True)

    y_hat = contrib_df.sum(axis=1)

    # Actual observed KPI — try common mammoth/Stan attribute names
    y_actual: pd.Series | None = None
    for attr in ("y", "target", "data"):
        val = getattr(stan_model, attr, None)
        if val is None:
            continue
        if isinstance(val, pd.Series):
            y_actual = val.reindex(contrib_df.index)
            break
        if isinstance(val, dict) and "y" in val:
            y_actual = pd.Series(val["y"], index=contrib_df.index)
            break

    # mmm_config from run params (set by mammoth when logging)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    mmm_config = dict(run.data.params)

    return UpgradeResult(
        model=stan_model,
        contrib_df=contrib_df,
        spend_df=pd.DataFrame(),   # populated by load_breakdown_spend()
        mmm_config=mmm_config,
        y_hat=y_hat,
        model_type="stan",
        y_actual=y_actual,
    )


def _patch_meridian_model_context(inner: Any) -> None:
    """Reconstruct _model_context for models saved with an older Meridian API.

    Older Meridian serialized _input_data and _model_spec directly on the Meridian
    object. Newer installed Meridian wraps them in ModelContext. This patch bridges
    the gap so MeridianVisualizations.from_model() can work.
    """
    if hasattr(inner, "_model_context"):
        return
    if not (hasattr(inner, "_input_data") and hasattr(inner, "_model_spec")):
        raise AttributeError(
            "Meridian object missing '_model_context' and cannot reconstruct it: "
            "neither '_input_data' nor '_model_spec' found."
        )
    from meridian.model.context import ModelContext
    inner._model_context = ModelContext(
        input_data=inner._input_data,
        model_spec=inner._model_spec,
    )


def load_meridian_upgrade(run_id: str, tracking_uri: str | None = None) -> UpgradeResult:
    """Load a Meridian upgrade run from MLflow.

    Meridian uses channel names (not variable slugs) as contrib_df columns.
    contrib_df will contain one column per media channel (e.g. "eletromidia", "tv").
    The eletro_var in the client YAML must match a channel name here.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/mmm")
    meridian_mmm = pyfunc_model.unwrap_python_model().model  # MeridianMMM wrapper
    meridian_model = meridian_mmm.model                      # inner Meridian object

    _patch_meridian_model_context(meridian_model)

    from mammoth.mmm.reports.meridian.visualizations import MeridianVisualizations
    from mammoth.mmm.reports.meridian.report import (
        get_pivoted_contributions,
        create_incremental_outcome_dataframe,
    )

    visualizations = MeridianVisualizations.from_model(meridian_model)
    pivoted = get_pivoted_contributions(visualizations, use_kpi=True, selected_geo=None)
    contributions = create_incremental_outcome_dataframe(visualizations, pivoted)

    y_actual: pd.Series | None = None
    if "y_true" in contributions.columns:
        _yt = contributions["y_true"]
        if "time" in contributions.columns:
            _yt = _yt.set_axis(pd.to_datetime(contributions["time"]).dt.normalize())
        else:
            _yt = _yt.set_axis(pd.to_datetime(_yt.index).normalize())
        y_actual = _yt.rename(None)

    _drop = [c for c in ["All Channels", "y_true", "y_pred", "baseline", "residual"]
             if c in contributions.columns]
    contrib_df = contributions.drop(columns=_drop)
    if "time" in contrib_df.columns:
        contrib_df = contrib_df.set_index("time")
    contrib_df.index = pd.to_datetime(contrib_df.index).normalize()
    contrib_df.index.name = None

    y_hat = contrib_df.sum(axis=1)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    mmm_config = dict(run.data.params)

    return UpgradeResult(
        model=meridian_mmm,
        contrib_df=contrib_df,
        spend_df=pd.DataFrame(),
        mmm_config=mmm_config,
        y_hat=y_hat,
        model_type="meridian",
        y_actual=y_actual,
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
        return load_upgrade(run_id, tracking_uri=tracking_uri)
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
