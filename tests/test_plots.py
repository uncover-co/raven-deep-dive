import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from plots import UNCOVER_DARK_TEMPLATE, plot_contributions, plot_weekly, plot_roas_index, plot_saturation_curves
from pipeline import DDResult
from config import DeepDiveConfig
import plotly.graph_objects as go


def _fake_ddresult():
    idx = pd.date_range("2023-01-02", periods=10, freq="W-MON")
    return DDResult(
        models={},
        contribs={"Praca": pd.DataFrame({"sp": np.ones(10), "rj": np.ones(10) * 0.5}, index=idx)},
        shares_model={"Praca": pd.Series({"sp": 0.67, "rj": 0.33})},
        shares_spend={"Praca": pd.Series({"sp": 0.60, "rj": 0.40})},
        proxy_ratios={"Praca": 0.98},
        csl_devs={"Praca": 0.04},
        eletro_contrib=pd.Series(np.ones(10) * 150, index=idx),
        config=DeepDiveConfig(
            dims=["Praca"],
            vars_per_dim={"Praca": ["sp", "rj"]},
            media_var="eletro",
        ),
    )


def test_template_exists():
    assert UNCOVER_DARK_TEMPLATE is not None


def test_plot_contributions_returns_figure():
    fig = plot_contributions(_fake_ddresult())
    assert isinstance(fig, go.Figure)


def test_plot_weekly_returns_figure():
    fig = plot_weekly(_fake_ddresult(), dim="Praca")
    assert isinstance(fig, go.Figure)


def test_plot_roas_index_returns_figure():
    fig = plot_roas_index(_fake_ddresult())
    assert isinstance(fig, go.Figure)


def test_plot_saturation_curves_raises_on_missing_model():
    import pytest
    # models dict is empty → should raise ValueError with dim name
    with pytest.raises(ValueError, match="Praca"):
        plot_saturation_curves(_fake_ddresult(), dim="Praca")
