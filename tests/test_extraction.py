import pandas as pd
import sys, os
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from extraction import UpgradeResult, load_upgrade


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


def test_load_upgrade_with_mocks():
    idx = pd.date_range("2023-01-02", periods=2, freq="W-MON")
    contrib = pd.DataFrame({"chan_a": [10.0, 20.0], "chan_b": [5.0, 5.0]}, index=idx)

    # fake pyfunc model that unwraps to a stan_model mock
    fake_stan_model = MagicMock()
    fake_pyfunc = MagicMock()
    fake_pyfunc.unwrap_python_model.return_value.model = fake_stan_model

    # fake Contribution class
    fake_contrib_instance = MagicMock()
    fake_contrib_instance.get_contribution.return_value = contrib
    FakeContribution = MagicMock(return_value=fake_contrib_instance)

    # fake MLflow client returning run params
    fake_run = MagicMock()
    fake_run.data.params = {"media_features": "chan_a,chan_b", "target": "kpi"}
    MockClient = MagicMock()
    MockClient.return_value.get_run.return_value = fake_run

    with patch("mlflow.pyfunc.load_model", return_value=fake_pyfunc), \
         patch("mlflow.tracking.MlflowClient", MockClient), \
         patch("mammoth.mmm.contribution.contribution.Contribution", FakeContribution):

        result = load_upgrade("fake-run-id", tracking_uri="http://fake")

    assert result.model is fake_stan_model
    assert list(result.contrib_df.columns) == ["chan_a", "chan_b"]
    assert result.spend_df.empty                    # populated by load_breakdown_spend()
    assert abs(float(result.y_hat.iloc[0]) - 15.0) < 1e-6   # 10 + 5
    assert abs(float(result.y_hat.iloc[1]) - 25.0) < 1e-6   # 20 + 5
