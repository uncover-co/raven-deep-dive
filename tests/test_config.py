import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from config import DeepDiveConfig, build_config
from extraction import UpgradeResult


def _fake_upgrade_for_config(cols):
    return UpgradeResult(
        model=None,
        contrib_df=pd.DataFrame(0.0, index=range(10), columns=cols),
        spend_df=pd.DataFrame(0.0, index=range(10), columns=cols),
        mmm_config={},
        y_hat=pd.Series(0.0, index=range(10)),
    )


def test_build_config_returns_dataclass():
    ur = _fake_upgrade_for_config(["investments:eletromidia:transacoes-cc:state:sao-paulo"])
    cfg = build_config(
        ur,
        specs_path="deepdive/configs/bradesco_eletro.yaml",
        media_var_override="investments:eletromidia:transacoes-cc:state:sao-paulo",
    )
    assert isinstance(cfg, DeepDiveConfig)
    assert len(cfg.dims) > 0
    assert all(d in cfg.vars_per_dim for d in cfg.dims)
    assert cfg.brand == "transacoes-cc"
    assert cfg.share_prior_scale == 0.05


def test_deepdivedconfig_defaults():
    cfg = DeepDiveConfig(
        dims=["Praca"],
        vars_per_dim={"Praca": ["invest:sp"]},
        media_var="eletro",
    )
    assert cfg.share_prior_scale == 0.05
    assert cfg.num_steps == 30_000
