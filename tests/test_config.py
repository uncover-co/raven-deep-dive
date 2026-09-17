import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from config import DeepDiveConfig, build_config, load_yaml
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
    specs_path = os.path.join(os.path.dirname(__file__), "../configs/bradesco_eletro.yaml")
    ur = _fake_upgrade_for_config(["investments:eletromidia:transacoes-cc:state:sao-paulo"])
    cfg = build_config(
        ur,
        specs_path=specs_path,
        media_var_override="investments:eletromidia:transacoes-cc:state:sao-paulo",
    )
    assert isinstance(cfg, DeepDiveConfig)
    assert len(cfg.dims) > 0
    assert all(d in cfg.vars_per_dim for d in cfg.dims)
    assert cfg.brand == "bradesco"
    assert cfg.share_prior_scale == 0.05


def test_deepdivedconfig_defaults():
    cfg = DeepDiveConfig(
        dims=["Praca"],
        vars_per_dim={"Praca": ["invest:sp"]},
        media_var="eletro",
    )
    assert cfg.share_prior_scale == 0.05
    assert cfg.num_steps == 30_000


def test_load_yaml_resolves_instance_params_tags_to_real_objects(tmp_path):
    """load_yaml() must deserialize !instance/!params into a real Python
    object, not a string or a plain dict -- this is what makes
    upper_funnel_adstock_effect_per_dim work at all (build_config -> Raven).
    A regression in the from_yaml wiring would otherwise leave the suite
    green while client YAMLs with this tag fail at startup."""
    from prophetverse.effects import WeibullAdstockEffect

    yaml_path = tmp_path / "adstock.yaml"
    yaml_path.write_text(
        "upper_funnel_adstock_effect_per_dim:\n"
        "  ProductDim:\n"
        "    'slug_a':\n"
        "      '!instance': prophetverse.effects.adstock.WeibullAdstockEffect\n"
        "      '!params':\n"
        "        max_lag: 7\n"
    )
    parsed = load_yaml(str(yaml_path))
    effect = parsed["upper_funnel_adstock_effect_per_dim"]["ProductDim"]["slug_a"]
    assert isinstance(effect, WeibullAdstockEffect)
    assert effect.max_lag == 7
