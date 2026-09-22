import pandas as pd
import pytest
import sys, os
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from config import DeepDiveConfig, build_config, load_yaml
from extraction import UpgradeResult


def _fake_upgrade_for_config(cols):
    return UpgradeResult(
        model=None,
        contrib_df=pd.DataFrame(0.0, index=range(10), columns=cols),
        spend_df=pd.DataFrame(0.0, index=range(10), columns=cols),
        mmm_config={},
    )


def test_build_config_returns_dataclass(tmp_path):
    """Synthetic vehicle_specs.yaml + client YAML -- doesn't depend on any
    real client/vehicle in configs/ or data/vehicle_specs.yaml, so this test
    stays valid regardless of which clients happen to exist on a given
    branch (e.g. a production branch with a trimmed-down vehicle_specs.yaml)."""
    vehicle_spec = {
        "vehicle_slug": "fake",
        "default_metric": "investments",
        "models": {"default_template": "$metric:{metric}$category:brand:{brand}$category:{category}:{value}"},
        "breakdowns": {"Region": {"category": "region", "values": ["north", "south"]}},
    }
    specs_path = tmp_path / "vehicle_specs.yaml"
    specs_path.write_text(yaml.dump({"vehicles": {"fake_vehicle": vehicle_spec}}))

    media_var = "$metric:investments$vehicle:fake_vehicle$category:brand:acme"
    client_path = tmp_path / "client.yaml"
    client_path.write_text(yaml.dump({
        "brand": "acme",
        "vehicle": "fake_vehicle",
        "vehicle_specs_path": specs_path.name,
        "model_type": "stan",
        "media_var": "configured-media-var",
        "auxiliary_metric": "investments",
    }))

    ur = _fake_upgrade_for_config([media_var])
    cfg = build_config(ur, specs_path=str(client_path), media_var_override=media_var)
    assert cfg.media_var == media_var
    assert isinstance(cfg, DeepDiveConfig)
    assert cfg.dims == ["Region"]
    assert cfg.vars_per_dim["Region"] == [
        "$metric:investments$category:brand:acme$category:region:north",
        "$metric:investments$category:brand:acme$category:region:south",
    ]
    assert cfg.brand == "acme"
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


def _specs(tmp_path, *, vehicle_aux=None, client_aux=None):
    vehicle_spec = {
        "vehicle_slug": "fake",
        "default_metric": "investments",
        "models": {"default_template":
                   "$metric:{metric}$category:brand:{brand}$category:{category}:{value}"},
        "breakdowns": {"Region": {"category": "region", "values": ["north", "south"]}},
    }
    if vehicle_aux is not None:
        vehicle_spec["auxiliary_metric"] = vehicle_aux
    specs_path = tmp_path / "vehicle_specs.yaml"
    specs_path.write_text(yaml.dump({"vehicles": {"fake_vehicle": vehicle_spec}}))

    client = {
        "brand": "acme", "vehicle": "fake_vehicle",
        "vehicle_specs_path": specs_path.name,
        "model_type": "stan", "media_var": "configured-media-var",
    }
    if client_aux is not None:
        client["auxiliary_metric"] = client_aux
    client_path = tmp_path / "client.yaml"
    client_path.write_text(yaml.dump(client))
    return client_path


def test_auxiliary_metric_is_inherited_from_the_vehicle(tmp_path):
    """It describes what the vehicle measures, so it lives on the vehicle and
    every client of that vehicle gets the same one."""
    path = _specs(tmp_path, vehicle_aux="impressions")
    cfg = build_config(_fake_upgrade_for_config(["c"]), specs_path=str(path))

    assert cfg.auxiliary_metric == "impressions"


def test_client_auxiliary_metric_wins_and_says_so(tmp_path, capsys):
    path = _specs(tmp_path, vehicle_aux="impressions", client_aux="reach")
    cfg = build_config(_fake_upgrade_for_config(["c"]), specs_path=str(path))

    assert cfg.auxiliary_metric == "reach"
    assert "overridden" in capsys.readouterr().out


def test_missing_auxiliary_metric_on_both_sides_raises(tmp_path):
    path = _specs(tmp_path)

    with pytest.raises(ValueError, match="auxiliary_metric"):
        build_config(_fake_upgrade_for_config(["c"]), specs_path=str(path))


def test_every_vehicle_in_the_repo_declares_an_auxiliary_metric():
    """A vehicle without it only fails at build_config time, per client."""
    specs = load_yaml(os.path.join(os.path.dirname(__file__), "../data/vehicle_specs.yaml"))
    missing = [name for name, v in specs["vehicles"].items() if not v.get("auxiliary_metric")]

    assert not missing, f"sem auxiliary_metric: {missing}"
