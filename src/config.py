from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import yaml


@dataclass
class DeepDiveConfig:
    dims: list[str]
    vars_per_dim: dict[str, list[str]]
    media_var: str
    brand: str = ""
    vehicle: str = "eletromidia"
    model_type: str = "stan"
    share_prior_scale: float = 0.05
    proxy_ct_tolerance: float = 0.15
    num_steps: int = 30_000
    min_spend_share: float = 0.02
    hhi_threshold: float = 0.85
    min_active_weeks: int = 2
    model_name: str = ""          # human-readable model identifier (e.g. "Transacoes CC PF - Nacional")
    vehicle_spec: dict = field(default_factory=dict)  # full spec from vehicle_specs.yaml


if "!class" not in yaml.SafeLoader.yaml_constructors:
    yaml.SafeLoader.add_constructor(
        "!class", lambda loader, node: loader.construct_scalar(node)
    )


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_template(vehicle_spec: dict, breakdown_spec: dict, model_type: str = "stan") -> str:
    """Select the right slug template for a breakdown, mirroring utils._get_template."""
    category = breakdown_spec["category"]

    # 1) Breakdown-level override has highest priority.
    breakdown_templates = breakdown_spec.get("templates", {})
    if model_type in breakdown_templates:
        return breakdown_templates[model_type]

    model_spec = vehicle_spec.get("models", {}).get(model_type, {})

    # 2) State-specific template when category == "state".
    if category == "state" and model_spec.get("state_template"):
        return model_spec["state_template"]

    # 3) Default model template.
    if model_spec.get("default_template"):
        return model_spec["default_template"]

    raise ValueError(
        f"No template defined for model_type='{model_type}' and category='{category}'."
    )


def _build_stan_vars(
    vehicle_spec: dict, brand: str, dims: list[str] | None, model_type: str = "stan"
) -> dict[str, list[str]]:
    """Build {dimension_name: [slug, ...]} mapping from vehicle spec."""
    vehicle_slug = vehicle_spec.get("vehicle_slug", "eletromidia")
    raw_metric = (
        vehicle_spec.get("metrics", {}).get(model_type)
        or vehicle_spec.get("default_metric", "investments")
    )
    # metrics entry can be str (single) or list (e.g. meridian uses investments + impressions)
    metrics = raw_metric if isinstance(raw_metric, list) else [raw_metric]

    all_breakdowns = vehicle_spec.get("breakdowns", {})
    # Default: model_dims from vehicle_spec (avoids Estado/Vertical/Tipo being modeled separately).
    # Explicit dims= or YAML dimensions: override this.
    default_dims = vehicle_spec.get("model_dims") or list(all_breakdowns.keys())
    selected = dims or default_dims

    result: dict[str, list[str]] = {}
    for bd_name in selected:
        if bd_name not in all_breakdowns:
            continue
        bd = all_breakdowns[bd_name]
        category = bd.get("category", "")
        if not category:
            raise ValueError(
                f"Breakdown '{bd_name}' missing required field 'category' in vehicle_specs."
            )
        values = bd.get("values", [])
        if not values:
            raise ValueError(
                f"Breakdown '{bd_name}' has no 'values' defined in vehicle_specs."
            )
        template = _get_template(vehicle_spec, bd, model_type=model_type)
        slugs = []
        for metric in metrics:
            for value in values:
                slug = template.format(
                    metric=metric,
                    vehicle=vehicle_slug,
                    brand=brand,
                    category=category,
                    value=value,
                )
                slugs.append(slug)
        if slugs:
            result[bd_name] = slugs
    return result


def build_config(
    upgrade: Any,
    specs_path: str,
    media_var_override: str | None = None,
) -> DeepDiveConfig:
    """Build DeepDiveConfig from client YAML + UpgradeResult.

    Args:
        upgrade: UpgradeResult with contrib_df (used to validate media_var).
        specs_path: path to the client YAML (e.g. deepdive/configs/bradesco_eletro.yaml).
        media_var_override: explicit aggregate channel column name; overrides YAML value.
    """
    cfg = _load_yaml(specs_path)
    brand = cfg.get("brand", "")
    dims_override = cfg.get("dimensions", None)

    vehicle_specs_rel = cfg.get("vehicle_specs_path", "../data/vehicle_specs.yaml")
    base_dir = os.path.dirname(os.path.abspath(specs_path))
    vehicle_specs_path = os.path.normpath(os.path.join(base_dir, vehicle_specs_rel))

    if not os.path.exists(vehicle_specs_path):
        raise FileNotFoundError(
            f"vehicle_specs not found: {vehicle_specs_path}\n"
            f"Check 'vehicle_specs_path' in {specs_path}."
        )
    vehicle_specs = _load_yaml(vehicle_specs_path)
    vehicle_key = cfg.get("vehicle", "eletromidia")
    available_vehicles = list(vehicle_specs.get("vehicles", {}).keys())
    if vehicle_key not in vehicle_specs.get("vehicles", {}):
        raise ValueError(
            f"Vehicle '{vehicle_key}' not found in {vehicle_specs_path}. "
            f"Available: {available_vehicles}"
        )
    vehicle_spec = vehicle_specs["vehicles"][vehicle_key]
    model_type = cfg.get("model_type", "stan")

    vars_per_dim = _build_stan_vars(vehicle_spec, brand, dims_override, model_type=model_type)
    dims = list(vars_per_dim.keys())

    media_var = media_var_override or cfg.get("media_var")
    if not media_var:
        raise ValueError(
            f"'media_var' not set in {specs_path}. "
            "Add 'media_var: <column_name>' matching an exact column in contrib_df. "
            f"Available columns (first 10): {list(upgrade.contrib_df.columns)[:10]}"
        )

    return DeepDiveConfig(
        dims=dims,
        vars_per_dim=vars_per_dim,
        media_var=media_var,
        brand=brand,
        vehicle=vehicle_key,
        model_type=model_type,
        model_name=cfg.get("model_name", ""),
        share_prior_scale=cfg.get("share_prior_scale", 0.05),
        proxy_ct_tolerance=cfg.get("proxy_ct_tolerance", 0.15),
        num_steps=cfg.get("num_steps", 30_000),
        min_spend_share=cfg.get("min_spend_share", 0.02),
        hhi_threshold=cfg.get("hhi_threshold", 0.85),
        min_active_weeks=cfg.get("min_active_weeks", 2),
        vehicle_spec=vehicle_spec,
    )