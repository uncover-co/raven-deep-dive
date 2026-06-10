from __future__ import annotations
import os
import re
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
    share_prior_scale: float = 0.05
    proxy_ct_tolerance: float = 0.15
    num_steps: int = 30_000
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
    vehicle_spec: dict, brand: str, dims: list[str] | None
) -> dict[str, list[str]]:
    """Build {dimension_name: [slug, ...]} mapping from vehicle spec."""
    vehicle_slug = vehicle_spec.get("vehicle_slug", "eletromidia")
    metric = vehicle_spec.get("default_metric", "investments")
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
        category = bd["category"]
        template = _get_template(vehicle_spec, bd, model_type="stan")
        slugs = []
        for value in bd.get("values", []):
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


def _infer_media_var(columns, vehicle_spec: dict, brand: str) -> str:
    """Heuristically find the aggregate media variable in contrib_df columns.

    Searches brand-scoped patterns first, then vehicle_slug, then common OOH names.
    For non-OOH vehicles, set media_var explicitly in the client YAML.
    """
    vslug = vehicle_spec.get("vehicle_slug", "")
    patterns = [
        *(
            [
                f"{re.escape(vslug)}.*{re.escape(brand)}",
                f"{re.escape(brand)}.*{re.escape(vslug)}",
            ]
            if vslug else []
        ),
        f"eletro.*{re.escape(brand)}",
        f"ooh.*{re.escape(brand)}",
        f"{re.escape(brand)}.*eletro",
        f"{re.escape(brand)}.*ooh",
        *(([vslug] if vslug else [])),
        "eletromidia",
        "ooh_macro",
        "eletro_total",
    ]
    for pat in patterns:
        for col in columns:
            if re.search(pat, col, flags=re.IGNORECASE):
                return col

    raise ValueError(
        "Cannot infer media_var from contrib_df columns. "
        "Set 'media_var' (or legacy 'eletro_var') in the client YAML, "
        "or pass media_var_override=. "
        f"Available columns (first 10): {list(columns)[:10]}"
    )


def build_config(
    upgrade: Any,
    specs_path: str,
    media_var_override: str | None = None,
) -> DeepDiveConfig:
    """Build DeepDiveConfig from client YAML + UpgradeResult.

    Args:
        upgrade: UpgradeResult with contrib_df (used to infer media_var if not set).
        specs_path: path to the client YAML (e.g. deepdive/configs/bradesco_eletro.yaml).
        media_var_override: explicit aggregate channel column name; skips inference.
    """
    cfg = _load_yaml(specs_path)
    brand = cfg.get("brand", "")
    dims_override = cfg.get("dimensions", None)

    vehicle_specs_rel = cfg.get("vehicle_specs_path", "../data/vehicle_specs.yaml")
    base_dir = os.path.dirname(os.path.abspath(specs_path))
    vehicle_specs_path = os.path.normpath(os.path.join(base_dir, vehicle_specs_rel))

    vehicle_specs = _load_yaml(vehicle_specs_path)
    vehicle_key = cfg.get("vehicle", "eletromidia")
    vehicle_spec = vehicle_specs.get("vehicles", {}).get(vehicle_key, {})

    vars_per_dim = _build_stan_vars(vehicle_spec, brand, dims_override)
    dims = list(vars_per_dim.keys())

    if media_var_override:
        media_var = media_var_override
    else:
        media_var = (
            cfg.get("media_var")
            or _infer_media_var(upgrade.contrib_df.columns, vehicle_spec, brand)
        )

    return DeepDiveConfig(
        dims=dims,
        vars_per_dim=vars_per_dim,
        media_var=media_var,
        brand=brand,
        vehicle=vehicle_key,
        share_prior_scale=cfg.get("share_prior_scale", 0.05),
        proxy_ct_tolerance=cfg.get("proxy_ct_tolerance", 0.15),
        num_steps=cfg.get("num_steps", 30_000),
        vehicle_spec=vehicle_spec,
    )


# Backward-compat alias — notebooks still import build_eletro_config
build_eletro_config = build_config
