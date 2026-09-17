from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from mmmverse.spec.yaml import from_yaml


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
    min_active_weeks: int = 2          # piso absoluto (séries curtas); ver min_active_weeks_frac
    min_active_weeks_frac: float = 0.05  # piso relativo: max(min_active_weeks, frac * n_weeks)
    model_name: str = ""          # human-readable model identifier (e.g. "Transacoes CC PF - Nacional")
    share_likelihood_metric: str = ""  # metric slug driving the Hill-curve regressor + diagnostics gate (defaults to investments)
    auxiliary_metric: str = ""    # metric slug used ONLY as CSL prior target + extra diagnostics guardrail (e.g. impressions) — never drives the regressor
    vehicle_spec: dict = field(default_factory=dict)  # full spec from vehicle_specs.yaml
    # {dim_name: [slug, ...]} fit WITHOUT adstock; rest of the dim keeps adstock.
    # Vehicle-agnostic: pipeline only sees slugs, no funnel/branding concept baked in.
    lower_funnel_vars_per_dim: dict[str, list[str]] = field(default_factory=dict)
    # {dim_name: BaseEffect | {slug: BaseEffect}} custom adstock for upper-funnel
    # vars in that dim (single effect for all, or one per slug -- mmmverse
    # requires every upper-funnel slug present when it's a dict). Missing dim
    # entry = library default for that whole dim.
    upper_funnel_adstock_effect_per_dim: dict[str, Any] = field(default_factory=dict)


def load_yaml(path: str) -> dict:
    """Load a YAML file. Uses mmmverse's spec loader, so !instance/!params
    mappings deserialize into real Python objects (e.g. a custom adstock
    effect) -- same mechanism production model_specs use."""
    with open(path, "r", encoding="utf-8") as f:
        return from_yaml(f.read()) or {}


def _get_template(vehicle_spec: dict, breakdown_spec: dict) -> str:
    """Select the slug template for a breakdown (one per vehicle, not split by model_type)."""
    category = breakdown_spec["category"]

    # 1) Breakdown-level override has highest priority.
    if breakdown_spec.get("template"):
        return breakdown_spec["template"]

    model_spec = vehicle_spec.get("models", {})

    # 2) State-specific template when category == "state".
    if category == "state" and model_spec.get("state_template"):
        return model_spec["state_template"]

    # 3) Default model template.
    if model_spec.get("default_template"):
        return model_spec["default_template"]

    raise ValueError(f"No template defined for category='{category}'.")


def _resolve_metrics(vehicle_spec: dict, auxiliary_metric: str) -> list[str]:
    """Metrics fetched for every breakdown slug: the vehicle's primary
    (investment) metric, plus the client's auxiliary_metric -- independent
    of which model anchors it (stan/meridian/raven). auxiliary_metric is
    mandatory (build_config raises if unset); fetches only the primary
    metric when it's set to the same value as the primary (no real
    exposure metric, investment used as its own proxy)."""
    primary = vehicle_spec.get("default_metric", "investments")
    if auxiliary_metric and auxiliary_metric != primary:
        return [primary, auxiliary_metric]
    return [primary]


def resolve_share_likelihood_metric(metrics: list[str], override: str | None) -> str:
    """Pick which fetched metric feeds ContributionShareLikelihood.

    Explicit `override` (client cfg's `share_likelihood_metric`) wins. Otherwise
    default to the metric that looks like investments — it's always fetched,
    regardless of vehicle — falling back to the first metric if none matches.
    """
    if override:
        if override not in metrics:
            raise ValueError(
                f"share_likelihood_metric override '{override}' is not one of "
                f"the fetched metrics {metrics} (vehicle's default_metric + "
                "client's auxiliary_metric)."
            )
        return override
    return next((m for m in metrics if "invest" in m.lower()), metrics[0])


def _build_vars_per_dim(
    vehicle_spec: dict, cfg: dict, dims: list[str] | None, metrics: list[str]
) -> dict[str, list[str]]:
    """Build {dimension_name: [slug, ...]} mapping from vehicle spec.

    Any scalar field in the client `cfg` (brand, nameplate, etc.) is available to
    templates as a placeholder — new per-vehicle template variables need no code change.

    metrics: fetched metrics (vehicle's primary + client's auxiliary_metric), from
    _resolve_metrics(). Passed in rather than recomputed here so build_config()'s
    single call stays the one source of truth -- see its own call to
    resolve_share_likelihood_metric() for the other consumer of the same list.
    """
    vehicle_slug = vehicle_spec.get("vehicle_slug", "eletromidia")
    brand = cfg.get("brand", "")

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
        template = _get_template(vehicle_spec, bd)
        slugs = []
        for metric in metrics:
            for value in values:
                slug = template.format(**{
                    **cfg,
                    "metric": metric,
                    "vehicle": vehicle_slug,
                    "brand": brand,
                    "category": category,
                    "value": value,
                })
                slugs.append(slug)
        if slugs:
            result[bd_name] = slugs
    return result


def build_config(
    upgrade: Any,
    specs_path: str,
    media_var_override: str | None = None,
    specs: dict | None = None,
) -> DeepDiveConfig:
    """Build DeepDiveConfig from client YAML + UpgradeResult.

    Args:
        upgrade: UpgradeResult with contrib_df (used to validate media_var).
        specs_path: path to the client YAML (e.g. deepdive/configs/bradesco_eletro.yaml).
            Still required even when `specs` is given -- used to resolve
            `vehicle_specs_path` relative to the client YAML's own location.
        media_var_override: explicit aggregate channel column name; overrides YAML value.
        specs: pre-parsed client YAML (via load_yaml), to skip re-parsing when
            the caller already loaded it (e.g. to read a few fields before
            calling build_config -- see batch.py's run_single_client).
    """
    cfg = specs if specs is not None else load_yaml(specs_path)
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
    vehicle_specs = load_yaml(vehicle_specs_path)
    vehicle_key = cfg.get("vehicle", "eletromidia")
    available_vehicles = list(vehicle_specs.get("vehicles", {}).keys())
    if vehicle_key not in vehicle_specs.get("vehicles", {}):
        raise ValueError(
            f"Vehicle '{vehicle_key}' not found in {vehicle_specs_path}. "
            f"Available: {available_vehicles}"
        )
    vehicle_spec = vehicle_specs["vehicles"][vehicle_key]
    model_type = cfg.get("model_type", "stan")
    auxiliary_metric = cfg.get("auxiliary_metric", "")
    if not auxiliary_metric:
        raise ValueError(
            f"'auxiliary_metric' not set in {specs_path}. Every Deep Dive needs an "
            "explicit metric to drive the share-likelihood proxy — set it to a real "
            "exposure metric (e.g. impressions) when available, or to the same value "
            "as the investment metric when the vehicle has no exposure metric."
        )

    metrics = _resolve_metrics(vehicle_spec, auxiliary_metric)
    vars_per_dim = _build_vars_per_dim(vehicle_spec, cfg, dims_override, metrics)
    dims = list(vars_per_dim.keys())

    share_likelihood_metric = resolve_share_likelihood_metric(
        metrics, cfg.get("share_likelihood_metric")
    )

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
        share_likelihood_metric=share_likelihood_metric,
        auxiliary_metric=auxiliary_metric,
        share_prior_scale=cfg.get("share_prior_scale", 0.05),
        proxy_ct_tolerance=cfg.get("proxy_ct_tolerance", 0.15),
        num_steps=cfg.get("num_steps", 30_000),
        min_spend_share=cfg.get("min_spend_share", 0.02),
        hhi_threshold=cfg.get("hhi_threshold", 0.85),
        min_active_weeks=cfg.get("min_active_weeks", 2),
        min_active_weeks_frac=cfg.get("min_active_weeks_frac", 0.05),
        vehicle_spec=vehicle_spec,
        lower_funnel_vars_per_dim=cfg.get("lower_funnel_vars_per_dim") or {},
        upper_funnel_adstock_effect_per_dim=cfg.get("upper_funnel_adstock_effect_per_dim") or {},
    )
