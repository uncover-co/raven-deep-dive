"""Batch Deep Dive runner and meta-analysis utilities.

Usage:
    from batch import run_deep_dive_batch, consolidate_results, meta_analysis_plots

    registry = load_registry("../configs/clients_registry.yaml")
    all_results, all_diags = run_deep_dive_batch(registry, output_base_dir="../outputs")
    df_meta = consolidate_results(all_results)
"""
from __future__ import annotations

import os
import re as _re
import traceback
from datetime import datetime
from typing import Any

import pandas as pd
import yaml

from config import DeepDiveConfig, build_config
from diagnostics import run_diagnostics
from extraction import UpgradeResult, load_upgrade_auto, load_breakdown_spend
from pipeline import DDResult, run_deep_dive
from report import generate_report

DEFAULT_MLFLOW_URI = "https://mlflow-dev.cloud.uncover.co"


# ── Registry ──────────────────────────────────────────────────────────────────

def load_registry(registry_path: str) -> dict[str, dict]:
    """Load clients_registry.yaml. Returns {client_name: cfg_dict}."""
    with open(registry_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("clients", {})


def _resolve_specs_path(specs_path: str, registry_path: str) -> str:
    """Resolve specs_path relative to registry file location."""
    base = os.path.dirname(os.path.abspath(registry_path))
    return os.path.normpath(os.path.join(base, specs_path))


# ── Single-client runner ───────────────────────────────────────────────────────

def run_single_client(
    client_name: str,
    client_cfg: dict,
    registry_path: str,
    output_base_dir: str,
    verbose: bool = True,
) -> tuple[DDResult | None, Any | None, str | None]:
    """Run full deep dive pipeline for one client.

    Returns (result, diag, error_message). error_message is None on success.
    """
    specs_path = _resolve_specs_path(client_cfg["specs_path"], registry_path)

    with open(specs_path, "r", encoding="utf-8") as f:
        specs = yaml.safe_load(f) or {}

    tracking_uri = specs.get("mlflow_tracking_uri", DEFAULT_MLFLOW_URI)
    run_id       = specs.get("upgrade_run_id")
    model_type   = specs.get("model_type", client_cfg.get("model_type", "stan"))
    workspace_dd = specs.get("workspace_dd")
    start_date   = datetime.fromisoformat(specs["start_date"])
    end_date     = datetime.fromisoformat(specs["end_date"])

    if not run_id:
        return None, None, f"upgrade_run_id not set in {specs_path}"
    if not workspace_dd:
        return None, None, f"workspace_dd not set in {specs_path}"
    if specs.get("media_var") is None and model_type in ("meridian", "stan"):
        return None, None, (
            f"media_var must be set in {specs_path} for {model_type} models. "
            "Load the upgrade manually and inspect upgrade.contrib_df.columns to find it."
        )

    try:
        print(f"\n{'='*66}")
        print(f"  [{client_name.upper()}]  model={model_type}  run={run_id[:8]}...")
        print(f"{'='*66}")

        upgrade = load_upgrade_auto(run_id, model_type=model_type, tracking_uri=tracking_uri)
        config  = build_config(upgrade, specs_path)

        all_vars = [v for slugs in config.vars_per_dim.values() for v in slugs]
        print(f"  Loading breakdown spend ({len(all_vars)} vars)...")
        upgrade.spend_df = load_breakdown_spend(workspace_dd, all_vars, start_date, end_date)

        config, diag = run_diagnostics(config, upgrade)

        result = run_deep_dive(config, upgrade, verbose=verbose, upgrade_run_id=run_id)

        generate_report(
            result,
            diag=diag,
            output_dir=output_base_dir,
            client_name=client_cfg.get("output_subdir", client_name),
        )

        return result, diag, None

    except Exception:
        err = traceback.format_exc()
        print(f"\n[FAIL] {client_name}:\n{err}")
        return None, None, err


# ── Batch runner ──────────────────────────────────────────────────────────────

def _iter_runs(registry: dict) -> list[tuple[str, str, dict]]:
    """Flatten registry into (client_name, run_key, cfg) triples.

    Supports both formats:
      - Legacy: client entry has specs_path at root (single vehicle).
      - Multi-vehicle: client entry has a `vehicles: {veh: {specs_path: ...}}` dict.

    run_key = client_name for legacy, or f"{client_name}_{vehicle}" for multi-vehicle.
    """
    runs = []
    for client_name, client_cfg in registry.items():
        if "vehicles" in client_cfg:
            base = {k: v for k, v in client_cfg.items() if k != "vehicles"}
            for veh_name, veh_cfg in client_cfg["vehicles"].items():
                run_key = f"{client_name}_{veh_name}"
                merged = {**base, **veh_cfg}
                runs.append((client_name, run_key, merged))
        else:
            runs.append((client_name, client_name, client_cfg))
    return runs


def run_deep_dive_batch(
    registry: dict[str, dict],
    registry_path: str,
    output_base_dir: str,
    clients: list[str] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, DDResult], dict[str, Any], dict[str, str]]:
    """Run deep dive for all (or selected) clients/vehicles in registry.

    Args:
        registry: output of load_registry()
        registry_path: path to registry file (for resolving relative specs paths)
        output_base_dir: base dir for per-client outputs
        clients: filter by client name (runs all vehicles for that client) OR
                 by run_key (e.g. "bradesco_tiktok" for a specific vehicle).
                 None = all.
        verbose: pass to pipeline

    Returns:
        (results, diagnostics, errors) keyed by run_key.
    """
    target = set(clients) if clients else None
    results, diagnostics, errors = {}, {}, {}

    for client_name, run_key, run_cfg in _iter_runs(registry):
        if target and client_name not in target and run_key not in target:
            print(f"[SKIP] {run_key}")
            continue
        result, diag, err = run_single_client(
            run_key, run_cfg,
            registry_path=registry_path,
            output_base_dir=output_base_dir,
            verbose=verbose,
        )
        if err:
            errors[run_key] = err
        else:
            results[run_key] = result
            diagnostics[run_key] = diag

    print(f"\n{'='*66}")
    print(f"  Batch concluído: {len(results)} ok, {len(errors)} erros")
    if errors:
        print(f"  Erros: {list(errors.keys())}")
    print(f"{'='*66}")

    return results, diagnostics, errors


# ── Hierarchy rollups ─────────────────────────────────────────────────────────

def _build_slug_extractor(vehicle_spec: dict, category: str):
    """Build a slug→value extractor from model templates in vehicle_spec."""
    patterns = []
    for model_spec in vehicle_spec.get("models", {}).values():
        for tmpl_key in ("default_template", "state_template"):
            tmpl = model_spec.get(tmpl_key, "")
            for segment in tmpl.split("$"):
                if "{value}" not in segment:
                    continue
                pat = segment.replace("{category}", _re.escape(category))
                pat = _re.sub(r"\{(?!value\})[^}]+\}", r"[^$]+", pat)
                pat = pat.replace("{value}", r"([^$]+)")
                patterns.append(_re.compile(r"\$" + pat))

    def extract(slug: str) -> str | None:
        for p in patterns:
            m = p.search(slug)
            if m:
                return m.group(1)
        return None

    return extract


def _aggregate_shares(
    shares_model: pd.Series,
    shares_spend: pd.Series,
    key_fn,
) -> tuple[dict[str, float], dict[str, float]]:
    """Accumulate model and spend shares grouped by key_fn(slug)."""
    agg_m: dict[str, float] = {}
    agg_s: dict[str, float] = {}
    for slug, sm in shares_model.items():
        k = key_fn(slug)
        agg_m[k] = agg_m.get(k, 0.0) + float(sm)
        agg_s[k] = agg_s.get(k, 0.0) + float(shares_spend.get(slug, 0.0))
    return agg_m, agg_s


def _shares_df(item_model: dict, item_spend: dict, extra_cols: dict | None = None) -> pd.DataFrame:
    """Build share DataFrame from accumulated dicts, renormalized to sum=1.

    extra_cols: {col_name: {item: value}} for additional per-item metadata.
    """
    sm_total = sum(item_model.values()) or 1.0
    ss_total = sum(item_spend.values()) or 1.0
    rows = []
    for k, sm in sorted(item_model.items(), key=lambda x: -x[1]):
        sm_n = sm / sm_total
        ss_n = item_spend.get(k, 0.0) / ss_total
        row: dict = {
            "item":        k,
            "share_model": sm_n,
            "share_spend": ss_n,
            "roas_index":  sm_n / ss_n if ss_n > 0 else float("nan"),
        }
        if extra_cols:
            for col, mapping in extra_cols.items():
                row[col] = mapping.get(k)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["item", "share_model", "share_spend", "roas_index"]
    )


def _rollup_flat(
    shares_model: pd.Series,
    shares_spend: pd.Series,
    extract,
    flat_map: dict[str, str],
) -> pd.DataFrame:
    """Aggregate slug shares via flat {value → target} mapping."""
    def _key(slug):
        val = extract(slug) or slug.split(":")[-1]
        return flat_map.get(val, val)
    agg_m, agg_s = _aggregate_shares(shares_model, shares_spend, _key)
    return _shares_df(agg_m, agg_s)


def _rollup_groups(
    shares_model: pd.Series,
    shares_spend: pd.Series,
    extract,
    groups_spec: dict[str, dict],
    members_key: str,
    attr: str | None,
) -> pd.DataFrame:
    """Aggregate slug shares via groups spec.

    attr=None  → aggregate to group name (adds other group attributes as extra cols).
    attr='foo' → aggregate to the value of attribute 'foo' within each group.
    """
    # Build member → key mapping
    member_to_key:  dict[str, str]         = {}
    group_attrs:    dict[str, dict]        = {}

    for gname, gspec in groups_spec.items():
        key = gspec.get(attr, gname) if attr else gname
        for member in gspec.get(members_key, []):
            member_to_key[member] = key
        if not attr:
            # Collect all scalar attributes for extra_cols metadata
            group_attrs[gname] = {
                k: v for k, v in gspec.items()
                if k != members_key and not isinstance(v, list)
            }

    def _key(slug):
        val = extract(slug) or slug.split(":")[-1]
        return member_to_key.get(val, val)
    agg_m, agg_s = _aggregate_shares(shares_model, shares_spend, _key)

    # Build extra_cols for group-level rollup (attr=None)
    extra: dict | None = None
    if not attr and group_attrs:
        all_extra_attrs = {a for attrs in group_attrs.values() for a in attrs}
        extra = {ea: {gname: attrs.get(ea) for gname, attrs in group_attrs.items()}
                 for ea in all_extra_attrs}

    return _shares_df(agg_m, agg_s, extra_cols=extra)


def rollup_dim(
    dim: str,
    shares_model: pd.Series,
    shares_spend: pd.Series,
    vehicle_spec: dict,
) -> dict[str, pd.DataFrame]:
    """Generic rollup for any dimension based on vehicle_spec rollup config.

    Reads breakdown.rollups list from vehicle_spec. Each entry declares:
      - level: str          rollup key in output dict
      - map: str            key in hierarchy for flat {value: target} mapping
      - groups: str         key in hierarchy for group spec dict
      - members_key: str    key inside each group listing member values
      - attr: str           (optional) aggregate by this group attribute instead of name

    Falls back to clean slug extraction ("raw") if no rollups are defined.
    """
    bd_spec      = vehicle_spec.get("breakdowns", {}).get(dim, {})
    category     = bd_spec.get("category", "")
    rollup_specs = bd_spec.get("rollups", [])
    hierarchy    = vehicle_spec.get("hierarchy", {})
    extract      = _build_slug_extractor(vehicle_spec, category) if category else lambda slug: None

    def _key(slug: str) -> str:
        return extract(slug) or slug.split(":")[-1]

    if not rollup_specs:
        agg_m, agg_s = _aggregate_shares(shares_model, shares_spend, _key)
        return {"raw": _shares_df(agg_m, agg_s)}

    result: dict[str, pd.DataFrame] = {}
    for rspec in rollup_specs:
        level = rspec["level"]
        if "map" in rspec:
            map_key = rspec["map"]
            if map_key not in hierarchy:
                raise ValueError(
                    f"Rollup '{level}' in dim '{dim}' references map '{map_key}' "
                    f"not found in hierarchy. Available: {list(hierarchy.keys())}"
                )
            flat_map = hierarchy[map_key]
            result[level] = _rollup_flat(shares_model, shares_spend, extract, flat_map)
        elif "groups" in rspec:
            groups_key = rspec["groups"]
            if groups_key not in hierarchy:
                raise ValueError(
                    f"Rollup '{level}' in dim '{dim}' references groups '{groups_key}' "
                    f"not found in hierarchy. Available: {list(hierarchy.keys())}"
                )
            groups_spec = hierarchy[groups_key]
            result[level] = _rollup_groups(
                shares_model, shares_spend, extract,
                groups_spec,
                members_key=rspec.get("members_key", "values"),
                attr=rspec.get("attr"),
            )
        else:
            agg_m, agg_s = _aggregate_shares(shares_model, shares_spend, _key)
            result[level] = _shares_df(agg_m, agg_s)
    return result


def apply_hierarchy_rollups(
    result: "DDResult",
    vehicle_spec_override: dict | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Apply vehicle_spec-driven rollups per dimension.

    vehicle_spec_override: pass explicitly if result.config.vehicle_spec is empty
    (e.g. batch ran before vehicle_spec was added to DeepDiveConfig).
    """
    vehicle_spec = vehicle_spec_override or result.config.vehicle_spec
    if not vehicle_spec:
        print(
            "  [!] vehicle_spec vazio — rollups não serão aplicados (tudo 'raw').\n"
            "      Re-rode o batch OU passe vehicle_spec_override= para consolidate_results()."
        )
    return {
        dim: rollup_dim(
            dim,
            result.shares_model.get(dim, pd.Series(dtype=float)),
            result.shares_spend.get(dim, pd.Series(dtype=float)),
            vehicle_spec,
        )
        for dim in result.config.dims
    }


def rollup_contribs_ts(
    contribs: pd.DataFrame,
    dim: str,
    vehicle_spec: dict,
) -> dict[str, pd.DataFrame]:
    """Aggregate a contribution (or spend) time-series by rollup levels in vehicle_spec.

    Mirrors rollup_dim() but operates on raw DataFrame(timestamp × slug) instead
    of pre-aggregated share Series. Works for both contribs and features_raw.

    Args:
        contribs: DataFrame with datetime index and slug columns.
        dim: breakdown dimension name (e.g. 'product_level_4').
        vehicle_spec: full vehicle spec dict from vehicle_specs.yaml.

    Returns:
        {level_name: DataFrame(index=timestamp, columns=aggregated_values)}
    """
    bd_spec = vehicle_spec.get("breakdowns", {}).get(dim, {})
    category = bd_spec.get("category", "")
    rollup_specs = bd_spec.get("rollups", [])
    hierarchy = vehicle_spec.get("hierarchy", {})
    extract = _build_slug_extractor(vehicle_spec, category) if category else lambda _: None

    def _value(slug: str) -> str:
        return extract(slug) or slug.split(":")[-1]

    def _group_and_sum(parent_fn) -> pd.DataFrame:
        groups: dict[str, list[str]] = {}
        for slug in contribs.columns:
            groups.setdefault(parent_fn(slug), []).append(slug)
        return pd.DataFrame(
            {p: contribs[cols].sum(axis=1) for p, cols in groups.items()}
        )

    if not rollup_specs:
        return {"raw": _group_and_sum(_value)}

    out: dict[str, pd.DataFrame] = {}
    for rspec in rollup_specs:
        level = rspec["level"]
        if "map" in rspec:
            flat_map = hierarchy.get(rspec["map"], {})

            def _pf(slug: str, _m: dict = flat_map) -> str:
                v = _value(slug)
                return _m.get(v, v)

            out[level] = _group_and_sum(_pf)
        elif "groups" in rspec:
            groups_spec = hierarchy.get(rspec["groups"], {})
            members_key = rspec.get("members_key", "values")
            attr = rspec.get("attr")
            member_to_key: dict[str, str] = {}
            for gname, gspec in groups_spec.items():
                key = gspec.get(attr, gname) if attr else gname
                for member in gspec.get(members_key, []):
                    member_to_key[member] = key

            def _pg(slug: str, _m: dict = member_to_key) -> str:
                v = _value(slug)
                return _m.get(v, v)

            out[level] = _group_and_sum(_pg)
        else:
            out[level] = _group_and_sum(_value)
    return out


# ── Meta-analysis ─────────────────────────────────────────────────────────────

def consolidate_results(
    all_results: dict[str, DDResult],
    vehicle_spec_override: dict | None = None,
) -> pd.DataFrame:
    """Aggregate shares + ROAS index + proxy_ratio across all clients.

    Args:
        all_results: output of run_deep_dive_batch().
        vehicle_spec_override: pass the full vehicle spec dict if all_results was produced
            by an older batch run where config.vehicle_spec was empty.  Example:
                from config import _load_yaml
                vs = _load_yaml("../data/vehicle_specs.yaml")["vehicles"]["eletromidia"]
                df_meta = consolidate_results(all_results, vehicle_spec_override=vs)

    Returns long-form DataFrame with columns:
        client, dim, rollup, item, share_model, share_spend, roas_index,
        proxy_ratio, csl_dev
        + any extra group attributes from rollup DataFrames (vehicle-specific)
    """
    rows = []
    for client, result in all_results.items():
        rollups = apply_hierarchy_rollups(result, vehicle_spec_override=vehicle_spec_override)

        for dim in result.config.dims:
            proxy = result.proxy_ratios.get(dim, float("nan"))
            csl_d = result.csl_devs.get(dim, float("nan"))

            _base_cols = {"item", "share_model", "share_spend", "roas_index"}
            for rollup_level, rollup_df in rollups.get(dim, {}).items():
                extra_cols = [c for c in rollup_df.columns if c not in _base_cols]
                for _, row in rollup_df.iterrows():
                    entry = {
                        "client":      client,
                        "dim":         dim,
                        "rollup":      rollup_level,
                        "item":        row["item"],
                        "share_model": row["share_model"],
                        "share_spend": row["share_spend"],
                        "roas_index":  row["roas_index"],
                        "proxy_ratio": proxy,
                        "csl_dev":     csl_d,
                    }
                    for col in extra_cols:
                        entry[col] = row.get(col)
                    rows.append(entry)
    return pd.DataFrame(rows)


