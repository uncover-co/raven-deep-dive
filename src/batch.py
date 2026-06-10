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

from config import DeepDiveConfig, build_config, build_eletro_config
from diagnostics import run_diagnostics
from extraction import UpgradeResult, load_upgrade_auto, load_breakdown_spend
from pipeline import DDResult, run_deep_dive_e1
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
    if (specs.get("media_var") is None
            and model_type == "meridian"):
        return None, None, (
            f"media_var must be set in {specs_path} for Meridian models. "
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

        result = run_deep_dive_e1(config, upgrade, verbose=verbose)

        out_dir = os.path.join(output_base_dir, client_cfg.get("output_subdir", client_name))
        generate_report(result, output_dir=out_dir, client_name=client_name)

        return result, diag, None

    except Exception:
        err = traceback.format_exc()
        print(f"\n[FAIL] {client_name}:\n{err}")
        return None, None, err


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_deep_dive_batch(
    registry: dict[str, dict],
    registry_path: str,
    output_base_dir: str,
    clients: list[str] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, DDResult], dict[str, Any], dict[str, str]]:
    """Run deep dive for all (or selected) clients in registry.

    Args:
        registry: output of load_registry()
        registry_path: path to registry file (for resolving relative specs paths)
        output_base_dir: base dir for per-client outputs
        clients: list of client names to run; None = all
        verbose: pass to pipeline

    Returns:
        (results, diagnostics, errors)
        results: {client_name: DDResult}
        diagnostics: {client_name: DiagResult}
        errors: {client_name: traceback_string} for failed clients
    """
    target = clients or list(registry.keys())
    results, diagnostics, errors = {}, {}, {}

    for client_name in target:
        if client_name not in registry:
            print(f"[SKIP] {client_name} not in registry")
            continue
        result, diag, err = run_single_client(
            client_name, registry[client_name],
            registry_path=registry_path,
            output_base_dir=output_base_dir,
            verbose=verbose,
        )
        if err:
            errors[client_name] = err
        else:
            results[client_name] = result
            diagnostics[client_name] = diag

    print(f"\n{'='*66}")
    print(f"  Batch concluído: {len(results)} ok, {len(errors)} erros")
    if errors:
        print(f"  Erros: {list(errors.keys())}")
    print(f"{'='*66}")

    return results, diagnostics, errors


# ── Hierarchy rollups ─────────────────────────────────────────────────────────

def _extract_value_from_slug(slug: str, category: str) -> str | None:
    """Extract dimension value from slug. Handles $category:{cat}:{val} and $state:{val}."""
    m = _re.search(rf"\$category:{_re.escape(category)}:([^$]+)", slug)
    if m:
        return m.group(1)
    if category == "state":
        m = _re.search(r"\$state:([^$]+)", slug)
        if m:
            return m.group(1)
    return None


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
    category: str,
    flat_map: dict[str, str],
) -> pd.DataFrame:
    """Aggregate slug shares via flat {value → target} mapping."""
    agg_m: dict[str, float] = {}
    agg_s: dict[str, float] = {}
    for slug, sm in shares_model.items():
        val    = _extract_value_from_slug(slug, category) or slug.split(":")[-1]
        target = flat_map.get(val, val)
        agg_m[target] = agg_m.get(target, 0.0) + float(sm)
        agg_s[target] = agg_s.get(target, 0.0) + float(shares_spend.get(slug, 0.0))
    return _shares_df(agg_m, agg_s)


def _rollup_groups(
    shares_model: pd.Series,
    shares_spend: pd.Series,
    category: str,
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

    agg_m: dict[str, float] = {}
    agg_s: dict[str, float] = {}
    for slug, sm in shares_model.items():
        val = _extract_value_from_slug(slug, category) or slug.split(":")[-1]
        key = member_to_key.get(val, val)
        agg_m[key] = agg_m.get(key, 0.0) + float(sm)
        agg_s[key] = agg_s.get(key, 0.0) + float(shares_spend.get(slug, 0.0))

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

    if not rollup_specs:
        # No rollup → extract clean values from slugs for cross-client comparability
        agg_m: dict[str, float] = {}
        agg_s: dict[str, float] = {}
        for slug, sm in shares_model.items():
            key = (_extract_value_from_slug(slug, category)
                   if category else slug.split(":")[-1])
            agg_m[key] = agg_m.get(key, 0.0) + float(sm)
            agg_s[key] = agg_s.get(key, 0.0) + float(shares_spend.get(slug, 0.0))
        return {"raw": _shares_df(agg_m, agg_s)}

    result: dict[str, pd.DataFrame] = {}
    for rspec in rollup_specs:
        level = rspec["level"]
        if "map" in rspec:
            flat_map = hierarchy.get(rspec["map"], {})
            result[level] = _rollup_flat(shares_model, shares_spend, category, flat_map)
        elif "groups" in rspec:
            groups_spec = hierarchy.get(rspec["groups"], {})
            result[level] = _rollup_groups(
                shares_model, shares_spend, category,
                groups_spec,
                members_key=rspec.get("members_key", "values"),
                attr=rspec.get("attr"),
            )
        else:
            # No map/groups → clean slug extraction (identity rollup)
            agg_m: dict[str, float] = {}
            agg_s: dict[str, float] = {}
            for slug, sm in shares_model.items():
                key = (_extract_value_from_slug(slug, category)
                       if category else slug.split(":")[-1])
                agg_m[key] = agg_m.get(key, 0.0) + float(sm)
                agg_s[key] = agg_s.get(key, 0.0) + float(shares_spend.get(slug, 0.0))
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
        + vertical, tipo  (non-null only for Ambiente/grupo rollup)
    """
    rows = []
    for client, result in all_results.items():
        rollups = apply_hierarchy_rollups(result, vehicle_spec_override=vehicle_spec_override)

        for dim in result.config.dims:
            proxy = result.proxy_ratios.get(dim, float("nan"))
            csl_d = result.csl_devs.get(dim, float("nan"))

            for rollup_level, df in rollups.get(dim, {}).items():
                for _, row in df.iterrows():
                    rows.append({
                        "client":      client,
                        "dim":         dim,
                        "rollup":      rollup_level,
                        "item":        row["item"],
                        "vertical":    row.get("vertical", None),
                        "tipo":        row.get("tipo", None),
                        "share_model": row["share_model"],
                        "share_spend": row["share_spend"],
                        "roas_index":  row["roas_index"],
                        "proxy_ratio": proxy,
                        "csl_dev":     csl_d,
                    })
    return pd.DataFrame(rows)


def meta_summary(df_meta: pd.DataFrame) -> pd.DataFrame:
    """Pivot ROAS index: rows = item, columns = client × dim."""
    pivot = df_meta.pivot_table(
        index="item",
        columns=["client", "dim"],
        values="roas_index",
        aggfunc="mean",
    ).round(3)
    return pivot


def print_meta_report(df_meta: pd.DataFrame) -> None:
    """Print tabular meta-analysis report."""
    print(f"\n{'═'*72}")
    print("  META-ANÁLISE  —  ROAS Index por Cliente × Dimensão × Item")
    print(f"{'═'*72}")

    for (client, dim), group in df_meta.groupby(["client", "dim"]):
        proxy_str = f"proxy={group['proxy_ratio'].iloc[0]:.3f}" if len(group) > 0 else ""
        print(f"\n  [{client}] {dim}  ({proxy_str})")
        print(f"  {'item':<32} {'share_model':>11}  {'share_spend':>11}  {'ROAS index':>10}")
        print(f"  {'─'*68}")
        for _, row in group.sort_values("roas_index", ascending=False).iterrows():
            ri = f"{row['roas_index']:.3f}" if pd.notna(row["roas_index"]) else "  n/a "
            print(f"  {str(row['item'])[:32]:<32} "
                  f"{row['share_model']:>10.1%}  "
                  f"{row['share_spend']:>10.1%}  "
                  f"{ri:>10}")

    print(f"\n{'═'*72}")
    print(f"  Clientes: {sorted(df_meta['client'].unique())}")
    print(f"  Dimensões: {sorted(df_meta['dim'].unique())}")
    print(f"{'═'*72}")