from __future__ import annotations
import os
from datetime import datetime
import pandas as pd

from diagnostics import DiagnosisResult, sanitize_dim_name
from pipeline import align_to, extract_hill_params
from plots import _clean_label, plot_contributions, plot_roas_index, plot_weekly_df


def generate_report(
    result,
    diag: DiagnosisResult | None = None,
    output_dir: str = "outputs/",
    client_name: str = "",
) -> dict[str, str]:
    """Export all artefacts: CSVs + interactive Plotly HTML.

    Returns dict of {key: absolute_file_path}.
    """
    vehicle = result.config.vehicle
    folder = f"{client_name}_{vehicle}" if client_name else vehicle
    out = os.path.join(output_dir, folder)
    os.makedirs(out, exist_ok=True)
    paths: dict[str, str] = {}

    # ── metadata.csv ─────────────────────────────────────────────────────────
    idx = result.media_dd_contrib.index
    metadata = {
        "model_name": result.config.model_name,
        "client": client_name,
        "vehicle": result.config.vehicle,
        "upgrade_run_id": result.upgrade_run_id,
        "dd_date": datetime.today().date().isoformat(),
        "period_start": str(idx.min().date()),
        "period_end": str(idx.max().date()),
    }
    csv_meta = os.path.join(out, "metadata.csv")
    pd.DataFrame([metadata]).to_csv(csv_meta, index=False)
    paths["csv_metadata"] = csv_meta

    # ── hierarchy rollup ─────────────────────────────────────────────────────
    from batch import rollup_contribs_ts, apply_hierarchy_rollups
    vehicle_spec = result.config.vehicle_spec
    rollup_contribs_map: dict = {}
    rollup_spend_map: dict = {}
    for dim in result.config.dims:
        c_df = result.contribs.get(dim)
        fr = result.features_raw.get(dim)
        if c_df is not None and vehicle_spec:
            rollup_contribs_map[dim] = rollup_contribs_ts(c_df, dim, vehicle_spec)
        if fr is not None and vehicle_spec:
            rollup_spend_map[dim] = rollup_contribs_ts(fr, dim, vehicle_spec)
    rollup_shares_map = apply_hierarchy_rollups(result)

    # ── contributions.csv ─────────────────────────────────────────────────────
    csv_contrib = os.path.join(out, "contributions.csv")
    _build_contributions_df(
        result, rollup_contribs_map, rollup_spend_map, rollup_shares_map
    ).to_csv(csv_contrib, index=False)
    paths["csv_contributions"] = csv_contrib

    # ── diagnostics.csv ───────────────────────────────────────────────────────
    if diag is not None:
        csv_diag = os.path.join(out, "diagnostics.csv")
        _build_diagnostics_df(result, diag).to_csv(csv_diag, index=False)
        paths["csv_diagnostics"] = csv_diag

    # ── hill_params.csv ───────────────────────────────────────────────────────
    y_max = float(result.media_dd_contrib.max())
    hill_frames = []
    for dim in result.config.dims:
        model = result.models.get(dim)
        if model is None:
            continue
        variaveis = result.config.vars_per_dim.get(dim, [])
        df_h = extract_hill_params(
            raven2_model=model,
            variaveis=variaveis,
            features_raw=result.features_raw.get(dim),
            col_maxes=result.col_maxes.get(dim),
            y_max=y_max,
        )
        df_h.insert(0, "dim", dim)
        hill_frames.append(df_h)
    if hill_frames:
        csv_hill = os.path.join(out, "hill_params.csv")
        pd.concat(hill_frames).reset_index().to_csv(csv_hill, index=False)
        paths["csv_hill_params"] = csv_hill

    # ── contributions.html ────────────────────────────────────────────────────
    html_c = os.path.join(out, "contributions.html")
    plot_contributions(result).write_html(html_c)
    paths["html_contributions"] = html_c

    # ── roas_index.html ───────────────────────────────────────────────────────
    html_r = os.path.join(out, "roas_index.html")
    plot_roas_index(result).write_html(html_r)
    paths["html_roas"] = html_r

    # ── weekly HTML per dim + rollup levels ───────────────────────────────────
    ct = result.media_dd_contrib
    for dim in result.config.dims:
        c_df = result.contribs.get(dim)
        if c_df is None:
            continue
        html_w = os.path.join(out, f"weekly_{dim}.html")
        plot_weekly_df(c_df, ct, title=f"Contribs Semanais — {dim}").write_html(html_w)
        paths[f"html_weekly_{dim}"] = html_w

        for level, rollup_df in rollup_contribs_map.get(dim, {}).items():
            html_w = os.path.join(out, f"weekly_{dim}_{level}.html")
            plot_weekly_df(
                rollup_df, ct, title=f"Contribs Semanais — {dim} → {level}"
            ).write_html(html_w)
            paths[f"html_weekly_{dim}_{level}"] = html_w

    print(f"Report saved → {out}")
    for k, v in paths.items():
        print(f"  {k}: {os.path.basename(v)}")

    return paths


def _build_contributions_df(
    result,
    rollup_contribs_map: dict | None = None,
    rollup_spend_map: dict | None = None,
    rollup_shares_map: dict | None = None,
) -> pd.DataFrame:
    rows = []
    for dim in result.config.dims:
        c_df = result.contribs.get(dim)
        if c_df is None:
            continue

        anchor_stan = float(align_to(result.media_dd_contrib, c_df.index).sum())
        dim_contrib_total = float(c_df.sum().sum())

        fr = result.features_raw.get(dim)
        sh_s = result.shares_spend.get(dim, pd.Series(dtype=float))

        proxy_r = result.proxy_ratios.get(dim, float("nan"))
        csl_d = result.csl_devs.get(dim, float("nan"))
        r2_v = result.r2.get(dim, float("nan"))
        wape_v = result.wape.get(dim, float("nan"))

        def _row(level, item, contrib_abs, contrib_total, spend_abs, spend_share):
            contrib_share = contrib_abs / (contrib_total + 1e-12)
            roas_idx = contrib_share / spend_share if spend_share > 0 else float("nan")
            return {
                "dim": dim,
                "level": level,
                "item": item,
                "item_label": _clean_label(str(item)),
                "anchor_stan": anchor_stan,
                "contrib_absolute": contrib_abs,
                "pct_anchor": contrib_abs / anchor_stan if anchor_stan > 0 else float("nan"),
                "contrib_share": contrib_share,
                "spend_absolute": spend_abs,
                "spend_share": spend_share,
                "roas_index": roas_idx,
                "proxy_ratio": proxy_r,
                "csl_max_dev": csl_d,
                "r2": r2_v,
                "wape": wape_v,
            }

        # ── Atomic-level rows ─────────────────────────────────────────────────
        for item in c_df.columns:
            contrib_abs = float(c_df[item].sum())
            spend_abs = float(fr[item].sum()) if fr is not None and item in fr.columns else float("nan")
            spend_share = float(sh_s.get(item, float("nan")))
            rows.append(_row(dim, item, contrib_abs, dim_contrib_total, spend_abs, spend_share))

        # ── Rollup-level rows ─────────────────────────────────────────────────
        if rollup_contribs_map:
            for level, rollup_df in rollup_contribs_map.get(dim, {}).items():
                rollup_total = float(rollup_df.sum().sum())
                spend_rollup_df = (rollup_spend_map or {}).get(dim, {}).get(level)
                shares_df = (rollup_shares_map or {}).get(dim, {}).get(level, pd.DataFrame())
                spend_share_lookup: dict = {}
                if not shares_df.empty and "item" in shares_df.columns:
                    spend_share_lookup = dict(
                        zip(shares_df["item"], shares_df["share_spend"])
                    )
                for item in rollup_df.columns:
                    contrib_abs = float(rollup_df[item].sum())
                    spend_abs = (
                        float(spend_rollup_df[item].sum())
                        if spend_rollup_df is not None and item in spend_rollup_df.columns
                        else float("nan")
                    )
                    spend_share = float(spend_share_lookup.get(item, float("nan")))
                    rows.append(_row(level, item, contrib_abs, rollup_total, spend_abs, spend_share))

    return pd.DataFrame(rows)


def _build_diagnostics_df(result, diag: DiagnosisResult) -> pd.DataFrame:
    contrib_lookup: dict[tuple, tuple] = {}
    for dim, c_df in result.contribs.items():
        dim_total = float(c_df.sum().sum())
        for slug in c_df.columns:
            total = float(c_df[slug].sum())
            contrib_lookup[(dim, slug)] = (total, total / (dim_total + 1e-12))

    _RC_TO_STATUS = {
        "dim_skip": "dim_skip",
        "no_spend":  "discarded_sem_spend",
        "low_weeks": "discarded_semanas",
        "low_pct":   "discarded_pct",
    }

    def _status(row) -> str:
        rc = row.get("reason_code", "")
        if rc in _RC_TO_STATUS:
            return _RC_TO_STATUS[rc]
        if str(row["slug"]).startswith("__outros__"):
            return "outros_aggregate"
        return "kept"

    rows = []
    for _, r in diag.spend_report.iterrows():
        ct, pct_c = contrib_lookup.get((r["dim"], r["slug"]), (None, None))
        rows.append({
            "dim": r["dim"],
            "slug": r["slug"],
            "status": _status(r),
            "reason": r["reason"],
            "semanas_ativas": r["semanas_ativas"],
            "spend_total": r["spend_total"],
            "pct_spend_dim": r["pct_dim"],
            "contrib_total": ct,
            "pct_contrib_dim": pct_c,
        })

    for dim, bucketed_slugs in diag.bucketed.items():
        outros_col = f"__outros__{sanitize_dim_name(dim)}"
        ct, pct_c = contrib_lookup.get((dim, outros_col), (None, None))
        base = diag.spend_report[
            (diag.spend_report["dim"] == dim) &
            (diag.spend_report["slug"].isin(bucketed_slugs))
        ]
        rows.append({
            "dim": dim,
            "slug": outros_col,
            "status": "outros_aggregate",
            "reason": f"agrupa {len(bucketed_slugs)} quebra(s)",
            "semanas_ativas": int(base["semanas_ativas"].max()) if len(base) else None,
            "spend_total": float(base["spend_total"].sum()) if len(base) else None,
            "pct_spend_dim": float(base["pct_dim"].sum()) if len(base) else None,
            "contrib_total": ct,
            "pct_contrib_dim": pct_c,
        })

    return pd.DataFrame(rows)
