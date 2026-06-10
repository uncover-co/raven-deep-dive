from __future__ import annotations
from dataclasses import dataclass
import re
import pandas as pd

from config import DeepDiveConfig
from extraction import UpgradeResult


@dataclass
class DiagnosisResult:
    spend_report: pd.DataFrame       # per-var stats: share, HHI, semanas_ativas, keep
    bucketed: dict[str, list[str]]   # dim → vars bucketed into __outros__
    skipped_dims: list[str]          # dims skipped (HHI too high or < 2 active)


def run_diagnostics(
    config: DeepDiveConfig,
    upgrade: UpgradeResult,
    min_spend_share: float = 0.05,
    hhi_threshold: float = 0.85,
    min_active_weeks: int = 2,
) -> tuple[DeepDiveConfig, DiagnosisResult]:
    """Filter config vars by spend structure; bucket tiny vars into __outros__.

    Ported from utils.py:diagnose_dd_config (line 782).
    Returns updated DeepDiveConfig (vars_per_dim filtered) + DiagnosisResult.

    Side-effect: adds __outros__ columns to upgrade.spend_df for bucketed dims.
    """
    df = upgrade.spend_df.copy()
    rows: list[dict] = []
    new_vars_per_dim: dict[str, list[str]] = {}
    bucketed: dict[str, list[str]] = {}
    skipped_dims: list[str] = []
    n_weeks = len(df)

    for dim, slugs in config.vars_per_dim.items():
        stats: dict[str, dict] = {}
        for slug in slugs:
            if slug in df.columns:
                s = df[slug]
                stats[slug] = {"total": float(s.sum()), "active": int((s > 0).sum())}
            else:
                stats[slug] = {"total": 0.0, "active": 0}

        cat_total = sum(v["total"] for v in stats.values())
        n_active = sum(1 for v in stats.values() if v["total"] > 0)
        hhi = (
            sum((v["total"] / cat_total) ** 2 for v in stats.values())
            if cat_total > 0 else 1.0
        )

        if n_active < 2 or hhi > hhi_threshold:
            skipped_dims.append(dim)
            for slug, d in stats.items():
                rows.append(_make_row(dim, slug, d, cat_total, n_weeks, hhi, rec="SKIP", keep=False, reason="dim SKIP"))
            continue

        kept, excl = [], []
        for slug, d in stats.items():
            pct = d["total"] / cat_total if cat_total > 0 else 0.0
            if d["total"] == 0:
                keep, reason = False, "sem spend"
            elif pct < min_spend_share:
                keep, reason = False, f"pct {pct:.1%} < {min_spend_share:.0%}"
            elif d["active"] < min_active_weeks:
                keep, reason = False, f"só {d['active']} semana(s)"
            else:
                keep, reason = True, ""

            rows.append(_make_row(dim, slug, d, cat_total, n_weeks, hhi, rec="DD", keep=keep, reason=reason))
            if keep:
                kept.append(slug)
            elif d["total"] > 0:
                excl.append(slug)

        if kept:
            if excl:
                outros_col = f"__outros__{dim.lower().replace(' ', '-')}"
                df[outros_col] = df[excl].sum(axis=1)
                kept.append(outros_col)
                bucketed[dim] = excl
            new_vars_per_dim[dim] = kept

    # Update spend_df in-place so downstream pipeline sees __outros__ cols
    upgrade.spend_df = df

    spend_report = pd.DataFrame(rows)
    _print_diagnosis(spend_report, min_spend_share, hhi_threshold)

    new_config = DeepDiveConfig(
        dims=[d for d in config.dims if d in new_vars_per_dim],
        vars_per_dim=new_vars_per_dim,
        media_var=config.media_var,
        brand=config.brand,
        vehicle=config.vehicle,
        share_prior_scale=config.share_prior_scale,
        proxy_ct_tolerance=config.proxy_ct_tolerance,
        num_steps=config.num_steps,
    )
    return new_config, DiagnosisResult(
        spend_report=spend_report,
        bucketed=bucketed,
        skipped_dims=skipped_dims,
    )


def _slug_label(slug: str) -> str:
    """Extract short human-readable label from a full slug for display."""
    parts = re.findall(r'\$category:[^$:]+:([^$]+)', slug)
    if parts:
        return parts[-1][:24]
    return slug[:24]


def _print_diagnosis(diag_df: pd.DataFrame, min_pct: float, hhi_threshold: float) -> None:
    w = 75
    print("─" * w)
    print(f"  DIAGNÓSTICO DEEP DIVE  (min_pct={min_pct:.0%}, HHI>{hhi_threshold}=SKIP)")
    print("─" * w)
    print(f"  {'Dimensão':<28}  {'Rec':>5}  {'HHI':>5}  {'Total':>6}  {'Mantém':>6}  {'Exclui':>6}")
    print(f"  {'─' * 68}")
    for dim, grp in diag_df.groupby("dim", sort=False):
        rec = grp["rec"].iloc[0]
        hhi = grp["hhi"].iloc[0]
        n_tot = len(grp)
        n_kp = int(grp["keep"].sum())
        n_ex = n_tot - n_kp
        flag = "⚠️  " if rec == "SKIP" else "✓  "
        print(f"  {flag}{dim:<26}  {rec:>5}  {hhi:>5.2f}  {n_tot:>6}  {n_kp:>6}  {n_ex:>6}")
        for _, row in grp[~grp["keep"]].iterrows():
            label = _slug_label(row["slug"])
            print(f"       ↳ {label:<24}  {row['pct_dim']:>6.1%}  {row['reason']}")
        outros_rows = grp[grp["slug"].str.startswith("__outros__") & grp["keep"]]
        for _, row in outros_rows.iterrows():
            print(f"       → {'outros':<24}  {row['pct_dim']:>6.1%}  {row['reason']}")
    print("─" * w)
    real_kept = diag_df[diag_df["keep"] & ~diag_df["slug"].str.startswith("__outros__")]
    n_dd = diag_df[diag_df["rec"] == "DD"]["dim"].nunique()
    n_sk = diag_df[diag_df["rec"] == "SKIP"]["dim"].nunique()
    n_q = int(real_kept.shape[0])
    n_outros = int(diag_df[diag_df["slug"].str.startswith("__outros__") & diag_df["keep"]].shape[0])
    print(f"  Dimensões com DD: {n_dd}  |  SKIP: {n_sk}  |  Quebras mantidas: {n_q}  |  Grupos 'outros': {n_outros}")
    print("─" * w)


def _make_row(dim, slug, d, cat_total, n_weeks, hhi, rec, keep, reason):
    pct = d["total"] / cat_total if cat_total > 0 else 0.0
    return {
        "dim": dim,
        "slug": slug,
        "spend_total": d["total"],
        "pct_dim": pct,
        "semanas_ativas": d["active"],
        "pct_ativo": d["active"] / n_weeks if n_weeks > 0 else 0.0,
        "hhi": round(hhi, 3),
        "rec": rec,
        "keep": keep,
        "reason": reason,
    }
