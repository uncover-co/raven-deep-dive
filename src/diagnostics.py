from __future__ import annotations
from dataclasses import dataclass
import re
import pandas as pd


def sanitize_dim_name(name: str) -> str:
    """Slugify a dimension name for use as a DataFrame column or filename."""
    return re.sub(r"[^\w-]", "", name.lower().replace(" ", "-"))

from config import DeepDiveConfig
from extraction import UpgradeResult


@dataclass
class DiagnosisResult:
    spend_report: pd.DataFrame       # per-var stats: share, HHI, active weeks, keep
    bucketed: dict[str, list[str]]  # dim -> variables bucketed into __others__
    skipped_dims: list[str]          # dims skipped (HHI too high or < 2 active)
    auxiliary_metric_dfs: dict[str, pd.DataFrame] | None = None  # dim -> df aligned to final kept cols (config.auxiliary_metric values), feeds ContributionShareLikelihood prior only


def run_diagnostics(
    config: DeepDiveConfig,
    upgrade: UpgradeResult,
    min_spend_share: float | None = None,
    hhi_threshold: float | None = None,
    min_active_weeks: int | None = None,
    min_active_weeks_frac: float | None = None,
) -> tuple[DeepDiveConfig, DiagnosisResult]:
    """Filter config vars by spend structure; bucket tiny vars into __others__.

    Returns updated DeepDiveConfig (vars_per_dim filtered) + DiagnosisResult.

    Side-effect: adds __others__ columns to upgrade.spend_df for bucketed dims.
    """
    min_spend_share = min_spend_share if min_spend_share is not None else config.min_spend_share
    hhi_threshold = hhi_threshold if hhi_threshold is not None else config.hhi_threshold
    min_active_weeks = min_active_weeks if min_active_weeks is not None else config.min_active_weeks
    min_active_weeks_frac = (
        min_active_weeks_frac if min_active_weeks_frac is not None else config.min_active_weeks_frac
    )
    if not config.share_likelihood_metric:
        raise ValueError(
            "config.share_likelihood_metric is not set. build_config() always fills this in; "
            "if you built DeepDiveConfig by hand, pass share_likelihood_metric explicitly."
        )
    if not config.auxiliary_metric:
        raise ValueError(
            "config.auxiliary_metric is not set. build_config() requires it; if you built "
            "DeepDiveConfig by hand, pass auxiliary_metric explicitly (the same value as "
            "share_likelihood_metric when the vehicle has no real exposure metric)."
        )

    df = upgrade.spend_df.copy()
    rows: list[dict] = []
    new_vars_per_dim: dict[str, list[str]] = {}
    bucketed: dict[str, list[str]] = {}
    skipped_dims: list[str] = []
    # Carried through as-is, except: an __others__ bucket where every member
    # was itself configured lower funnel inherits that (a residual made only
    # of no-adstock variables is still no-adstock). A mixed bucket (some
    # members lower, some not) has no unambiguous answer -- it keeps the
    # existing system-wide default (upper/adstocked) rather than guessing;
    # see README Sec. 8 for why this is a known, accepted limitation.
    new_lower_funnel_vars_per_dim = {k: list(v) for k, v in config.lower_funnel_vars_per_dim.items()}
    n_weeks = len(df)
    effective_min_weeks = max(min_active_weeks, round(min_active_weeks_frac * n_weeks))

    # share_likelihood_metric is always the Hill-curve regressor; never changes.
    share_metric = config.share_likelihood_metric
    metric_prefix = f"$metric:{share_metric}$"

    # auxiliary_metric never becomes a regressor -- only decides the gate
    # (concentration/active weeks) and feeds the CSL prior. Always set (see
    # build_config); when the vehicle has no real exposure metric, the DS
    # points auxiliary_metric to the same value as share_likelihood_metric.
    aux_metric = config.auxiliary_metric
    aux_prefix = f"$metric:{aux_metric}$"
    aux_dfs: dict[str, pd.DataFrame] = {}

    def _stats_for(prefix: str, tail_of: dict[str, str]) -> dict[str, dict]:
        out = {}
        for key, tail in tail_of.items():
            slug = prefix + tail
            if slug in df.columns:
                s = df[slug]
                out[key] = {"total": float(s.sum()), "active": int((s > 0).sum())}
            else:
                out[key] = {"total": 0.0, "active": 0}
        return out

    def _hhi_of(stats_: dict[str, dict]) -> tuple[float, float, int]:
        total = sum(v["total"] for v in stats_.values())
        n_active = sum(1 for v in stats_.values() if v["total"] > 0)
        hhi = sum((v["total"] / total) ** 2 for v in stats_.values()) if total > 0 else 1.0
        return total, hhi, n_active

    for dim, all_slugs in config.vars_per_dim.items():
        slugs = [s for s in all_slugs if s.startswith(metric_prefix)]

        if not slugs:
            skipped_dims.append(dim)
            continue

        tail_of = {slug: slug[len(metric_prefix):] for slug in slugs}
        primary_stats: dict[str, dict] = _stats_for(metric_prefix, tail_of)
        aux_stats: dict[str, dict] = _stats_for(aux_prefix, tail_of)
        if sum(v["total"] for v in aux_stats.values()) == 0:
            raise ValueError(
                f"[{dim}] auxiliary_metric '{aux_metric}' has no real data for this "
                "dimension -- can't drive the share-likelihood gate. Fix the exposure "
                "data upstream, or set auxiliary_metric to the same value as "
                "share_likelihood_metric if this vehicle truly has no exposure metric."
            )

        # gate_stats decides keep/exclude; kept/excl stay with the investment
        # slugs (tail_of), which is what becomes the regressor.
        gate_stats = aux_stats
        cat_total, hhi, n_active = _hhi_of(gate_stats)

        if n_active < 2 or hhi > hhi_threshold:
            skipped_dims.append(dim)
            for slug, d in gate_stats.items():
                rows.append(_make_row(dim, slug, d, cat_total, n_weeks, hhi, rec="SKIP", keep=False, reason=f"dim SKIP ({aux_metric})", reason_code="dim_skip"))
            continue

        kept, excl = [], []
        for slug in tail_of:
            d = gate_stats[slug]
            pct = d["total"] / cat_total if cat_total > 0 else 0.0
            if d["total"] == 0:
                keep, reason, rc = False, f"no signal in {aux_metric}", "no_gate_signal"
            elif pct < min_spend_share:
                keep, reason, rc = False, f"pct {pct:.1%} < {min_spend_share:.0%} ({aux_metric})", "low_pct"
            elif d["active"] < effective_min_weeks:
                keep, reason, rc = False, f"only {d['active']} week(s) < {effective_min_weeks} ({aux_metric})", "low_weeks"
            elif primary_stats[slug]["total"] == 0:
                # Passed the exposure gate, but there's no investment for this
                # slug (missing/all-zero column in extraction) -- without it
                # there's no series to become a regressor.
                keep, reason, rc = False, "no investment (missing column)", "no_primary_col"
            else:
                keep, reason, rc = True, "", "kept"

            # Report on the same basis that decided it (investment or auxiliary).
            row = _make_row(dim, slug, d, cat_total, n_weeks, hhi, rec="DD", keep=keep, reason=reason, reason_code=rc)
            rows.append(row)
            if keep:
                kept.append(slug)
            elif primary_stats[slug]["total"] > 0:
                excl.append(slug)

        if kept:
            if len(excl) > 1:
                others_col = f"__others__{sanitize_dim_name(dim)}"
                df[others_col] = df[excl].sum(axis=1)
                kept.append(others_col)
                bucketed[dim] = excl

                configured_lower = set(config.lower_funnel_vars_per_dim.get(dim, []))
                excl_lower = [v for v in excl if v in configured_lower]
                # excl members no longer exist as standalone slugs.
                if dim in new_lower_funnel_vars_per_dim:
                    new_lower_funnel_vars_per_dim[dim] = [
                        v for v in new_lower_funnel_vars_per_dim[dim] if v not in excl
                    ]
                if excl_lower and len(excl_lower) == len(excl):
                    new_lower_funnel_vars_per_dim.setdefault(dim, []).append(others_col)
                    print(f"  [{dim}] {others_col}: all {len(excl)} bucketed members are "
                          f"configured lower funnel -> others inherits lower funnel too.")
                elif excl_lower:
                    print(f"  [WARNING] [{dim}] {others_col}: {len(excl_lower)}/{len(excl)} "
                          f"bucketed members are configured lower funnel, mixed with upper-"
                          f"funnel members -> no unambiguous classification, defaulting the "
                          f"whole aggregate to upper funnel (adstocked). See README Sec. 8.")
            new_vars_per_dim[dim] = kept

            aux_cols = {}
            for slug in new_vars_per_dim[dim]:
                if slug.startswith("__others__"):
                    members = bucketed.get(dim, [])
                    aux_slugs = [aux_prefix + m[len(metric_prefix):] for m in members]
                    present = [a for a in aux_slugs if a in df.columns]
                    aux_cols[slug] = df[present].sum(axis=1) if present else pd.Series(0.0, index=df.index)
                else:
                    aux_slug = aux_prefix + slug[len(metric_prefix):]
                    aux_cols[slug] = df[aux_slug] if aux_slug in df.columns else pd.Series(0.0, index=df.index)
            aux_dfs[dim] = pd.DataFrame(aux_cols, index=df.index)

    # Update spend_df in-place so downstream pipeline sees __others__ cols
    upgrade.spend_df = df

    spend_report = pd.DataFrame(rows)
    _print_diagnosis(spend_report, min_spend_share, hhi_threshold)

    new_config = DeepDiveConfig(
        dims=[d for d in config.dims if d in new_vars_per_dim],
        vars_per_dim=new_vars_per_dim,
        media_var=config.media_var,
        brand=config.brand,
        vehicle=config.vehicle,
        model_type=config.model_type,
        model_name=config.model_name,
        share_likelihood_metric=config.share_likelihood_metric,
        auxiliary_metric=config.auxiliary_metric,
        share_prior_scale=config.share_prior_scale,
        proxy_ct_tolerance=config.proxy_ct_tolerance,
        num_steps=config.num_steps,
        min_spend_share=min_spend_share,
        hhi_threshold=hhi_threshold,
        min_active_weeks=min_active_weeks,
        min_active_weeks_frac=min_active_weeks_frac,
        vehicle_spec=config.vehicle_spec,
        lower_funnel_vars_per_dim=new_lower_funnel_vars_per_dim,
        upper_funnel_adstock_effect_per_dim=dict(config.upper_funnel_adstock_effect_per_dim),
    )
    return new_config, DiagnosisResult(
        spend_report=spend_report,
        bucketed=bucketed,
        skipped_dims=skipped_dims,
        auxiliary_metric_dfs=aux_dfs or None,
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
    print(f"  DEEP DIVE DIAGNOSIS  (min_pct={min_pct:.0%}, HHI>{hhi_threshold}=SKIP)")
    print("─" * w)
    print(f"  {'Dimension':<28}  {'Rec':>5}  {'HHI':>5}  {'Total':>6}  {'Kept':>6}  {'Excl':>6}")
    print(f"  {'─' * 68}")
    for dim, grp in diag_df.groupby("dim", sort=False):
        info = grp[grp["rec"] == "INFO"]
        main = grp[grp["rec"] != "INFO"]
        if main.empty:
            print(f"  [!]  {dim:<26}  no share_likelihood_metric data -- dimension skipped")
        else:
            rec = main["rec"].iloc[0]
            hhi = main["hhi"].iloc[0]
            n_tot = len(main)
            n_kp = int(main["keep"].sum())
            n_ex = n_tot - n_kp
            flag = "[!]  " if rec == "SKIP" else "[ok] "
            print(f"  {flag}{dim:<26}  {rec:>5}  {hhi:>5.2f}  {n_tot:>6}  {n_kp:>6}  {n_ex:>6}")
            for _, row in main[~main["keep"]].iterrows():
                label = _slug_label(row["slug"])
                print(f"       ↳ {label:<24}  {row['pct_gate_dim']:>6.1%}  {row['reason']}")
            others_rows = main[main["slug"].str.startswith("__others__") & main["keep"]]
            for _, row in others_rows.iterrows():
                print(f"       → {'others':<24}  {row['pct_gate_dim']:>6.1%}  {row['reason']}")
        for _, row in info.iterrows():
            label = _slug_label(row["slug"])
            print(f"       ·  {label:<24}  {row['reason']}")
    print("─" * w)
    dd_df = diag_df[diag_df["rec"] != "INFO"]
    real_kept = dd_df[dd_df["keep"] & ~dd_df["slug"].str.startswith("__others__")]
    n_dd = dd_df[dd_df["rec"] == "DD"]["dim"].nunique()
    n_sk = dd_df[dd_df["rec"] == "SKIP"]["dim"].nunique()
    n_q = int(real_kept.shape[0])
    n_others = int(dd_df[dd_df["slug"].str.startswith("__others__") & dd_df["keep"]].shape[0])
    print(f"  Dimensions with DD: {n_dd}  |  SKIP: {n_sk}  |  Kept breakdowns: {n_q}  |  'Others' groups: {n_others}")
    print("─" * w)


def _make_row(dim, slug, d, cat_total, n_weeks, hhi, rec, keep, reason, reason_code="kept"):
    pct = d["total"] / cat_total if cat_total > 0 else 0.0
    return {
        "dim": dim,
        "slug": slug,
        "gate_total": d["total"],
        "pct_gate_dim": pct,
        "active_weeks": d["active"],
        "pct_active": d["active"] / n_weeks if n_weeks > 0 else 0.0,
        "hhi": round(hhi, 3),
        "rec": rec,
        "keep": keep,
        "reason": reason,
        "reason_code": reason_code,
    }
