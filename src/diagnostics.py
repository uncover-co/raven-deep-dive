from __future__ import annotations
from dataclasses import dataclass
import re
import pandas as pd


def sanitize_dim_name(name: str) -> str:
    """Slugify a dimension name for use as a DataFrame column or filename."""
    return re.sub(r"[^\w-]", "", name.lower().replace(" ", "-"))

from config import DeepDiveConfig, _get_template
from extraction import UpgradeResult


@dataclass
class DiagnosisResult:
    spend_report: pd.DataFrame       # per-var stats: share, HHI, active weeks, keep
    bucketed: dict[str, list[str]]  # dim -> variables bucketed into __others__
    skipped_dims: list[str]          # dims skipped (HHI too high or < 2 active)
    auxiliary_metric_dfs: dict[str, pd.DataFrame] | None = None  # dim -> df aligned to final kept cols (config.auxiliary_metric values), feeds ContributionShareLikelihood prior only
    bucketed_raw: dict[str, pd.DataFrame] | None = None  # dim -> long-form (variable, date, investment, auxiliary_metric) for each __others__ member, pre-aggregation. Audit-only, not fed to the model.


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
    bucket_notes: dict[str, str] = {}
    bucket_info: dict[str, dict] = {}
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
    bucketed_raw: dict[str, pd.DataFrame] = {}

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

                # Audit-only: original per-member series before the __others__
                # aggregation, so a human can trace what went into that bucket.
                # Never fed to the model -- see aux_dfs below for that.
                inv_orig = df[excl]
                aux_orig_cols = {
                    m: (df[aux_prefix + tail_of[m]] if aux_prefix + tail_of[m] in df.columns
                        else pd.Series(0.0, index=df.index))
                    for m in excl
                }
                aux_orig = pd.DataFrame(aux_orig_cols, index=df.index)
                inv_long = (
                    inv_orig.rename_axis("date").reset_index()
                    .melt(id_vars="date", var_name="variable", value_name="investment")
                )
                aux_long = (
                    aux_orig.rename_axis("date").reset_index()
                    .melt(id_vars="date", var_name="variable", value_name="auxiliary_metric")
                )
                bucketed_raw[dim] = inv_long.merge(aux_long, on=["date", "variable"], how="left")

                configured_lower = set(config.lower_funnel_vars_per_dim.get(dim, []))
                excl_lower = [v for v in excl if v in configured_lower]
                # excl members no longer exist as standalone slugs.
                if dim in new_lower_funnel_vars_per_dim:
                    new_lower_funnel_vars_per_dim[dim] = [
                        v for v in new_lower_funnel_vars_per_dim[dim] if v not in excl
                    ]
                # The bucket name is deterministic, so the DS may have declared it
                # in the client YAML ahead of time. That declaration wins over the
                # inference below -- don't claim a default that didn't happen.
                pre_declared = others_col in new_lower_funnel_vars_per_dim.get(dim, [])
                if pre_declared:
                    bucket_notes[dim] = "bucket declared as lower funnel in the client YAML."
                elif excl_lower and len(excl_lower) == len(excl):
                    new_lower_funnel_vars_per_dim.setdefault(dim, []).append(others_col)
                    bucket_notes[dim] = (
                        f"all {len(excl)} members were lower funnel, so the aggregate "
                        f"inherits lower."
                    )
                elif excl_lower:
                    bucket_notes[dim] = (
                        f"mixed members ({len(excl_lower)}/{len(excl)} lower funnel): no "
                        f"unambiguous class, so the aggregate stays upper."
                    )

                # What the DS needs to place this bucket: how much it weighs and
                # how its members were classified before being merged.
                bucket_share = (
                    sum(gate_stats[m]["total"] for m in excl) / cat_total
                    if cat_total > 0 else 0.0
                )
                bucket_info[dim] = {
                    "share": bucket_share,
                    "declared_lower": excl_lower,
                    "default_upper": [v for v in excl if v not in configured_lower],
                }
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
    _print_diagnosis(spend_report, min_spend_share, hhi_threshold, len(bucketed))
    _print_model_composition(
        new_vars_per_dim, new_lower_funnel_vars_per_dim, bucketed, bucket_notes,
        bucket_info,
    )

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
        bucketed_raw=bucketed_raw or None,
    )


def _print_model_composition(
    vars_per_dim: dict[str, list[str]],
    lower_per_dim: dict[str, list[str]],
    bucketed: dict[str, list[str]],
    bucket_notes: dict[str, str],
    bucket_info: dict[str, dict],
) -> None:
    """Show, per dimension, exactly which variables entered the model and how
    each one is classified (upper = adstocked, lower = immediate response).

    Funnel classification is only fully resolved here, after bucketing: a
    `__others__` column doesn't exist until diagnostics builds it, so its
    classification can't be reviewed in the client YAML beforehand. Printing
    it means the DS can override it in the notebook, before the fit, instead
    of editing the YAML and re-running diagnostics.
    """
    w = 75
    print("─" * w)
    print("  FINAL MODEL COMPOSITION  (upper = adstocked · lower = immediate response)")
    print("─" * w)
    # These names get pasted straight into override_funnel(), so they must be
    # complete -- _slug_label truncates at 24 chars and would silently eat a
    # trailing hyphen (e.g. "brand-auction--videoview-").
    def _label(s: str) -> str:
        if s.startswith("__others__"):
            return s
        return _slug_label(s, truncate=False)

    for dim, slugs in vars_per_dim.items():
        lower = set(lower_per_dim.get(dim, []))
        up = [s for s in slugs if s not in lower]
        lo = [s for s in slugs if s in lower]
        print(f"  [{dim}]  {len(slugs)} variable(s)")
        if up:
            print(f"       upper  {', '.join(_label(s) for s in up)}")
        if lo:
            print(f"       lower  {', '.join(_label(s) for s in lo)}")
        members = bucketed.get(dim)
        if not members:
            continue
        others = next((s for s in slugs if s.startswith("__others__")), None)
        if others is None:
            continue
        cls = "lower" if others in lower else "upper"
        info = bucket_info.get(dim, {})
        decl_lower = info.get("declared_lower", [])
        decl_upper = info.get("default_upper", list(members))

        print()
        print(f"       {others}")
        print(f"         current class      {cls}"
              f"{' (adstocked)' if cls == 'upper' else ' (immediate response)'}")
        print(f"         weight in dim      {info.get('share', 0.0):.1%} "
              f"of the gate metric")
        print(f"         aggregates {len(members)} variable(s), previously classified as:")
        print(f"           declared lower ({len(decl_lower)})")
        print(_wrap_members(decl_lower, _label))
        print(f"           undeclared, default upper ({len(decl_upper)})")
        print(_wrap_members(decl_upper, _label))
        note = bucket_notes.get(dim)
        if note:
            print(f"         note: {note}")
        # override_funnel, not raw dict surgery: appending to
        # lower_funnel_vars_per_dim alone leaves a dict adstock covering a
        # variable that is no longer upper funnel, and the fit then raises.
        lower_now = [s for s in slugs if s in lower]
        if cls == "upper":
            wanted = lower_now + [others]
            print(f"         to treat this bucket as lower funnel, before the fit:")
        else:
            wanted = [s for s in lower_now if s != others]
            print(f"         to treat this bucket as upper funnel, before the fit:")
        labels = ", ".join(f'"{_label(s)}"' for s in wanted)
        print(f"           override_funnel(config, \"{dim}\", lower=[{labels}])")
    print("─" * w)


def _wrap_members(members: list[str], label_fn) -> str:
    """One indented, wrapped block listing bucket members (dash when empty)."""
    import textwrap

    pad = " " * 13
    if not members:
        return pad + "—"
    return textwrap.fill(
        ", ".join(label_fn(m) for m in members),
        width=76, initial_indent=pad, subsequent_indent=pad,
        # slugs carry hyphens; breaking on them splits a name across lines
        break_on_hyphens=False, break_long_words=False,
    )


def _slug_label(slug: str, truncate: bool = True) -> str:
    """Extract short human-readable label from a full slug for display."""
    # Any $key:value segment can hold the label, not just $category (e.g.
    # state_template's $state:{value}) -- skip fixed framing/brand, take the
    # last remaining segment so a new filter type needs no special case.
    candidates = []
    for key, val in re.findall(r"\$([a-z_]+):([^$]+)", slug):
        if key in ("metric", "vehicle"):
            continue
        if key == "category":
            cat, _, v = val.partition(":")
            if cat == "brand":
                continue
            candidates.append(v)
        else:
            candidates.append(val)
    label = candidates[-1] if candidates else slug
    return label[:24] if truncate else label


def _print_diagnosis(
    diag_df: pd.DataFrame, min_pct: float, hhi_threshold: float, n_others: int = 0
) -> None:
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
    # n_others comes from `bucketed`: spend_report has one row per original
    # slug and never gains an __others__ row, so counting it here was always 0.
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


# ── Spend coverage: do the declared breakdowns add up to the vehicle? ────────

def _vehicle_level_filter(template: str) -> str:
    """Drop the breakdown segment from a slug template.

    `{value}` is not always the last segment (see `state_template` and
    tiktok's `campaign_category`), so the segment carrying it is removed by
    name, not by position. What's left is the same filter without the
    breakdown -- i.e. the whole vehicle.
    """
    return "".join(
        f"${seg}" for seg in template.split("$") if seg and "{value}" not in seg
    )


def _to_weekly(series: pd.Series) -> pd.Series:
    """Normalise to W-MON week ends so two sources can be compared by label."""
    s = series.copy()
    s.index = pd.DatetimeIndex(s.index).to_period("W-MON").end_time.normalize()
    return s.groupby(level=0).sum()


def check_spend_coverage(
    config: DeepDiveConfig,
    spend_df: pd.DataFrame,
    *,
    against: str = "both",
    workspace: str | None = None,
    upgrade=None,
    upgrade_spend_col: str | None = None,
    vehicle_filter: str | None = None,
    start_date=None,
    end_date=None,
    tolerance: float = 0.05,
    time_interval: str = "week",
    timezone: str = "America/Sao_Paulo",
    data_version: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reconcile the Deep Dive's breakdown spend against the vehicle's total.

    Optional, and meant to run BEFORE `run_diagnostics`: a breakdown value
    that exists upstream but is missing from `vehicle_specs.yaml` is never
    requested, so its spend stays out of the Deep Dive with no error and no
    `__others__`. The model then explains the whole `C_t` with less spend
    than the vehicle actually had, inflating the surviving sub-channels.

    Two references, because they answer different questions:

    ``workspace``
        The same workspace, queried without the breakdown segment. Answers
        "does my ``values:`` list cover everything the workspace has?".
        Needs `workspace`, `start_date`, `end_date`.
    ``upgrade``
        The main model's own investment series, from `input_data.parquet`.
        Answers "is the Deep Dive starting from the same money that produced
        `C_t`?" -- it catches a window or metric mismatch between the two
        stages, which the workspace check cannot see. Needs `upgrade` and
        `upgrade_spend_col`: the column in `input_data.parquet` that fed this
        vehicle's contribution, which only the DS can name.

    Everything is compared on the weeks the two series share, so a main model
    trained on a longer window than the Deep Dive doesn't read as a shortfall.
    `ref_total_full`/`dim_total_full` keep the unrestricted sums.

    Totals alone miss a series that sums right but is shaped wrong, so each
    row also carries the correlation and the largest weekly deviation -- and
    the series themselves land in `report.attrs["series"]`, for
    `plots.plot_spend_coverage()` to overlay them.

    Returns one row per (dim, reference).
    """
    valid = {"workspace", "upgrade", "both"}
    if against not in valid:
        raise ValueError(f"against must be one of {sorted(valid)}, got {against!r}.")
    wanted = ["workspace", "upgrade"] if against == "both" else [against]

    if "workspace" in wanted and not (workspace and start_date and end_date):
        raise ValueError(
            "against='workspace' needs workspace=, start_date= and end_date= "
            "(without the dates the reference would cover a different window "
            "than the breakdown, and the gap would be meaningless)."
        )

    upgrade_series = None
    if "upgrade" in wanted:
        if upgrade is None or upgrade_spend_col is None:
            raise ValueError(
                "against='upgrade' needs upgrade= and upgrade_spend_col= (the "
                "input_data.parquet column that fed this vehicle's contribution)."
            )
        inp = upgrade.input_df
        if inp is None:
            raise ValueError("upgrade.input_df is empty -- reload with a current loader.")
        if upgrade_spend_col not in inp.columns:
            raise ValueError(
                f"'{upgrade_spend_col}' is not a column of input_data.parquet. "
                f"Columns containing 'invest': "
                f"{[c for c in inp.columns if 'invest' in str(c).lower()][:10]}"
            )
        idx = pd.to_datetime(inp["timestamp"]) if "timestamp" in inp.columns else inp.index
        upgrade_series = _to_weekly(pd.Series(inp[upgrade_spend_col].values, index=idx))

    rows: list[dict] = []
    series: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}
    ws = None
    ws_totals: dict[str, pd.Series] = {}
    spend_metric = config.vehicle_spec.get("default_metric") or config.share_likelihood_metric
    spend_prefix = f"$metric:{spend_metric}$"

    for dim, slugs in config.vars_per_dim.items():
        cols = [
            s for s in slugs
            if (s.startswith(spend_prefix) or s.startswith("__others__"))
            and s in spend_df.columns
        ]
        bucket = f"__others__{sanitize_dim_name(dim)}"
        if bucket in spend_df.columns and bucket not in cols:
            cols.append(bucket)
        dim_series = _to_weekly(spend_df[cols].sum(axis=1))

        for ref in wanted:
            if ref == "workspace":
                filt = vehicle_filter
                if filt is None:
                    breakdown = config.vehicle_spec.get("breakdowns", {}).get(dim, {})
                    raw = _vehicle_level_filter(_get_template(config.vehicle_spec, breakdown))
                    try:
                        filt = raw.format(
                            metric=config.share_likelihood_metric,
                            vehicle=config.vehicle_spec.get("vehicle_slug", ""),
                            brand=config.brand,
                            category=breakdown.get("category", ""),
                        )
                    except KeyError as e:
                        raise ValueError(
                            f"[{dim}] the vehicle template needs a placeholder this "
                            f"check doesn't know how to fill: {e}. It only supplies "
                            f"metric/vehicle/brand/category -- pass the whole filter "
                            f"explicitly with vehicle_filter=."
                        ) from e
                if ws is None:
                    import ducks

                    ws = ducks.workspace(workspace)
                if filt not in ws_totals:
                    df = ws.build_modelling_dataset(
                        [filt], start_date=start_date, end_date=end_date,
                        time_interval=time_interval, timezone=timezone,
                        zero_fill=True, data_version=data_version,
                    )
                    ws_totals[filt] = _to_weekly(df.sum(axis=1))
                ref_series = ws_totals[filt]
            else:
                ref_series = upgrade_series

            series[(dim, ref)] = (ref_series, dim_series)
            # Compare only where both exist: the main model's window is often
            # wider than the Deep Dive's, and summing the whole of each would
            # report that difference as missing spend.
            common = ref_series.index.intersection(dim_series.index)
            a, b = ref_series.reindex(common), dim_series.reindex(common)
            ref_total, dim_total = float(a.sum()), float(b.sum())
            gap = ref_total - dim_total
            # No reference at all is not perfect coverage: a 0.0 here read as
            # "nothing missing", and `corr` is NaN in that case, so nothing
            # warned. Undefined unless both sides are empty.
            # A disjoint window reindexes both to nothing regardless of the
            # original totals -- always undefined, not genuine zero coverage.
            if common.empty:
                gap_pct = float("nan")
            elif ref_total:
                gap_pct = gap / ref_total
            else:
                gap_pct = 0.0 if not dim_total else float("nan")
            rows.append({
                "dim": dim,
                "reference": ref,
                "ref_total": ref_total,
                "dim_total": dim_total,
                "gap": gap,
                "gap_pct": gap_pct,
                "weeks_compared": len(common),
                "corr": float(a.corr(b)) if len(common) > 2 else float("nan"),
                "max_week_dev_pct": (
                    float((a - b).abs().max() / a.mean()) if len(common) and a.mean() else float("nan")
                ),
                "ref_total_full": float(ref_series.sum()),
                "dim_total_full": float(dim_series.sum()),
            })

    report = pd.DataFrame(rows)
    report.attrs["series"] = series
    if verbose and not report.empty:
        for _, r in report.iterrows():
            flags = []
            if pd.isna(r.gap_pct):
                flags.append("no reference spend, but the breakdowns have some")
            elif abs(r.gap_pct) > tolerance:
                flags.append(f"gap {r.gap_pct:.1%}")
            if pd.notna(r["corr"]) and r["corr"] < 0.95:
                flags.append(f"corr {r['corr']:.2f}")
            shortest = min(
                len(series[(r.dim, r.reference)][0]), len(series[(r.dim, r.reference)][1])
            )
            if shortest and r.weeks_compared < 0.8 * shortest:
                flags.append(
                    f"only {r.weeks_compared} of {shortest} weeks overlap "
                    f"(different week anchoring?)"
                )
            if flags:
                print(
                    f"  [WARNING] [{r.dim}] vs {r.reference}: {', '.join(flags)} "
                    f"(reference {r.ref_total:,.0f} · breakdowns {r.dim_total:,.0f})"
                )
    return report
