from __future__ import annotations
import re as _re_plots
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pipeline import _align_to

# ── Uncover dark theme ────────────────────────────────────────────────────────

UNCOVER_COLORS: list[str] = [
    "#00CC44",  # green (primary)
    "#FFC107",  # yellow
    "#CC0000",  # red
    "#7700CC",  # purple
    "#00AAFF",  # blue
    "#FF6600",  # orange
    "#00CCCC",  # cyan
    "#8A8A8A",  # gray
]

_LAYOUT_DEFAULTS = dict(
    plot_bgcolor="#141414",
    paper_bgcolor="#1E1E1E",
    font=dict(color="#E0E0E0", family="Inter, Arial, sans-serif", size=12),
    colorway=UNCOVER_COLORS,
    legend=dict(bgcolor="#1E1E1E", bordercolor="#2A2A2A", borderwidth=1),
    xaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A"),
    yaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A"),
    height=560,
    width=1100,
    margin=dict(l=60, r=40, t=60, b=60),
)

UNCOVER_DARK_TEMPLATE = go.layout.Template(layout=go.Layout(**_LAYOUT_DEFAULTS))


def _styled(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_LAYOUT_DEFAULTS)
    for ax in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
        fig.update_layout(**{ax: dict(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A")})
    return fig


# ── Plot 1: Contribution share vs Spend share (grouped bar) ──────────────────

def plot_contributions(result) -> go.Figure:
    """Grouped bar: contribution share vs spend share per dimension x item."""
    rows_data = []
    for dim in result.config.dims:
        sh_m = result.shares_model.get(dim, pd.Series(dtype=float))
        sh_s = result.shares_spend.get(dim, pd.Series(dtype=float))
        for item in sh_m.index:
            rows_data.append({
                "label": f"{dim} — {item}",
                "contrib_share": float(sh_m[item]),
                "spend_share": float(sh_s.get(item, 0.0)),
            })
    df = pd.DataFrame(rows_data)
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["label"], y=df["contrib_share"],
        name="Contrib Share", marker_color=UNCOVER_COLORS[0],
    ))
    fig.add_trace(go.Bar(
        x=df["label"], y=df["spend_share"],
        name="Spend Share", marker_color=UNCOVER_COLORS[1], opacity=0.75,
    ))
    fig.update_layout(
        barmode="group",
        title="Contribution Share vs Spend Share",
        xaxis_title="",
        yaxis=dict(tickformat=".0%", title="Share"),
    )
    return _styled(fig)


# ── Plot 2: Weekly contributions stacked area ─────────────────────────────────

def plot_weekly(result, dim: str) -> go.Figure:
    """Stacked area: weekly contributions per sub-channel + C_t total overlay."""
    contribs = result.contribs.get(dim)
    if contribs is None:
        raise ValueError(f"dim '{dim}' not in result.contribs. Available: {list(result.contribs)}")

    fig = go.Figure()
    for i, col in enumerate(contribs.columns):
        fig.add_trace(go.Scatter(
            x=contribs.index,
            y=contribs[col],
            name=col,
            stackgroup="one",
            mode="none",
            fillcolor=UNCOVER_COLORS[i % len(UNCOVER_COLORS)],
            line=dict(width=0),
        ))

    ct = _align_to(result.eletro_contrib, contribs.index)
    fig.add_trace(go.Scatter(
        x=ct.index, y=ct.values,
        name="C_t total (Stan)",
        mode="lines",
        line=dict(color="#FFFFFF", width=2, dash="dot"),
    ))

    fig.update_layout(
        title=f"Contribs Semanais — {dim}",
        xaxis_title="Semana",
        yaxis_title="Contribuição",
    )
    return _styled(fig)


# ── Plot 3: Hill saturation curves ────────────────────────────────────────────

def plot_saturation_curves(result, dim: str) -> go.Figure:
    """Hill saturation curves per sub-channel with operating point marked."""
    from pipeline import extract_hill_params  # avoid circular import at module level

    model = result.models.get(dim)
    if model is None:
        raise ValueError(f"No fitted model for dim '{dim}'. Available: {list(result.models)}")

    variaveis = list(result.contribs[dim].columns)
    features_raw = result.features_raw.get(dim)
    col_maxes = result.col_maxes.get(dim)
    params = extract_hill_params(model, variaveis, features_raw, col_maxes)

    x_range = np.linspace(0.0, 1.0, 200)
    fig = go.Figure()

    for i, var in enumerate(variaveis):
        if var not in params.index:
            continue
        row = params.loc[var]
        hm = row.get("half_max_norm")
        sl = row.get("slope")
        me = row.get("max_effect") or 1.0
        if hm is None or sl is None or np.isnan(hm) or np.isnan(sl):
            continue

        y_curve = me * x_range**sl / (hm**sl + x_range**sl + 1e-12)
        color = UNCOVER_COLORS[i % len(UNCOVER_COLORS)]

        fig.add_trace(go.Scatter(
            x=x_range, y=y_curve, name=var,
            line=dict(color=color, width=2),
        ))

        mean_spend = row.get("mean_spend")
        raw_max = float(col_maxes[var]) if col_maxes is not None and var in col_maxes else None
        if mean_spend and raw_max and raw_max > 0:
            x_op = mean_spend / raw_max
            y_op = me * x_op**sl / (hm**sl + x_op**sl + 1e-12)
            fig.add_trace(go.Scatter(
                x=[x_op], y=[y_op],
                mode="markers",
                marker=dict(size=10, color=color, symbol="circle"),
                showlegend=False,
                name=f"{var} (op)",
            ))

    fig.update_layout(
        title=f"Curvas de Saturação Hill — {dim}",
        xaxis_title="Spend Normalizado [0–1]",
        yaxis_title="Contribuição",
    )
    return _styled(fig)


# ── Plot 4: ROAS index heatmap ────────────────────────────────────────────────

def plot_roas_index(result) -> go.Figure:
    """Heatmap: contrib_share / spend_share per dimension x item."""
    dims, items_all, z = [], [], []

    for dim in result.config.dims:
        sh_m = result.shares_model.get(dim, pd.Series(dtype=float))
        sh_s = result.shares_spend.get(dim, pd.Series(dtype=float))
        common = sh_m.index.intersection(sh_s.index)
        for item in common:
            sv = float(sh_s[item])
            ratio = float(sh_m[item]) / sv if sv > 0 else float("nan")
            dims.append(dim)
            items_all.append(item)
            z.append(ratio)

    if not dims:
        return go.Figure()

    df = pd.DataFrame({"dim": dims, "item": items_all, "roas_index": z})
    pivot = df.pivot(index="item", columns="dim", values="roas_index")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, "#CC0000"],
            [0.5, "#FFC107"],
            [1.0, "#00CC44"],
        ],
        zmid=1.0,
        colorbar=dict(title="ROAS Index", tickfont=dict(color="#E0E0E0")),
        text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="ROAS Index (contrib_share / spend_share)",
        xaxis=dict(title="Dimensão"),
        yaxis=dict(title="Item"),
    )
    return _styled(fig)


# ── Helpers: label + summary ──────────────────────────────────────────────────

def _clean_label(s: str) -> str:
    if not s.startswith("$"):
        return s
    parts = s.split("$")
    return parts[-1].split(":")[-1] if parts else s


def _print_breakdown_summary(result, dim: str) -> None:
    contribs = result.contribs.get(dim)
    if contribs is None:
        return
    anchor = _align_to(result.eletro_contrib, contribs.index)
    total_anchor = float(anchor.sum())
    total_dim = float(contribs.sum(axis=1).sum())
    proxy_r = result.proxy_ratios.get(dim, float("nan"))
    csl_d = result.csl_devs.get(dim, float("nan"))

    print(f"\n{'='*66}")
    print(f"  {dim}  (proxy_ratio={proxy_r:.3f}  CSL_max_dev={csl_d:.3f})")
    print(f"{'='*66}")
    print(f"  Âncora Stan     : {total_anchor:>14,.0f}")
    print(f"  Soma sub-canais : {total_dim:>14,.0f}  (deve ≈ âncora)")
    print(f"{'─'*66}")
    print(f"  {'Sub-canal':<30} {'Absoluto':>12}  {'% âncora':>9}")
    print(f"{'─'*66}")
    for col in contribs.columns:
        label = _clean_label(col)
        val = float(contribs[col].sum())
        pct = val / total_anchor * 100 if total_anchor else float("nan")
        print(f"  {label:<30} {val:>12,.0f}  {pct:>8.1f}%")
    print(f"{'─'*66}")
    print(f"  {'TOTAL':<30} {total_dim:>12,.0f}  {total_dim / total_anchor * 100:>8.1f}%")
    print(f"{'='*66}")


# ── Plot 5: Breakdown total (horizontal stacked bar) ──────────────────────────

def plot_breakdown_total(result, dim: str) -> go.Figure:
    """Horizontal stacked bar: total contribution per sub-channel vs Stan anchor."""
    contribs = result.contribs.get(dim)
    if contribs is None:
        raise ValueError(f"dim '{dim}' not in result.contribs. Available: {list(result.contribs)}")

    anchor = _align_to(result.eletro_contrib, contribs.index)
    total_anchor = float(anchor.sum())
    totals = contribs.sum()

    fig = go.Figure()
    cumulative = 0.0
    for i, col in enumerate(contribs.columns):
        val = float(totals[col])
        label = _clean_label(col)
        pct = val / total_anchor * 100 if total_anchor else 0.0
        fig.add_trace(go.Bar(
            x=[val],
            y=[""],
            name=f"{label} ({pct:.1f}%)",
            orientation="h",
            marker_color=UNCOVER_COLORS[i % len(UNCOVER_COLORS)],
            base=cumulative,
            hovertemplate=f"<b>{label}</b><br>Contribuição: %{{x:,.0f}}<br>{pct:.1f}% da âncora<extra></extra>",
        ))
        cumulative += val

    fig.add_vline(
        x=total_anchor,
        line_color="#FFFFFF",
        line_dash="dot",
        line_width=2,
        annotation_text="Âncora Stan",
        annotation_font_color="#FFFFFF",
        annotation_position="top right",
    )
    fig.update_layout(
        barmode="stack",
        title=f"Contribs Total do Período — {dim}",
        xaxis_title="Contribuição",
        height=280,
    )
    return _styled(fig)


# ── analyze_deepdive: relatório por dimensão (single client) ─────────────────

def analyze_deepdive(result, title_prefix: str = "") -> dict[str, tuple]:
    """Relatório por dimensão de quebra: summary + total + semanal.

    Para cada dim em result.config.dims imprime tabela de shares e mostra
    dois gráficos: barra total do período e área empilhada semanal.

    Returns {dim: (fig_total, fig_weekly)}.
    """
    figs: dict[str, tuple] = {}
    prefix = f"{title_prefix} — " if title_prefix else ""
    for dim in result.config.dims:
        if result.contribs.get(dim) is None:
            continue
        _print_breakdown_summary(result, dim)
        fig_total = plot_breakdown_total(result, dim)
        fig_total.update_layout(title=f"{prefix}Contribs Total — {dim}")
        fig_weekly = plot_weekly(result, dim)
        fig_weekly.update_layout(title=f"{prefix}Contribs Semanais — {dim}")
        fig_total.show()
        fig_weekly.show()
        figs[dim] = (fig_total, fig_weekly)
    return figs


# ── analyze_batch: relatório multi-cliente por dimensão/rollup ────────────────

_ROLLUP_LABEL = {
    "grupo":    "Grupo",
    "vertical": "Vertical",
    "tipo":     "Tipo",
    "estado":   "Estado",
    "praca":    "Praça",
    "raw":      "",
}


def _rollup_order_for_dim(
    dim: str,
    vehicle_spec: dict,
    df_dim: "pd.DataFrame",
) -> list[str]:
    """Derive rollup display order from vehicle_spec; fallback = most-items-first."""
    bd_spec = vehicle_spec.get("breakdowns", {}).get(dim, {})
    rollup_specs = bd_spec.get("rollups", [])
    if rollup_specs:
        ordered = [r["level"] for r in rollup_specs]
    else:
        # No spec: sort by unique item count desc (more granular levels first)
        counts = df_dim.groupby("rollup")["item"].nunique()
        ordered = counts.sort_values(ascending=False).index.tolist()

    present = set(df_dim["rollup"].unique())
    result = [r for r in ordered if r in present]
    # Append any levels present in data but absent from spec (shouldn't normally happen)
    result += sorted(present - set(ordered))
    return result


def _short_label(s: str, maxlen: int = 30) -> str:
    """Strip $metric slug prefix and truncate."""
    if s.startswith("$"):
        s = s.split("$")[-1].split(":")[-1]
    return s[:maxlen]


def _row_height(n_items: int) -> int:
    return max(180, n_items * 24 + 60)


def plot_batch_dim(
    dim: str,
    df_dim: pd.DataFrame,
    vehicle_spec: dict | None = None,
) -> go.Figure:
    """One figure per dimension: rows = rollup levels, cols = [Share | ROAS Index].

    df_dim: slice of consolidate_results() for this dim.
    vehicle_spec: used to derive rollup display order; falls back to item-count sort.
    """
    clients       = sorted(df_dim["client"].unique())
    rollup_levels = _rollup_order_for_dim(dim, vehicle_spec or {}, df_dim)

    n_rows   = len(rollup_levels)
    row_h    = [_row_height(len(df_dim[df_dim["rollup"] == rl]["item"].unique()))
                for rl in rollup_levels]
    total_h  = sum(row_h) + 80 * n_rows + 100

    # Subplot titles: left col = rollup label + "Share", right = "ROAS Index"
    subplot_titles: list[str] = []
    for rl in rollup_levels:
        lbl = _ROLLUP_LABEL.get(rl, rl)
        subplot_titles += [f"{lbl} — Share" if lbl else "Share",
                           f"{lbl} — ROAS Index" if lbl else "ROAS Index"]

    row_heights_norm = [h / total_h for h in row_h]

    fig = make_subplots(
        rows=n_rows,
        cols=2,
        shared_xaxes=False,
        shared_yaxes=False,
        row_heights=row_heights_norm,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
        vertical_spacing=0.10,
    )

    for ri, rollup in enumerate(rollup_levels, start=1):
        df_r = df_dim[df_dim["rollup"] == rollup].copy()
        # Sort items by mean share_model across clients (descending)
        item_order = (
            df_r.groupby("item")["share_model"].mean()
            .sort_values(ascending=True)  # ascending for horizontal bar (bottom=lowest)
            .index.tolist()
        )
        y_labels = [_short_label(it) for it in item_order]

        for ci, client in enumerate(clients):
            df_c  = df_r[df_r["client"] == client].set_index("item")
            color = UNCOVER_COLORS[ci % len(UNCOVER_COLORS)]
            show_legend = ri == 1  # legend only for first row

            sm  = [float(df_c.loc[it, "share_model"]) if it in df_c.index else 0.0 for it in item_order]
            ss  = [float(df_c.loc[it, "share_spend"])  if it in df_c.index else 0.0 for it in item_order]
            roi = [float(df_c.loc[it, "roas_index"])   if it in df_c.index else float("nan") for it in item_order]

            # ── Share: model (solid) + spend (hatched) ──────────────────────
            fig.add_trace(go.Bar(
                x=sm, y=y_labels, orientation="h",
                name=client,
                legendgroup=client,
                showlegend=show_legend,
                marker=dict(color=color, opacity=0.9),
                hovertemplate="%{y}<br>Share Modelo: %{x:.1%}<extra>" + client + "</extra>",
            ), row=ri, col=1)

            fig.add_trace(go.Bar(
                x=ss, y=y_labels, orientation="h",
                name=f"{client} (spend)",
                legendgroup=f"{client}_spend",
                showlegend=show_legend,
                marker=dict(
                    color=color,
                    opacity=0.35,
                    pattern=dict(shape="/", fgcolor=color, bgcolor="rgba(0,0,0,0)"),
                ),
                hovertemplate="%{y}<br>Share Spend: %{x:.1%}<extra>" + client + " spend</extra>",
            ), row=ri, col=1)

            # ── ROAS index ───────────────────────────────────────────────────
            bar_colors = [
                "#CC0000" if (not np.isnan(v) and v < 0.85) else
                "#FFC107" if (not np.isnan(v) and v < 1.15) else
                "#00CC44"
                for v in roi
            ]
            fig.add_trace(go.Bar(
                x=roi, y=y_labels, orientation="h",
                name=client,
                legendgroup=client,
                showlegend=False,
                marker=dict(color=bar_colors, opacity=0.85),
                hovertemplate="%{y}<br>ROAS Index: %{x:.3f}<extra>" + client + "</extra>",
            ), row=ri, col=2)

        # vline at 1.0 (parity) in ROAS subplot
        # Plotly add_vline doesn't target specific subplots cleanly — use shape
        fig.add_shape(
            type="line", x0=1.0, x1=1.0, y0=-0.5, y1=len(item_order) - 0.5,
            line=dict(color="#FFFFFF", dash="dot", width=1.5),
            row=ri, col=2,
        )

        # Format X axes as percentages for share
        fig.update_xaxes(tickformat=".0%", row=ri, col=1,
                         gridcolor="#2A2A2A", zerolinecolor="#2A2A2A")
        fig.update_xaxes(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A", row=ri, col=2)
        fig.update_yaxes(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A", row=ri, col=1)
        fig.update_yaxes(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A", row=ri, col=2)

    fig.update_layout(
        **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k not in ("height", "width", "legend")},
        title=dict(text=f"<b>{dim}</b> — Share Modelo vs Spend  |  ROAS Index",
                   font=dict(size=15, color="#E0E0E0")),
        barmode="overlay",
        height=total_h,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left",   x=0,
            bgcolor="#1E1E1E", bordercolor="#2A2A2A", borderwidth=1,
        ),
    )
    return fig


def _print_batch_dim_summary(
    dim: str,
    df_dim: pd.DataFrame,
    vehicle_spec: dict | None = None,
) -> None:
    """Print tabular summary for one dimension across all clients."""
    rollup_levels = _rollup_order_for_dim(dim, vehicle_spec or {}, df_dim)

    for rollup in rollup_levels:
        df_r   = df_dim[df_dim["rollup"] == rollup]
        clients = sorted(df_r["client"].unique())
        items   = df_r.groupby("item")["share_model"].mean().sort_values(ascending=False).index.tolist()
        rl_lbl  = _ROLLUP_LABEL.get(rollup, rollup)

        print(f"\n{'═'*80}")
        title = f"  {dim}{'/' + rl_lbl if rl_lbl else ''}  ×  {', '.join(clients)}"
        print(title)
        print(f"{'═'*80}")

        # Header
        hdr = f"  {'Item':<28}"
        for c in clients:
            hdr += f"  {c[:10]:>10} mod  {c[:10]:>10} spd  {'ROAS':>6}"
        print(hdr)
        print(f"  {'─'*78}")

        for item in items:
            row_str = f"  {_short_label(item):<28}"
            for c in clients:
                sub = df_r[(df_r["client"] == c) & (df_r["item"] == item)]
                if sub.empty:
                    row_str += f"  {'n/a':>13}  {'n/a':>13}  {'n/a':>6}"
                else:
                    sm  = sub["share_model"].iloc[0]
                    ss  = sub["share_spend"].iloc[0]
                    roi = sub["roas_index"].iloc[0]
                    roi_s = f"{roi:.2f}" if pd.notna(roi) else "n/a"
                    row_str += f"  {sm:>12.1%}  {ss:>12.1%}  {roi_s:>6}"
            print(row_str)

        print(f"  {'─'*78}")
        for c in clients:
            proxy = df_dim[df_dim["client"] == c]["proxy_ratio"].iloc[0] if not df_dim[df_dim["client"] == c].empty else float("nan")
            print(f"  [{c}] proxy_ratio={proxy:.3f}")
    print(f"{'═'*80}")


def analyze_batch(
    all_results: dict,
    df_meta: pd.DataFrame,
    dims: list[str] | None = None,
    show: bool = True,
    vehicle_spec_override: dict | None = None,
) -> dict[str, go.Figure]:
    """Relatório multi-cliente por dimensão: tabela + gráfico share/ROAS.

    Args:
        all_results: dict[client_name, DDResult].
        df_meta: output of consolidate_results() — long-form with rollup column.
        dims: subset of dimensions to plot; None = all.
        show: call fig.show() in notebook.
        vehicle_spec_override: explicit vehicle spec for rollup ordering; overrides
            what's stored in all_results (use when old batch produced empty vehicle_spec).

    Returns:
        {dim: fig}
    """
    # Derive vehicle_spec: prefer override, then result config, then empty
    vehicle_spec: dict = vehicle_spec_override or {}
    if not vehicle_spec and all_results:
        first = next(iter(all_results.values()))
        vehicle_spec = getattr(getattr(first, "config", None), "vehicle_spec", {})

    available_dims = dims or df_meta["dim"].unique().tolist()
    figs: dict[str, go.Figure] = {}

    for dim in available_dims:
        df_dim = df_meta[df_meta["dim"] == dim]
        if df_dim.empty:
            continue
        _print_batch_dim_summary(dim, df_dim, vehicle_spec)
        fig = plot_batch_dim(dim, df_dim, vehicle_spec)
        if show:
            fig.show()
        figs[dim] = fig

    return figs


# ── Tree (sunburst / treemap) visualization ────────────────────────────────────

def _slug_val(slug: str, category: str) -> str:
    """Extract clean dimension value from raw slug."""
    m = _re_plots.search(rf"\$category:{_re_plots.escape(category)}:([^$]+)", slug)
    if m:
        return m.group(1)
    if category == "state":
        m = _re_plots.search(r"\$state:([^$]+)", slug)
        if m:
            return m.group(1)
    return slug.split(":")[-1]


# Diverging RYG scale centered at 1.0 (ROAS = 1 → neutral yellow)
_ROAS_COLORSCALE = [
    [0.00, "#CC0000"],
    [0.35, "#FF6600"],
    [0.50, "#FFEB3B"],
    [0.65, "#66BB6A"],
    [1.00, "#00CC44"],
]
_ROAS_CMIN, _ROAS_CMAX = 0.6, 1.4


def _flat_map_sunburst(
    shares_model: "pd.Series",
    shares_spend: "pd.Series",
    category: str,
    flat_map: dict[str, str],
) -> dict:
    """Build sunburst node data for a 2-level flat {leaf: parent} hierarchy."""
    leaf_m: dict[str, float] = {}
    leaf_s: dict[str, float] = {}
    for slug, sm in shares_model.items():
        member = _slug_val(slug, category)
        leaf_m[member] = leaf_m.get(member, 0.0) + float(sm)
        leaf_s[member] = leaf_s.get(member, 0.0) + float(shares_spend.get(slug, 0.0))

    parent_m: dict[str, float] = {}
    parent_s: dict[str, float] = {}
    for leaf, sm in leaf_m.items():
        p = flat_map.get(leaf, leaf)
        parent_m[p] = parent_m.get(p, 0.0) + sm
        parent_s[p] = parent_s.get(p, 0.0) + leaf_s.get(leaf, 0.0)

    ids, labels, parents, vals_m, vals_s = [], [], [], [], []
    for p, sm in parent_m.items():
        ids.append(f"p:{p}"); labels.append(p)
        parents.append(""); vals_m.append(sm); vals_s.append(parent_s[p])
    for leaf, sm in leaf_m.items():
        p = flat_map.get(leaf, leaf)
        ids.append(f"l:{leaf}"); labels.append(leaf)
        parents.append(f"p:{p}"); vals_m.append(sm); vals_s.append(leaf_s[leaf])

    return dict(ids=ids, labels=labels, parents=parents, values_m=vals_m, values_s=vals_s)


def _groups_sunburst(
    shares_model: "pd.Series",
    shares_spend: "pd.Series",
    category: str,
    groups_spec: dict,
    members_key: str,
) -> dict:
    """Build sunburst node data from a groups hierarchy.

    Intermediate levels are auto-derived from group attributes sorted by cardinality
    (fewest unique values → closest to root). Tree:
        attr_low_card → attr_high_card → group_name → member
    """
    # Discover scalar attrs and sort by cardinality (ascending = closer to root)
    attr_vals: dict[str, set] = {}
    for gspec in groups_spec.values():
        for k, v in gspec.items():
            if k != members_key and not isinstance(v, list) and v is not None:
                attr_vals.setdefault(k, set()).add(v)
    sorted_attrs = sorted(attr_vals.keys(), key=lambda a: len(attr_vals[a]))
    # e.g. ["tipo", "vertical"] for Ambiente (tipo=2, vertical=6)

    # Raw member shares
    leaf_m: dict[str, float] = {}
    leaf_s: dict[str, float] = {}
    for slug, sm in shares_model.items():
        member = _slug_val(slug, category)
        leaf_m[member] = leaf_m.get(member, 0.0) + float(sm)
        leaf_s[member] = leaf_s.get(member, 0.0) + float(shares_spend.get(slug, 0.0))

    # group_name → (attr0_val, attr1_val, ...) path tuple
    def gpath(gname: str) -> tuple:
        gspec = groups_spec.get(gname, {})
        return tuple(gspec.get(a, f"?{a}") for a in sorted_attrs)

    # Aggregate shares bottom-up through each path level
    node_m: dict[tuple, float] = {}
    node_s: dict[tuple, float] = {}
    node_parent: dict[tuple, tuple] = {}

    for gname, gspec in groups_spec.items():
        path = gpath(gname) + (gname,)
        for member in gspec.get(members_key, []):
            sm = leaf_m.get(member, 0.0)
            ss = leaf_s.get(member, 0.0)

            # Leaf node
            mpath = path + (member,)
            node_m[mpath] = node_m.get(mpath, 0.0) + sm
            node_s[mpath] = node_s.get(mpath, 0.0) + ss
            node_parent[mpath] = path

            # Aggregate up every prefix level
            for depth in range(len(path), 0, -1):
                prefix = path[:depth]
                parent = path[:depth - 1]
                node_m[prefix] = node_m.get(prefix, 0.0) + sm
                node_s[prefix] = node_s.get(prefix, 0.0) + ss
                node_parent[prefix] = parent

    ids, labels, parents, vals_m, vals_s = [], [], [], [], []
    for path, sm in node_m.items():
        node_id = "/".join(str(p) for p in path)
        parent_path = node_parent.get(path, ())
        parent_id = "/".join(str(p) for p in parent_path)
        ids.append(node_id)
        labels.append(str(path[-1]))
        parents.append(parent_id)
        vals_m.append(sm)
        vals_s.append(node_s.get(path, 0.0))

    return dict(ids=ids, labels=labels, parents=parents, values_m=vals_m, values_s=vals_s)


def _roas_colors(vals_m: list[float], vals_s: list[float]) -> list[float]:
    """Compute per-node ROAS index for sunburst coloring."""
    total_m = sum(vals_m) or 1.0
    total_s = sum(vals_s) or 1.0
    colors = []
    for sm, ss in zip(vals_m, vals_s):
        sm_n = sm / total_m
        ss_n = ss / total_s
        colors.append(sm_n / ss_n if ss_n > 1e-9 else 1.0)
    return colors


def plot_tree_dim(
    dim: str,
    all_results: dict,
    vehicle_spec: dict,
    chart_type: str = "sunburst",
    show: bool = False,
) -> go.Figure | None:
    """Tree visualization for one dimension across all clients.

    One sunburst/treemap per client, side-by-side.
    Size = share_model. Color = ROAS index (red < 1 → yellow ≈ 1 → green > 1).

    chart_type: "sunburst" | "treemap" | "icicle"
    """
    clients = sorted(all_results.keys())
    n = len(clients)
    if n == 0:
        return None

    bd_spec = vehicle_spec.get("breakdowns", {}).get(dim, {})
    category = bd_spec.get("category", "")
    rollup_specs = bd_spec.get("rollups", [])
    hierarchy = vehicle_spec.get("hierarchy", {})

    # Pick best hierarchy: groups (full tree) > flat map > nothing
    groups_rspec = next(
        (r for r in rollup_specs if "groups" in r and "attr" not in r), None
    )
    map_rspec = next((r for r in rollup_specs if "map" in r), None)

    if not groups_rspec and not map_rspec:
        return None

    # Each spec dict must be a separate object — [d] * n shares the same ref
    specs_grid = [[{"type": "domain"} for _ in range(n)]]
    fig = make_subplots(
        rows=1, cols=n,
        specs=specs_grid,
        subplot_titles=clients,
        horizontal_spacing=0.04,
    )

    for ci, client in enumerate(clients, start=1):
        result = all_results[client]
        sh_m = result.shares_model.get(dim, pd.Series(dtype=float))
        sh_s = result.shares_spend.get(dim, pd.Series(dtype=float))
        if sh_m.empty:
            continue

        if groups_rspec:
            groups_data = hierarchy.get(groups_rspec["groups"], {})
            data = _groups_sunburst(sh_m, sh_s, category, groups_data,
                                    groups_rspec.get("members_key", "values"))
        else:
            flat_map = hierarchy.get(map_rspec["map"], {})
            data = _flat_map_sunburst(sh_m, sh_s, category, flat_map)

        colors = _roas_colors(data["values_m"], data["values_s"])
        show_scale = ci == n

        marker_kwargs: dict = dict(
            colors=colors,
            colorscale=_ROAS_COLORSCALE,
            cmin=_ROAS_CMIN,
            cmax=_ROAS_CMAX,
            showscale=show_scale,
        )
        if show_scale:
            marker_kwargs["colorbar"] = dict(
                title=dict(text="ROAS", side="right"),
                thickness=10,
                len=0.6,
                tickvals=[0.6, 0.8, 1.0, 1.2, 1.4],
                ticktext=["0.6", "0.8", "1.0", "1.2", "1.4"],
            )

        common = dict(
            ids=data["ids"],
            labels=data["labels"],
            parents=data["parents"],
            values=data["values_m"],
            branchvalues="total",
            marker=marker_kwargs,
            name=client,
        )

        if chart_type == "sunburst":
            trace = go.Sunburst(
                **common,
                textinfo="label+percent root",
                insidetextorientation="radial",
            )
        elif chart_type == "treemap":
            trace = go.Treemap(
                **common,
                textinfo="label+percent root",
                tiling=dict(packing="squarify"),
            )
        else:  # icicle
            trace = go.Icicle(
                **common,
                textinfo="label+percent root",
            )

        fig.add_trace(trace, row=1, col=ci)

    fig.update_layout(
        **{k: v for k, v in _LAYOUT_DEFAULTS.items()
           if k not in ("height", "width", "legend", "xaxis", "yaxis", "margin")},
        title=dict(
            text=f"<b>{dim}</b> — Árvore de Contribuições  (tamanho = share modelo · cor = ROAS Index)",
            font=dict(size=14, color="#E0E0E0"),
        ),
        height=560,
        width=520 * n,
        showlegend=False,
        margin=dict(l=20, r=60, t=80, b=20),
    )

    if show:
        fig.show()
    return fig


def analyze_trees(
    all_results: dict,
    dims: list[str] | None = None,
    chart_type: str = "sunburst",
    show: bool = True,
    vehicle_spec_override: dict | None = None,
) -> dict[str, go.Figure]:
    """Tree visualization for all dims with hierarchy rollups.

    Skips dims without any groups/map rollup config in vehicle_spec.
    Returns {dim: fig}.

    chart_type: "sunburst" (default) | "treemap" | "icicle"
    """
    vehicle_spec: dict = vehicle_spec_override or {}
    if not vehicle_spec and all_results:
        first = next(iter(all_results.values()))
        vehicle_spec = getattr(getattr(first, "config", None), "vehicle_spec", {})

    breakdowns = vehicle_spec.get("breakdowns", {})
    all_dims = dims or list(breakdowns.keys())
    hierarchy = vehicle_spec.get("hierarchy", {})
    figs: dict[str, go.Figure] = {}

    for dim in all_dims:
        bd_spec = breakdowns.get(dim, {})
        rollup_specs = bd_spec.get("rollups", [])
        has_groups = any(
            "groups" in r and hierarchy.get(r["groups"]) for r in rollup_specs
        )
        has_map = any(
            "map" in r and hierarchy.get(r["map"]) for r in rollup_specs
        )
        if not has_groups and not has_map:
            continue

        fig = plot_tree_dim(dim, all_results, vehicle_spec, chart_type=chart_type)
        if fig is not None:
            if show:
                fig.show()
            figs[dim] = fig

    return figs
