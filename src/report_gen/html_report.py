"""Relatório HTML do deep dive no Fractal Z, a partir dos CSVs de `generate_report()`.

Lê só o que `report.generate_report` já grava em `outputs/{client}_{vehicle}/` —
`contributions.csv`, `weekly_contributions.csv`, `hill_params.csv`, `diagnostics.csv`
e `metadata.csv` —, então o HTML se refaz em segundos, sem refazer o fit.

Um documento só, com uma ou várias pastas (cada uma é um "veículo"): visão geral
fixa e um filtro que troca os blocos do veículo. Quando o veículo tem mais de uma
quebra (`dim`), elas se empilham dentro do mesmo bloco.

O visual e o JS são do Fractal Z (`_fractal_z`, copiado do model-development). A
estrutura — sidebar `[ NN ]`, blocos, filtro de mídia — segue o
`deepdive_report.py` da dupla de DS do projeto TIM Ultrafibra.
"""

from __future__ import annotations

import html as _html
import itertools
import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import _fractal_z as fz
from ._payload import pack_figures

# Gates do raven-deep-dive (README §6).
PROXY_MIN, PROXY_MAX = 0.85, 1.15
CSL_MAX = 0.20
# Mesmos limites do `diagnostics`: filha saturada na média, ou com slope de degrau,
# não tem resposta a investimento identificada.
SATURACAO_ALERTA = 0.90
SLOPE_ALERTA = 8.0
_ALERTA_TXT = {
    "saturada": "Hill saturada no gasto médio: a contribuição é quase constante sempre "
                "que o item veicula, então o share dele vem de semanas ativas × efeito "
                "máximo, não do quanto se investiu. ROAS index não mede resposta.",
    "degrau": "Slope de degrau: a resposta é liga/desliga. ROAS index não mede "
              "resposta a investimento.",
}
_LEGENDA_AJUSTE = (
    "<b>R²</b>: quanto da variação semanal da âncora a soma dos itens reproduz (1 = perfeito). "
    "Mede o formato da série.",
    "<b>WAPE</b>: erro semanal médio da soma dos itens contra a âncora, em % da âncora "
    "(Σ|âncora − soma| / Σ|âncora|). Pode ser alto mesmo com proxy_ratio ≈ 1, que só confere o total.",
    "R² e WAPE são calculados na própria amostra do fit, sem holdout.",
)
_LEGENDA_SLOPE = (
    "<b>slope</b>: inclinação da curva Hill. Até ~1 a curva é côncava (retorno decrescente desde o "
    "primeiro real); acima de 1 ela tem forma de S, e o efeito acelera antes de saturar.",
    f"Slope acima de {SLOPE_ALERTA:g} é <b>degrau</b>: a resposta é liga/desliga, e o ROAS index não "
    f"mede resposta a investimento. Saturação no gasto médio acima de {SATURACAO_ALERTA:.0%} "
    "também gera alerta.",
)
_STATUS_TXT = {
    "kept": ("mantida", "ok"),
    "others_aggregate": ("agregada em __others__", "neutral"),
    "dim_skip": ("dimensão ignorada", "neutral"),
}

BLOCOS_GERAIS = (("b01", 1, "Visão geral"),)
# ordem de leitura; o número do bloco é atribuído na montagem, sem buraco
BLOCOS_VEICULO = ("b02", "b03", "b04", "b05", "b06")

_EXTRA_JS = """
document.addEventListener('click', function (e) {
  var tr = e.target.closest ? e.target.closest('tr[data-veiculo]') : null;
  if (!tr || !window.FZ_NAV) return;
  FZ_NAV.filtro('veiculo', tr.dataset.veiculo, false);
  FZ_NAV.ir('b02', true);
});
"""


# ---------------------------------------------------------------------------
# Formatação e marcação
# ---------------------------------------------------------------------------

def _esc(v) -> str:
    return _html.escape("" if v is None else str(v))


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and not np.isfinite(v))


def _num(v, casas: int = 2) -> str:
    return f"{v:,.{casas}f}" if _ok(v) else "—"


def _pct(v, casas: int = 1) -> str:
    return f"{v * 100:,.{casas}f}%" if _ok(v) else "—"


def _tabela(headers: list, rows: list, row_attrs: list | None = None,
            caption: str | None = None) -> str:
    """Tabela ordenável. Célula: HTML (str) ou `(html, valor_cru)`; o valor cru vai
    em `data-sort` porque o texto formatado ("9,0%" vs "12,4%") ordena errado."""
    head = "".join(
        f'<th data-sort-col="{i}"><span>{_esc(h)}</span></th>' for i, h in enumerate(headers)
    )
    body = []
    for i, row in enumerate(rows):
        attrs = "".join(f' {k}="{_esc(v)}"' for k, v in (row_attrs[i] if row_attrs else {}).items())
        tds = []
        for cell in row:
            html, raw = cell if isinstance(cell, tuple) else (cell, None)
            sort = f' data-sort="{_esc(raw)}"' if _ok(raw) else (' data-sort=""' if raw is not None else "")
            tds.append(f"<td{sort}>{html}</td>")
        body.append(f"<tr{attrs}>{''.join(tds)}</tr>")
    cap = f'<div class="table-caption" data-table-caption="{_esc(caption)}"></div>' if caption else ""
    return (
        '<div class="sortable-table"><div class="table-wrap"><table data-sortable>'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>{cap}</div>"
    )


def _tabs(items: list[tuple[str, str]]) -> str:
    items = [(rot, conteudo) for rot, conteudo in items if conteudo]
    if len(items) == 1:
        return items[0][1]
    uid = uuid.uuid4().hex[:8]
    botoes = "".join(
        f'<button type="button" class="tab-btn{" active" if i == 0 else ""}" '
        f'data-target="panel-{uid}-{i}">{_esc(rot)}</button>'
        for i, (rot, _) in enumerate(items)
    )
    paineis = "".join(
        f'<div class="tab-panel{" active" if i == 0 else ""}" id="panel-{uid}-{i}">{c}</div>'
        for i, (_, c) in enumerate(items)
    )
    return (
        f'<div class="tabs" data-tabs="{uid}"><div class="tab-bar">{botoes}</div>'
        f'<div class="tab-panels">{paineis}</div></div>'
    )


class _Figuras:
    """Registro das figuras de um documento: guarda a spec e devolve um placeholder
    que o JS do Fractal Z hidrata quando o painel fica visível."""

    def __init__(self) -> None:
        self._specs: dict[str, dict] = {}
        self._ids = itertools.count()

    def html(self, fig: go.Figure) -> str:
        spec = json.loads(fig.to_json())
        fid = f"f{next(self._ids)}"
        self._specs[fid] = spec
        altura = spec.get("layout", {}).get("height") or 400
        return f'<div class="pfig" data-fig="{fid}" style="height:{altura}px"></div>'

    def store(self) -> str:
        return (
            '<script type="application/json" id="cv-figs" data-fz-figs>'
            f"{pack_figures(self._specs)}</script>"
        )


def _plotly_js_tag(fonte: str) -> str:
    if fonte == "inline":
        from plotly.offline import get_plotlyjs

        return f"<script>{get_plotlyjs()}</script>"
    from plotly.offline import get_plotlyjs_version

    return f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>'


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def carregar_veiculo(pasta: Path) -> dict:
    """Lê os CSVs de uma pasta `outputs/{client}_{vehicle}/`. Os opcionais
    (`hill_params`, `diagnostics`, `weekly_contributions`) ficam `None` se faltarem."""
    pasta = Path(pasta)

    def _csv(nome: str, obrigatorio: bool = False) -> pd.DataFrame | None:
        arq = pasta / nome
        if not arq.exists():
            if obrigatorio:
                raise FileNotFoundError(f"{arq} não existe — rode generate_report() antes")
            return None
        return pd.read_csv(arq)

    meta = _csv("metadata.csv", True).iloc[0].to_dict()
    contrib = _csv("contributions.csv", True)
    weekly = _csv("weekly_contributions.csv")
    if weekly is not None:
        weekly["date"] = pd.to_datetime(weekly["date"])
        rotulos = contrib.drop_duplicates(["dim", "level", "item"]).set_index(["dim", "level", "item"])["item_label"]
        weekly["item_label"] = [
            rotulos.get((d, l, i), i) for d, l, i in zip(weekly["dim"], weekly["level"], weekly["item"])
        ]
    hill = _csv("hill_params.csv")
    if hill is not None:
        # a curva precisa do gasto máximo da coluna: half_max_spend = half_max_norm × col_max
        hill["col_max"] = hill["half_max_spend"] / hill["half_max_norm"]
        hill["alerta"] = np.where(
            hill["saturation_at_mean"] > SATURACAO_ALERTA, "saturada",
            np.where(hill["slope"] > SLOPE_ALERTA, "degrau", ""),
        )
    return {
        "key": pasta.name,
        "nome": str(meta.get("vehicle") or pasta.name),
        "meta": meta,
        "contrib": contrib,
        "weekly": weekly,
        "hill": hill,
        "diag": _csv("diagnostics.csv"),
        "dims": list(dict.fromkeys(contrib["dim"])),
    }


def _gates(linha: pd.Series) -> tuple[bool, bool]:
    pr, csl = linha["proxy_ratio"], linha["csl_max_dev"]
    return (_ok(pr) and PROXY_MIN <= pr <= PROXY_MAX, _ok(csl) and csl < CSL_MAX)


def _itens(v: dict, dim: str, nivel: str | None = None) -> pd.DataFrame:
    c = v["contrib"]
    c = c[(c["dim"] == dim) & (c["level"] == (nivel or dim))].copy()
    return c.sort_values("contrib_share", ascending=False).reset_index(drop=True)


def _niveis(v: dict, dim: str) -> list[str]:
    """Nível atômico (o próprio nome da dim) primeiro, depois os rollups."""
    c = v["contrib"]
    niveis = list(dict.fromkeys(c[c["dim"] == dim]["level"]))
    return [dim] + [n for n in niveis if n != dim]


# ---------------------------------------------------------------------------
# Figuras — cor de série por posição (fz.serie), chrome pelo tema (THEME_JS)
# ---------------------------------------------------------------------------

def _fig_contrib_vs_gasto(itens: pd.DataFrame) -> go.Figure:
    rot = itens["item_label"].tolist()[::-1]
    fig = go.Figure()
    for n, (col, nome) in enumerate(
        (("contrib_share", "share de contribuição"), ("spend_share", "share de investimento")), 1
    ):
        fig.add_trace(go.Bar(
            y=rot, x=itens[col].tolist()[::-1], orientation="h", name=nome, marker_color=fz.serie(n),
            hovertemplate="%{y}: %{x:.1%}<extra>" + nome.split(" de ")[1] + "</extra>",
        ))
    fig.update_layout(barmode="group")
    fz.layout_figura(fig, height=max(260, 70 + 56 * len(itens)))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(tickformat=".0%")
    return fig


def _fig_serie(weekly: pd.DataFrame, dim: str, nivel: str, coluna: str, ordem: list[str]) -> go.Figure:
    w = weekly[(weekly["dim"] == dim) & (weekly["level"] == nivel)]
    fig = go.Figure()
    for i, rot in enumerate(ordem, 1):
        d = w[w["item_label"] == rot].sort_values("date")
        fig.add_trace(go.Scatter(
            x=d["date"], y=d[coluna], name=rot, stackgroup="one", mode="lines",
            line=dict(width=1, color=fz.serie(i)), fillcolor=fz.serie_subtle(i),
        ))
    fz.layout_figura(fig, height=400)
    fig.update_layout(legend_traceorder="normal")  # o Plotly inverte a legenda de área empilhada
    return fig


def _fig_hill(hill: pd.DataFrame, rotulos: dict[str, str], ordem: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, slug in enumerate(ordem, 1):
        h = hill[hill["variable"] == slug]
        if h.empty:
            continue
        h = h.iloc[0]
        col_max = float(h["col_max"])
        if not np.isfinite(col_max) or col_max <= 0:
            continue
        rot = rotulos.get(slug, slug)
        x = np.linspace(0, 1.25 * col_max, 120)
        k, s = float(h["half_max_norm"]), float(h["slope"])
        me = float(h["max_effect_kpi"]) if "max_effect_kpi" in h and _ok(h["max_effect_kpi"]) else float(h["max_effect"])
        curva = lambda xn: me * xn**s / (xn**s + k**s)  # noqa: E731
        cor = fz.serie(i)
        fig.add_trace(go.Scatter(
            x=x, y=curva(x / col_max), name=rot, legendgroup=rot, line=dict(color=cor, width=2),
            hovertemplate="investimento %{x:,.0f}<br>efeito %{y:,.1f}<extra>" + _esc(rot) + "</extra>",
        ))
        if _ok(h["mean_spend"]):
            fig.add_trace(go.Scatter(
                x=[h["mean_spend"]], y=[curva(h["mean_spend"] / col_max)], mode="markers",
                showlegend=False, legendgroup=rot,  # some junto com a curva ao desmarcar na legenda
                marker=dict(color=cor, size=9, symbol="circle"),
                hovertemplate=f"{_esc(rot)} — gasto médio: %{{x:,.0f}}<extra></extra>",
            ))
    fz.layout_figura(fig, height=420)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="investimento semanal")
    fig.update_yaxes(title_text="contribuição semanal (unidade do target)")
    return fig


# ---------------------------------------------------------------------------
# Blocos
# ---------------------------------------------------------------------------

def _pill_gate(ok: bool) -> str:
    return fz.status_pill("dentro" if ok else "fora", "ok" if ok else "fail")


def _roas_tag(roas) -> str:
    if not _ok(roas):
        return fz.tag_trend("N/A", "neutral")
    return fz.tag_trend(_num(roas, 2), "positive" if roas >= 1 else "negative")


def _legenda(linhas: tuple[str, ...]) -> str:
    return '<ul class="fz-notes">' + "".join(f"<li>{t}</li>" for t in linhas) + "</ul>"


def _qualidade(linha: pd.Series) -> str:
    ok_pr, ok_csl = _gates(linha)
    return fz.tiles([
        fz.tile(_num(linha["proxy_ratio"], 3), "gate", "proxy_ratio", _pill_gate(ok_pr),
                hint=f"soma das contribuições / âncora · ok entre {PROXY_MIN} e {PROXY_MAX}"),
        fz.tile(_num(linha["csl_max_dev"], 3), "gate", "csl_max_dev", _pill_gate(ok_csl),
                hint=f"maior desvio do share contra a prior CSL · ok abaixo de {CSL_MAX}"),
        fz.tile(_num(linha["r2"], 3), "ajuste", "R² da âncora",
                hint="quanto da variação semanal da âncora a soma dos itens reproduz"),
        fz.tile(_pct(linha["wape"], 1), "ajuste", "WAPE da âncora",
                hint="erro semanal médio da soma dos itens contra a âncora, em % da âncora"),
    ]) + _legenda(_LEGENDA_AJUSTE)


def _tabela_contrib(itens: pd.DataFrame, alertas: dict[str, str]) -> str:
    headers = ["item", "contribuição", "share contrib.", "investimento", "share invest.", "ROAS index"]
    rows = []
    for i, r in enumerate(itens.itertuples(index=False), 1):
        alerta = alertas.get(r.item, "")
        pill = " " + fz.status_pill(alerta, "warn", title=_ALERTA_TXT[alerta]) if alerta else ""
        rows.append([
            f'<span class="fz-cell">{fz.swatch(i)}<span>{_esc(r.item_label)}</span>{pill}</span>',
            (_num(r.contrib_absolute, 0), r.contrib_absolute),
            (_pct(r.contrib_share, 1), r.contrib_share),
            (_num(r.spend_absolute, 0), r.spend_absolute),
            (_pct(r.spend_share, 1), r.spend_share),
            (_roas_tag(r.roas_index), r.roas_index),
        ])
    return _tabela(headers, rows)


def _tabela_hill(hill: pd.DataFrame, rotulos: dict[str, str], ordem: list[str]) -> str:
    headers = ["item", "max_effect", "half_max (gasto)", "slope", "gasto médio (semanas ativas)",
               "saturação no gasto médio", "alerta"]
    rows = []
    h = hill.set_index("variable")
    for i, slug in enumerate(ordem, 1):
        if slug not in h.index:
            continue
        r = h.loc[slug]
        alerta = r["alerta"]
        rows.append([
            f'<span class="fz-cell">{fz.swatch(i)}<span>{_esc(rotulos.get(slug, slug))}</span></span>',
            (_num(r["max_effect"], 3), r["max_effect"]),
            (_num(r["half_max_spend"], 0), r["half_max_spend"]),
            (_num(r["slope"], 2), r["slope"]),
            (_num(r["mean_spend"], 0), r["mean_spend"]),
            (_pct(r["saturation_at_mean"], 0), r["saturation_at_mean"]),
            fz.status_pill(alerta, "warn", title=_ALERTA_TXT[alerta]) if alerta else "—",
        ])
    return _tabela(headers, rows)


def _tabela_diag(diag: pd.DataFrame, dim: str) -> str:
    d = diag[diag["dim"] == dim]
    headers = ["item", "status", "motivo", "semanas ativas", "% do gasto da dim", "% da contribuição da dim"]
    rows = []
    for r in d.itertuples(index=False):
        txt, st = _STATUS_TXT.get(r.status, (r.status.replace("discarded_", "descartada: "), "fail"))
        rows.append([
            _esc(r.slug), fz.status_pill(txt, st), _esc(r.reason),
            (_num(r.active_weeks, 0), r.active_weeks),
            (_pct(r.pct_gate_dim, 1), r.pct_gate_dim),
            (_pct(r.pct_contrib_dim, 1), r.pct_contrib_dim),
        ])
    return _tabela(headers, rows)


_TEXTOS = {
    "b02": ("Qualidade", "Os dois gates do raven-deep-dive (proxy_ratio entre 0,85 e 1,15; "
            "csl_max_dev abaixo de 0,20) e o quanto os itens reproduzem a série da âncora."),
    "b03": ("Contribuição vs investimento", "Share da contribuição de cada item contra o share do "
            "investimento. ROAS index = share contrib. / share invest. — relativo à própria dimensão, "
            "não é ROAS absoluto."),
    "b04": ("Série semanal", "Contribuição e investimento semanais de cada item, empilhados."),
    "b05": ("Saturação", "Curva Hill de cada item sobre o investimento semanal, com o gasto médio "
            "nas semanas ativas (círculo). Em item com adstock a curva recebe o investimento já "
            "carregado, então lê-se como forma, não como resposta exata a um real a mais."),
    "b06": ("Diagnóstico", "Itens mantidos, descartados e agregados em __others__ pelos gates de "
            "investimento, com a contribuição que cada um recebeu."),
}


def _partes_veiculo(v: dict, figs: _Figuras) -> dict[str, str]:
    """O corpo dos blocos 02–06 de um veículo (`bid` → HTML; bloco sem dado fica de
    fora). Mais de uma quebra empilha as quebras no mesmo bloco, com subtítulo."""
    uid = v["key"].replace(".", "-")
    varias = len(v["dims"]) > 1
    partes: dict[str, list[str]] = {bid: [] for bid in BLOCOS_VEICULO}

    def sub(dim: str) -> str:
        return fz.text_display(f"Quebra por {dim}", size="small") if varias else ""

    for dim in v["dims"]:
        atomicos = _itens(v, dim)
        ordem = atomicos["item"].tolist()
        rotulos = dict(zip(atomicos["item"], atomicos["item_label"]))
        hill = v["hill"][v["hill"]["dim"] == dim] if v["hill"] is not None else None
        alertas = dict(zip(hill["variable"], hill["alerta"])) if hill is not None and not hill.empty else {}

        partes["b02"].append(sub(dim) + _qualidade(atomicos.iloc[0]))

        abas = []
        for nivel in _niveis(v, dim):
            it = _itens(v, dim, nivel)
            abas.append((nivel, fz.block(figs.html(_fig_contrib_vs_gasto(it)))
                         + fz.block(_tabela_contrib(it, alertas if nivel == dim else {}))))
        partes["b03"].append(sub(dim) + fz.painel(_tabs(abas), f"contrib-{dim}-{uid}"))

        if v["weekly"] is not None:
            abas = []
            for nivel in _niveis(v, dim):
                ord_n = _itens(v, dim, nivel)["item_label"].tolist()
                abas.append((
                    nivel,
                    _tabs([
                        ("Contribuição", figs.html(_fig_serie(v["weekly"], dim, nivel, "contrib", ord_n))),
                        ("Investimento", figs.html(_fig_serie(v["weekly"], dim, nivel, "spend", ord_n))),
                    ]),
                ))
            partes["b04"].append(sub(dim) + fz.painel(fz.block(_tabs(abas)), f"serie-{dim}-{uid}"))

        if hill is not None and not hill.empty:
            partes["b05"].append(sub(dim) + fz.painel(
                fz.block(figs.html(_fig_hill(hill, rotulos, ordem)))
                + fz.block(_tabela_hill(hill, rotulos, ordem) + _legenda(_LEGENDA_SLOPE)),
                f"hill-{dim}-{uid}",
            ))

        if v["diag"] is not None:
            partes["b06"].append(sub(dim) + fz.painel(fz.block(_tabela_diag(v["diag"], dim)), f"diag-{dim}-{uid}"))

    return {bid: "".join(p) for bid, p in partes.items() if p}


def _blocos_veiculo(v: dict, partes: dict[str, str], numeros: dict[str, int]) -> str:
    out = []
    for bid, corpo in partes.items():
        titulo, desc = _TEXTOS[bid]
        head = fz.bloco_head(numeros[bid], titulo, v["nome"], desc)
        out.append(fz.bloco(bid, head + '<div class="fz-stack">' + corpo + "</div>", "veiculo"))
    return "".join(out)


def _kpis_topo(veiculos: list[dict]) -> str:
    linhas = [_gates(_itens(v, d).iloc[0]) for v in veiculos for d in v["dims"]]
    n_ok = sum(all(g) for g in linhas)
    invest = sum(float(_itens(v, d)["spend_absolute"].sum()) for v in veiculos for d in v["dims"])
    m0 = veiculos[0]["meta"]
    return fz.tiles([
        fz.tile(str(len(veiculos)), "escopo", "veículos"),
        fz.tile(f"{n_ok}/{len(linhas)}", "gates", "quebras dentro dos gates", _pill_gate(n_ok == len(linhas))),
        fz.tile(_num(invest / 1e6, 1) + " mi", "investimento", "investimento total (todas as quebras)"),
        fz.tile(str(m0.get("upgrade_run_id", ""))[:8], "âncora", "run base",
                hint=f"janela {m0.get('period_start')} → {m0.get('period_end')}"),
    ])


def _visao_geral(veiculos: list[dict], figs: _Figuras) -> str:
    headers = ["veículo", "quebra", "itens", "proxy_ratio", "csl_max_dev", "R²", "WAPE",
               "investimento", "maior ROAS idx", "menor ROAS idx"]
    rows, attrs = [], []
    for n, v in enumerate(veiculos, 1):
        for dim in v["dims"]:
            it = _itens(v, dim)
            ok_pr, ok_csl = _gates(it.iloc[0])
            ok_roas = it.dropna(subset=["roas_index"])
            mx = ok_roas.loc[ok_roas["roas_index"].idxmax()] if len(ok_roas) else None
            mn = ok_roas.loc[ok_roas["roas_index"].idxmin()] if len(ok_roas) else None
            invest = float(it["spend_absolute"].sum())
            r = it.iloc[0]
            rows.append([
                f'<span class="fz-cell">{fz.swatch(n)}<span>{_esc(v["nome"])}</span></span>',
                _esc(dim),
                (str(len(it)), len(it)),
                (fz.status_pill(_num(r["proxy_ratio"], 3), "ok" if ok_pr else "fail"), r["proxy_ratio"]),
                (fz.status_pill(_num(r["csl_max_dev"], 3), "ok" if ok_csl else "fail"), r["csl_max_dev"]),
                (_num(r["r2"], 3), r["r2"]),
                (_pct(r["wape"], 1), r["wape"]),
                (_num(invest, 0), invest),
                (f"{_esc(mx.item_label)} {_num(mx.roas_index, 2)}", mx.roas_index) if mx is not None else "—",
                (f"{_esc(mn.item_label)} {_num(mn.roas_index, 2)}", mn.roas_index) if mn is not None else "—",
            ])
            attrs.append({"data-veiculo": v["key"]})
    head = fz.bloco_head(
        1, "Visão geral", "Todos os veículos",
        "Uma linha por veículo e quebra, com os gates (proxy_ratio entre 0,85 e 1,15; csl_max_dev "
        "abaixo de 0,20) e os itens extremos em ROAS index.",
    )
    corpo = fz.block(_tabela(headers, rows, attrs, "Clique numa linha para abrir a qualidade do veículo."))
    return fz.bloco("b01", head + fz.painel(corpo, "visao-geral"))


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------

def generate_html_report(
    run_dirs: list[str | Path],
    out_path: str | Path | None = None,
    client_name: str = "",
    plotly_source: str = "cdn",
) -> Path:
    """HTML de todos os veículos: blocos gerais + blocos do veículo do filtro.

    Args:
        run_dirs: pastas `outputs/{client}_{vehicle}/` gravadas por `generate_report`.
        out_path: destino; por padrão `deepdive.html` na pasta-mãe da primeira.
        client_name: overline do sidebar.
        plotly_source: `"cdn"` (leve, exige rede) ou `"inline"` (~3 MB, abre offline).
    """
    pastas = [Path(p) for p in run_dirs]
    if not pastas:
        raise ValueError("run_dirs vazio")
    destino = Path(out_path) if out_path else pastas[0].parent / "deepdive.html"
    veiculos = [carregar_veiculo(p) for p in pastas]
    figs = _Figuras()

    m0 = veiculos[0]["meta"]
    barra = fz.barra_filtro("veiculo", "veículo", [(v["key"], v["nome"]) for v in veiculos], "veiculo")
    titulo = "Deep dive"
    intro = fz.bloco("b00", (
        fz.bloco_head(
            0, "Introdução", titulo,
            f'{_esc(m0.get("model_name", ""))} · âncora run <span class="fz-code">'
            f'{_esc(str(m0.get("upgrade_run_id", ""))[:8])}</span> · gerado em {datetime.now():%d/%m/%Y %H:%M}',
            size="xlarge", subtitle_html=True,
        ) + _kpis_topo(veiculos)
    ))
    partes = [_partes_veiculo(v, figs) for v in veiculos]
    presentes = [bid for bid in BLOCOS_VEICULO if any(bid in p for p in partes)]
    numeros = {bid: n for n, bid in enumerate(presentes, start=2)}
    paineis = "".join(
        fz.painel_filtro("veiculo", v["key"], _blocos_veiculo(v, p, numeros), ativo=(i == 0))
        for i, (v, p) in enumerate(zip(veiculos, partes))
    )
    sidebar = fz.sidebar(
        contexto=[f"janela {m0.get('period_start')} → {m0.get('period_end')}",
                  f"gerado em {datetime.now():%d/%m/%Y %H:%M}"],
        grupos=[("Geral", list(BLOCOS_GERAIS)),
                ("Por veículo", [(bid, numeros[bid], _TEXTOS[bid][0]) for bid in presentes])],
        titulo=titulo,
        overline=client_name or str(m0.get("client", "")),
    )
    html = fz.documento(
        title=titulo,
        sidebar_html=sidebar,
        body=intro + _visao_geral(veiculos, figs) + barra + paineis,
        plotly_js=_plotly_js_tag(plotly_source),
        figures=figs.store(),
        script=fz.scripts(_EXTRA_JS),
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(html, encoding="utf-8")
    return destino
