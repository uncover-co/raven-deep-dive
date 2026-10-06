"""Empacotamento compacto de dados e figuras dos relatórios HTML.

Compartilhado por `eda_report` e `cv_report`. Uma figura Plotly serializada inline
custa caro por motivos que nada têm a ver com o gráfico: o eixo de datas é reescrito
por inteiro em cada traço, os floats saem com a precisão completa do float64, e o
layout repete as mesmas dezenas de chaves de tema. Aqui a figura vira especificação
num registro; `pack_figures` deduplica os arrays num pool, extrai o layout comum numa
base única e o JS (`HYDRATE_JS`) monta a figura quando o painel fica visível.

Painéis que o navegador mostra um de cada vez (um escopo regional, por exemplo) podem
ainda ir em gzip/base64 (`gzip_b64`), descomprimidos pelo JS só quando abertos.
"""

import base64
import gzip
import html as _html
import json

# Qualquer subárvore que se repita compensa virar referência: o eixo de datas, a série
# de um canal que aparece em cinco gráficos diferentes, o `xaxis` com as cores da grade.
# A regra é uma só — serializou acima do limiar e apareceu mais de uma vez, vira
# `{"$": <índice>}`.
POOL_MIN_BYTES = 24    # medido: abaixo disso o tamanho estabiliza e o pool só cresce
FLOAT_DIGITS = 6       # dígitos significativos preservados nos dados numéricos
PANEL_GZIP_LEVEL = 9


def round_scalar(value):
    """Arredonda float para `FLOAT_DIGITS` significativos; nan/inf viram null."""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None  # JSON.parse rejeita NaN/Infinity literais
        return float(f"%.{FLOAT_DIGITS}g" % value)
    return value


def round_deep(node):
    """Aplica `round_scalar` em toda a árvore, para o pool comparar formas idênticas."""
    if isinstance(node, list):
        return [round_deep(v) for v in node]
    if isinstance(node, tuple):
        return [round_deep(v) for v in node]
    if isinstance(node, dict):
        return {k: round_deep(v) for k, v in node.items()}
    return round_scalar(node)


def dumps(node) -> str:
    return json.dumps(node, separators=(",", ":"), allow_nan=False)


def tally(node, counter: dict) -> None:
    """Conta quantas vezes cada subárvore grande aparece no documento inteiro.

    Só desce em nós que já passam do limiar: a serialização de um filho nunca é maior
    que a do pai, então um pai pequeno não esconde um filho grande.
    """
    if not isinstance(node, (list, dict)):
        return
    key = dumps(node)
    if len(key) < POOL_MIN_BYTES:
        return
    counter[key] = counter.get(key, 0) + 1
    for child in node if isinstance(node, list) else node.values():
        tally(child, counter)


def _descend(node, counter: dict, pool: list, index: dict):
    if isinstance(node, list):
        return [pool_walk(v, counter, pool, index) for v in node]
    return {k: pool_walk(v, counter, pool, index) for k, v in node.items()}


def pool_walk(node, counter: dict, pool: list, index: dict):
    """Troca pela referência a maior subárvore repetida encontrada em cada ramo.

    A entrada do pool é ela própria percorrida: sem isso, um traço inteiro que se repete
    guardaria o eixo de datas embutido, ao lado da cópia solta que as outras figuras
    referenciam. As referências internas só apontam para subárvores menores, então a
    resolução no JS termina.
    """
    if not isinstance(node, (list, dict)):
        return node
    key = dumps(node)
    if len(key) >= POOL_MIN_BYTES and counter.get(key, 0) >= 2:
        slot = index.get(key)
        if slot is None:
            slot = len(pool)
            index[key] = slot
            pool.append(None)  # reserva o índice antes de descer
            pool[slot] = _descend(node, counter, pool, index)
        return {"$": slot}
    return _descend(node, counter, pool, index)


def common_layout(layouts: list[dict]) -> dict:
    """Chaves de layout presentes e idênticas em TODAS as figuras."""
    if not layouts:
        return {}
    shared = {k: dumps(v) for k, v in layouts[0].items()}
    for layout in layouts[1:]:
        for key in list(shared):
            if key not in layout or dumps(layout[key]) != shared[key]:
                del shared[key]
        if not shared:
            break
    return {k: layouts[0][k] for k in shared}


def pack_figures(fig_specs: dict) -> str:
    """Serializa um registro ``{id: spec plotly}`` como pool + base de layout + specs."""
    specs = {fid: round_deep(spec) for fid, spec in fig_specs.items()}
    base = common_layout([spec.get("layout", {}) for spec in specs.values()])
    trimmed = {
        fid: {
            "d": spec.get("data", []),
            "l": {k: v for k, v in spec.get("layout", {}).items() if k not in base},
        }
        for fid, spec in specs.items()
    }

    counter: dict = {}
    for spec in trimmed.values():
        tally(spec["d"], counter)
        tally(spec["l"], counter)

    # a lista `pool` entra no payload por referência e é preenchida pelo walk abaixo
    pool: list = []
    index: dict = {}
    payload = {
        "pool": pool,
        "base": base,
        "specs": {
            fid: {
                "d": pool_walk(spec["d"], counter, pool, index),
                "l": pool_walk(spec["l"], counter, pool, index),
            }
            for fid, spec in trimmed.items()
        },
    }
    # `</` escapado: o JSON mora dentro de <script>, e a sequência fecharia a tag
    return dumps(payload).replace("</", "<\\/")


def gzip_b64(text: str) -> str:
    """Texto em gzip/base64, para o JS descomprimir com `payloadInflate`."""
    raw = gzip.compress(text.encode("utf-8"), compresslevel=PANEL_GZIP_LEVEL, mtime=0)
    return base64.b64encode(raw).decode("ascii")


def gunzip_b64(b64: str) -> str:
    """Inverso de `gzip_b64`: serve para inspecionar e testar o HTML em Python."""
    return gzip.decompress(base64.b64decode(b64)).decode("utf-8")


# Contraparte JS. Globais de propósito: cada relatório tem o próprio registro de
# figuras e a própria montagem de painel, mas a resolução do pool e a descompressão
# são as mesmas.
HYDRATE_JS = """
// resolve as referências {"$": n} contra o pool e copia: o mesmo eixo de datas serve
// centenas de gráficos, e o Plotly escreve metadados nos objetos que recebe. Entradas
// do pool também podem conter referências, e sempre para subárvores menores — a
// recursão termina.
function payloadHydrate(pack, node) {
  if (Array.isArray(node)) return node.map(function (v) { return payloadHydrate(pack, v); });
  if (node && typeof node === 'object') {
    if (typeof node['$'] === 'number' && Object.keys(node).length === 1) {
      return payloadHydrate(pack, pack.pool[node['$']]);
    }
    var out = {};
    for (var k in node) out[k] = payloadHydrate(pack, node[k]);
    return out;
  }
  return node;
}

// merge profundo: a base traz xaxis/yaxis/font já prontos e a figura sobrepõe
// apenas as chaves internas que mudou (um título de eixo, um range)
function payloadPlain(v) { return v && typeof v === 'object' && !Array.isArray(v); }
function payloadMerge(base, over) {
  var out = {};
  for (var k in base) out[k] = base[k];
  for (var j in over) {
    out[j] = (payloadPlain(out[j]) && payloadPlain(over[j])) ? payloadMerge(out[j], over[j]) : over[j];
  }
  return out;
}

// base64 -> gzip -> texto, com o descompressor nativo do navegador
function payloadInflate(b64) {
  var bin = atob(b64.trim());
  var bytes = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  var stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).text();
}
"""


# --------------------------------------------------------------------------------------
# filtro de região
# --------------------------------------------------------------------------------------
# Relatório com um painel por escopo (Total + cada região): cada escopo vai em
# gzip/base64 com as próprias figuras e é montado pelo navegador quando escolhido no
# filtro. Uma parte de escopo é um `<div data-scope-panel=<escopo>
# data-scope-part=<parte>>`; o pacote do escopo traz o HTML de cada parte. O relatório
# que usa o filtro define `window.REPORT_SCOPE = {addFigs, init, refresh}`, os ganchos
# para registrar as figuras do pacote, ligar os widgets do painel montado e redesenhar
# o painel que acabou de aparecer.


def volume_label(value) -> str:
    """Volume curto com vírgula decimal: ``1,15mi``, ``86,3k``, ``862``."""
    if value is None or value != value or value in (float("inf"), float("-inf")):
        return "—"
    for divisor, suffix in ((1e6, "mi"), (1e3, "k")):
        if abs(value) >= divisor:
            text = f"{value / divisor:.3g}"
            if "e" in text:
                text = f"{value / divisor:.0f}"
            return text.replace(".", ",") + suffix
    return f"{value:.3g}".replace(".", ",")


def scope_part_html(key: str, part: str, active: bool, fixed: bool = False) -> str:
    """Contêiner vazio de uma parte de escopo, preenchido pelo JS ao abrir o escopo.

    ``fixed`` marca a parte que não troca com o filtro (um comparativo entre
    regiões): ela é montada no carregamento e fica sempre visível.
    """
    cls = f'scope-panel{" active" if active or fixed else ""}'
    fixed_attr = " data-scope-fixed" if fixed else ""
    return (
        f'<div class="{cls}" data-scope-panel="{_html.escape(key)}" '
        f'data-scope-part="{_html.escape(part)}"{fixed_attr} data-scope-lazy></div>'
    )


def scope_payload_html(key: str, parts: dict, figures_json: str) -> str:
    """Pacote de um escopo: o HTML de cada parte + o registro de figuras dele."""
    payload = '{"h":' + json.dumps(parts, ensure_ascii=False) + ',"f":' + figures_json + "}"
    return (
        f'<script type="application/octet-stream" data-scope-payload="{_html.escape(key)}">'
        f"{gzip_b64(payload)}</script>"
    )


def scope_bar_html(scopes: list, hint: str) -> str:
    """Filtro de região: ``scopes`` é ``[(chave, rótulo, volume)]``, o primeiro ativo."""
    options = "".join(
        f'<option value="{_html.escape(key)}"{" selected" if i == 0 else ""}>'
        f"{_html.escape(label)} ({_html.escape(volume_label(volume))})</option>"
        for i, (key, label, volume) in enumerate(scopes)
    )
    return (
        '<div class="scope-bar"><label class="scope-tag" for="report-scope">região</label>'
        f'<select id="report-scope" data-scope-filter>{options}</select>'
        f'<span class="scope-hint">{_html.escape(hint)}</span></div>'
    )


SCOPE_CSS = """
.scope-bar { position: sticky; top: 0; z-index: 30; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 12px 16px; margin: 0 0 24px; background: rgba(11,13,18,.94); border: 1px solid var(--border, #232a3a); border-radius: 12px; backdrop-filter: blur(6px); }
.scope-tag { font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted, #8a93a6); }
.scope-bar select { background: #151a24; color: inherit; border: 1px solid var(--border, #232a3a); border-radius: 8px; padding: 6px 10px; font: inherit; min-width: 220px; }
.scope-hint { font-size: 12px; color: var(--muted, #8a93a6); }
.scope-panel { display: none; }
.scope-panel.active { display: block; }
.scope-loading { padding: 48px 0; text-align: center; color: var(--muted, #8a93a6); }
.scope-fixed-badge { display: inline-block; margin-left: 8px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; padding: 2px 8px; border-radius: 20px; vertical-align: middle; background: rgba(138,147,166,.12); color: var(--muted, #8a93a6); border: 1px solid rgba(138,147,166,.35); }
tr.scope-row-active > td { background: rgba(0, 200, 255, .12); font-weight: 600; }
"""


SCOPE_JS = """
// Filtro de região e montagem dos painéis. Vem por último: quando roda, o relatório
// já definiu os ganchos em window.REPORT_SCOPE.
(function () {
  var hooks = window.REPORT_SCOPE || {};
  var select = document.querySelector('[data-scope-filter]');
  var loading = {};
  var current = null;

  function parts(key) {
    return Array.prototype.slice.call(
      document.querySelectorAll('[data-scope-panel="' + key + '"]'));
  }

  // cada escopo traz as suas partes e as suas figuras num pacote só: é descomprimido
  // uma vez e ligado
  function ensure(key) {
    var lazy = parts(key).filter(function (p) { return p.hasAttribute('data-scope-lazy'); });
    if (!lazy.length) return Promise.resolve();
    if (loading[key]) return loading[key];
    var src = document.querySelector('script[data-scope-payload="' + CSS.escape(key) + '"]');
    if (!src) return Promise.resolve();
    if (typeof DecompressionStream === 'undefined') {
      lazy.forEach(function (p) {
        p.innerHTML = '<p class="scope-loading">Este navegador não descomprime o '
          + 'relatório (DecompressionStream). Abra em um Chrome, Edge, Firefox ou Safari '
          + 'recente.</p>';
      });
      return Promise.resolve();
    }
    lazy.forEach(function (p) { p.innerHTML = '<p class="scope-loading">carregando…</p>'; });
    loading[key] = payloadInflate(src.textContent).then(function (text) {
      var pack = JSON.parse(text);
      if (hooks.addFigs) hooks.addFigs(pack.f);
      lazy.forEach(function (p) {
        p.innerHTML = pack.h[p.dataset.scopePart] || '';
        p.removeAttribute('data-scope-lazy');
        if (hooks.init) hooks.init(p);
      });
      src.remove();  // o base64 já virou DOM: não precisa ficar na memória
    }).catch(function (e) {
      console.error(e);
      lazy.forEach(function (p) {
        p.innerHTML = '<p class="scope-loading">Falha ao montar o painel: ' + e.message + '</p>';
      });
      delete loading[key];
    });
    return loading[key];
  }

  function highlight() {
    document.querySelectorAll('tr[data-scope-row]').forEach(function (row) {
      row.classList.toggle('scope-row-active', row.dataset.scopeRow === current);
    });
  }

  function refresh(panel) { if (hooks.refresh) hooks.refresh(panel); }

  function show(key) {
    current = key;
    document.querySelectorAll('[data-scope-panel]').forEach(function (p) {
      if (p.hasAttribute('data-scope-fixed')) return;
      p.classList.toggle('active', p.dataset.scopePanel === key);
    });
    highlight();
    ensure(key).then(function () {
      if (current !== key) return;  // o usuário já trocou de região
      parts(key).forEach(refresh);
    });
  }

  var fixed = [];
  document.querySelectorAll('[data-scope-fixed]').forEach(function (p) {
    if (fixed.indexOf(p.dataset.scopePanel) === -1) fixed.push(p.dataset.scopePanel);
  });
  fixed.forEach(function (key) {
    ensure(key).then(function () { parts(key).forEach(refresh); highlight(); });
  });
  if (select) {
    select.addEventListener('change', function () { show(select.value); });
    show(select.value);
  }
})();
"""
