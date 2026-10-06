"""Fractal Z, o design system da Uncover, para os relatórios HTML da pasta.

Cópia de uncover-co/model-development, branch model/tim-ultrafibra-upgrade, commit
9351c2e: models/tim/ultrafibra/nacional_upgrade/src/report_gen/_fractal_z.py. Única
alteração: `rotulo_cliente` não lê mais o globals.yml do projeto de modelo (aqui o
cliente é passado por argumento). Atualize por cópia, não por edição.


Todos os relatórios da pasta usam (`deepdive_report`, `cv_report`,
`prior_posterior_report`, `eda_report`). Os nomes de classe próprios levam o
prefixo `fz-`; as classes sem prefixo que o motor trata (`.tab-btn`,
`.sortable-table`, `table[data-sortable]`) são as que os relatórios já emitiam.

Fonte de verdade: `design_uncover.md` (skill `fractal-z-design-system`) e a camada
de tokens da skill `uncover-deck` (`references/tokens.md`), que traz os valores
dos dois modos vindos do Supernova. Nenhuma cor, raio, espaço ou tipo entra aqui
sem sair de lá.

* `css()` — `@font-face` embutido, tokens dos dois modos e os componentes.
* `serie(n)` / `serie_subtle(n)` — cor de série por posição (`data-viz-01..20`).
  É invariante de modo, então pode ir cravada na figura Plotly.
* `layout_figura()` — layout Plotly sem cor nenhuma de chrome: texto, grade e
  hover são lidos dos tokens pelo `THEME_JS` na hora de montar a figura e
  refeitos quando o tema troca. O Plotly não entende `var(--…)`.
* helpers de marcação: `sidebar`, `bloco`, `text_display`, `tile`, `section`,
  `painel`, `tag_trend`, `status_pill`, `filter_select`, `accordion`, `swatch`,
  `ajuda`.
* `documento()` — o esqueleto: Menu Sidebar à esquerda, blocos à direita;
  `blocos()` numera sem buraco; `barra_filtro`/`painel_filtro` são o filtro do
  topo (mídia, fold); `scripts()` junta o JS na ordem certa.
* `cor(nome)` — cor de chrome ou feedback que acompanha o tema dentro da figura
  Plotly (sentinela trocada pelo `MOTOR_JS`). Série continua `serie(n)`.

Navegação por blocos (`NAV_JS`): cada bloco numerado é uma aba. O Menu Sidebar
segue o mapeamento do deck (`uncover-deck`, components.md §2): logo, tema e
colapsar no topo; contexto do relatório; lista `[ NN ] Nome`; 280px, colapsa
para 64px com só os numerais. Um bloco por vez; "Mostrar tudo" volta à rolagem
contínua; o pé de cada bloco tem voltar/avançar; setas do teclado navegam; o
hash guarda o estado (`#bloco=b03&midia=…`); a impressão mostra todos. Quando os
blocos chegam depois do carregamento (o corpo do EDA vem em gzip), `nav_js(adiado=True)`
só define `window.fzNavInit()`, e o relatório chama depois de montar o DOM.

Tema: segue o `prefers-color-scheme` do sistema, e o toggle do header sobrescreve
(`data-tema` + localStorage). É escolha deliberada para relatório; o deck da
Uncover abre claro.

Lacunas do spec, compostas a partir dos primitivos:
* Abas — o sistema não tem Tabs. Viraram chips Filter: `standard` o inativo,
  `active` o selecionado, sem ícones.
* Pílula de status (gate dentro/fora, filha mantida/descartada) — geometria da
  Tag Trend com a família Feedback, porque status não é delta.
* Swatch de série na tabela — 8px, raio pill (a regra de status dot).
* Scrim de hover — `overlay-hover` a `opacity-moderate` (32%) com o blend do
  deck (`multiply` no claro, `screen` no escuro), a mecânica do Accordion.
* Altura do header do Accordion — sem spec; 48px, a da linha de Data Table.
* O painel do `<select>` é o nativo do navegador, não o `filter-dropdown`.
* Mono — o sistema não tem face monoespaçada; slug e ID de run saem em Inter.
* Barra sticky do filtro de mídia — fundo `layout-bg`, o próprio canvas.
* Tile dentro de Section — `surface-subtle`, o degrau do painel aninhado; com a
  cor padrão (`surface-default`) ele sumiria na Section.
* Cor de qualidade/limiar em gráfico (R-hat, ESS) — família Feedback via
  `cor()`, por decisão explícita: é status, não série categórica.
* Escala divergente (heatmap de correlação do EDA) — o sistema não tem; fica o
  `RdBu` do Plotly, igual nos dois modos.
* Texto sobre segmento colorido de barra — `content-accent-strong`, quase branco
  nos dois modos, como a série que ele cobre.
* Ordem de uso das séries (`ORDEM_SERIE`) — a especificação, com a 7 e a 12
  trocadas. Em ordem, a 7 (#265fec) é um segundo azul ao lado da 1 (#4d59c9),
  ΔE2000 5,6; a 12 (#bd005a) fica a 22,8 das oito primeiras (14,1 sob
  deuteranopia). Os tokens `--data-viz-NN` continuam os da especificação.
* Versalete (`font-variant-caps: small-caps`) no rótulo dos itens do sidebar e
  dos botões voltar/avançar — pedido explícito, sem token no Fractal Z. Nos
  botões ele substitui o UPPERCASE do Button `secondary`.
* Botões voltar/avançar — Button `secondary` `sm`, com o pé do bloco separado
  por uma Divider `stroke-subtle`.
* Ajuda (`?`) — o Tile prevê o `help_outline`, mas não diz o que ele faz. Virou
  popover acima do `?` (abaixo só sem espaço), com a sombra do nível 4
  (`0 12px 64px 0`, cor `--fz-shadow`). Fundo `surface-accent-subtle` + texto
  `content-default` e barra de 4px `stroke-accent` à esquerda — o par do estado
  Active, para destacar do painel; o spec não tem entrada de popover.
  Hover abre prévia, clique fixa; o `?` fixado fica em `content-accent`.
"""

from __future__ import annotations

import base64
import html as _html
from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).parent / "assets" / "fractal_z"

# Data Viz comparison — idêntica nos dois modos. A ordem é a especificação.
DATA_VIZ = (
    "#4d59c9", "#c83300", "#007a6d", "#922bbe", "#007e99",
    "#a55d00", "#265fec", "#809d00", "#7936d0", "#e46134",
    "#00844d", "#bd005a", "#0072ac", "#b74b00", "#b029ae",
    "#578000", "#6e4ee8", "#178900", "#009ad8", "#906900",
)
# Ordem em que as séries usam os tokens: a da especificação, com 7 e 12 trocadas
# (ver a lacuna na docstring). Posição n usa `--data-viz-{ORDEM_SERIE[n-1]}`.
ORDEM_SERIE = (1, 2, 3, 4, 5, 6, 12, 8, 9, 10, 11, 7, 13, 14, 15, 16, 17, 18, 19, 20)
# companheiro -subtle: a mesma matiz a 58% (sufixo hex 94)
SUBTLE_ALPHA = 0x94 / 255


def serie(n: int) -> str:
    """Cor da n-ésima série (1-based): o token ``ORDEM_SERIE[(n - 1) mod 20]``."""
    return DATA_VIZ[ORDEM_SERIE[(n - 1) % len(DATA_VIZ)] - 1]


def serie_subtle(n: int) -> str:
    """O `-subtle` da série n em ``rgba()``, que o validador do Plotly aceita."""
    h = serie(n).lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{SUBTLE_ALPHA:.3f})"


# Cor que muda com o tema, para dentro de figura Plotly: o Python grava uma
# sentinela válida (#fe00NN, ou rgba(254,0,N,a)) e o `fzCores` do MOTOR_JS troca
# pelo token do tema ativo ao montar a figura e a cada troca de tema.
_SENTINELAS = {
    "texto": ("--layout-content-default", 1),
    "texto-sutil": ("--layout-content-subtle", 2),
    "linha": ("--layout-stroke-strong", 3),
    "grade": ("--layout-stroke-subtle", 4),
    "superficie": ("--layout-surface-subtle", 5),
    "sucesso": ("--success-content", 6),
    "atencao": ("--warning-content", 7),
    "erro": ("--error-content", 8),
    "neutro": ("--trend-neutral-content", 9),
    "fundo": ("--layout-bg", 10),
    "desabilitado": ("--layout-content-disabled", 11),
}


def cor(nome: str, alpha: float | None = None) -> str:
    """Cor de chrome/feedback que acompanha o tema (ver `_SENTINELAS`)."""
    _, n = _SENTINELAS[nome]
    if alpha is None:
        return f"#fe00{n:02x}"
    return f"rgba(254,0,{n},{alpha:g})"


def serie_var(n: int) -> str:
    """A mesma cor como token CSS, para swatch e series bar."""
    return f"var(--data-viz-{ORDEM_SERIE[(n - 1) % len(DATA_VIZ)]:02d})"


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

_CLARO = """
  --layout-bg:#f2f2f2;
  --layout-surface-strong:#f2f2f2;
  --layout-surface-default:#e4e4e5;
  --layout-surface-subtle:#d0d1d2;
  --layout-surface-accent:#4f4fff;
  --layout-surface-accent-subtle:#e2e7ff;
  --layout-surface-inverse:#15181f;
  --layout-surface-disabled:#b8b9bc;
  --layout-content-default:#090d11;
  --layout-content-subtle:#4e535b;
  --layout-content-inverse:#fafafb;
  --layout-content-disabled:#9ca0a5;
  --layout-content-accent:#4f4fff;
  --layout-content-accent-strong:#e6ecff;
  --layout-stroke-default:#b8b9bc;
  --layout-stroke-subtle:#d0d1d2;
  --layout-stroke-strong:#82878d;
  --layout-stroke-accent:#6362e0;
  --layout-stroke-deep:#f2f2f2;
  --overlay-hover:#d0d1d2;
  --overlay-pressed:#9ca0a5;
  --success-surface-subtle:#e8f3de;
  --success-content:#577a20;
  --warning-surface-subtle:#f7ecd1;
  --warning-content:#995b00;
  --error-surface-subtle:#ffe7e6;
  --error-content:#b75151;
  --trend-positive-surface:#9cdfa6;
  --trend-positive-content:#2c633a;
  --trend-negative-surface:#ffb7b2;
  --trend-negative-content:#9a3b3f;
  --trend-neutral-surface:#d0d1d2;
  --trend-neutral-content:#4e535b;
  --fz-shadow:var(--layout-stroke-default);
  --overlay-blend:multiply;
"""

_ESCURO = """
  --layout-bg:#101215;
  --layout-surface-strong:#101215;
  --layout-surface-default:#191b1f;
  --layout-surface-subtle:#23252a;
  --layout-surface-accent:#4f4fff;
  --layout-surface-accent-subtle:#1b2138;
  --layout-surface-inverse:#caccce;
  --layout-surface-disabled:#23252a;
  --layout-content-default:#d8d9da;
  --layout-content-subtle:#878c93;
  --layout-content-inverse:#191b1f;
  --layout-content-disabled:#3b3f44;
  --layout-content-accent:#7788ef;
  --layout-content-accent-strong:#e5eaff;
  --layout-stroke-default:#3b3f44;
  --layout-stroke-subtle:#23252a;
  --layout-stroke-strong:#5c6168;
  --layout-stroke-accent:#4f4fff;
  --layout-stroke-deep:#0b0d0f;
  --overlay-hover:#3b3f44;
  --overlay-pressed:#3b3f44;
  --success-surface-subtle:#181c14;
  --success-content:#90c04d;
  --warning-surface-subtle:#1d1a13;
  --warning-content:#dd9b54;
  --error-surface-subtle:#211615;
  --error-content:#ec8c87;
  --trend-positive-surface:#111411;
  --trend-positive-content:#43ca6d;
  --trend-negative-surface:#171212;
  --trend-negative-content:#ec8c87;
  --trend-neutral-surface:#0b0d0f;
  --trend-neutral-content:#5c6168;
  --fz-shadow:var(--layout-stroke-deep);
  --overlay-blend:screen;
"""


def _tokens_css() -> str:
    viz = "".join(f"--data-viz-{i:02d}:{c};" for i, c in enumerate(DATA_VIZ, 1))
    # três estados: claro no :root, escuro pelo sistema (se o leitor não escolheu
    # claro) e escuro explícito pelo toggle
    return (
        ":root{color-scheme:light;"
        '--font-display:"PP Neue Machina",system-ui,sans-serif;'
        '--font-ui:"Inter",system-ui,sans-serif;'
        f"{viz}{_CLARO}}}\n"
        '@media (prefers-color-scheme:dark){:root:not([data-tema="light"]){'
        f"color-scheme:dark;{_ESCURO}}}}}\n"
        f':root[data-tema="dark"]{{color-scheme:dark;{_ESCURO}}}\n'
    )


@lru_cache(maxsize=1)
def _fontes_css() -> str:
    faces = (
        ("Inter", 400, "Inter-400.woff2"),
        ("Inter", 600, "Inter-600.woff2"),
        ("PP Neue Machina", 400, "PPNeueMachina-400.woff2"),
        ("PP Neue Machina", 600, "PPNeueMachina-600.woff2"),
    )
    out = []
    for familia, peso, arquivo in faces:
        b64 = base64.b64encode((ASSETS / arquivo).read_bytes()).decode("ascii")
        out.append(
            f'@font-face{{font-family:"{familia}";font-weight:{peso};font-style:normal;'
            f'font-display:swap;src:url(data:font/woff2;base64,{b64}) format("woff2")}}'
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Ícones (Material, 24px de viewBox, cor = currentColor)
# ---------------------------------------------------------------------------

_ICONES = {
    "expand_more": "M16.59 8.59 12 13.17 7.41 8.59 6 10l6 6 6-6z",
    "arrow_upward": "M4 12l1.41 1.41L11 7.83V20h2V7.83l5.58 5.59L20 12l-8-8-8 8z",
    "arrow_downward": "M20 12l-1.41-1.41L13 16.17V4h-2v12.17l-5.58-5.59L4 12l8 8 8-8z",
    "remove": "M19 13H5v-2h14v2z",
    "download": "M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z",
    "help_outline": (
        "M11 18h2v-2h-2v2zm1-16C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"
        "m0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-14c-2.21 0-4 1.79-4 4h2"
        "c0-1.1.9-2 2-2s2 .9 2 2c0 2-3 1.75-3 5h2c0-2.25 3-2.5 3-5 0-2.21-1.79-4-4-4z"
    ),
    "dark_mode": (
        "M12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36a5.39 5.39 0 0 1-4.4 2.26 5.4 5.4 0 0 1-3.14-9.8"
        "c-.44-.06-.9-.1-1.36-.1z"
    ),
    "light_mode": (
        "M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10zM2 13h2a1 1 0 0 0 0-2H2a1 1 0 0 0 0 2zm18 0h2a1 1 0 0 0 0-2"
        "h-2a1 1 0 0 0 0 2zM11 2v2a1 1 0 0 0 2 0V2a1 1 0 0 0-2 0zm0 18v2a1 1 0 0 0 2 0v-2a1 1 0 0 0-2 0z"
        "M5.99 4.58a1 1 0 0 0-1.41 1.41l1.06 1.06a1 1 0 0 0 1.41-1.41L5.99 4.58zm12.37 12.37a1 1 0 0 0"
        "-1.41 1.41l1.06 1.06a1 1 0 0 0 1.41-1.41l-1.06-1.06zm1.06-10.96a1 1 0 0 0-1.41-1.41l-1.06 1.06"
        "a1 1 0 0 0 1.41 1.41l1.06-1.06zM7.05 18.36a1 1 0 0 0-1.41-1.41l-1.06 1.06a1 1 0 0 0 1.41 1.41"
        "l1.06-1.06z"
    ),
    "menu": "M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z",
    "arrow_back": "M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z",
    "arrow_forward": "M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z",
    "unfold_more": (
        "M12 5.83 15.17 9l1.41-1.41L12 3 7.41 7.59 8.83 9 12 5.83zm0 12.34L8.83 15l-1.41 1.41L12 21"
        "l4.59-4.59L15.17 15 12 18.17z"
    ),
}


def icone(nome: str, tamanho: int = 16) -> str:
    return (
        f'<svg class="fz-icon" width="{tamanho}" height="{tamanho}" viewBox="0 0 24 24" '
        f'fill="currentColor" aria-hidden="true"><path d="{_ICONES[nome]}"/></svg>'
    )


def _mascara(nome: str) -> str:
    """O ícone como data URI, para `mask-image` no glifo de ordenação da tabela."""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
        f"<path d='{_ICONES[nome]}'/></svg>"
    )
    return 'url("data:image/svg+xml,' + svg.replace("#", "%23").replace('"', "'") + '")'


@lru_cache(maxsize=1)
def wordmark() -> str:
    """Wordmark da Uncover monocromático, na cor do texto (regra do Menu Header)."""
    svg = (ASSETS / "logo-uncover.svg").read_text(encoding="utf-8")
    return svg.replace('fill="#F3F4F6"', 'fill="currentColor"').replace(
        "<svg ", '<svg class="fz-wordmark" role="img" aria-label="Uncover" ', 1
    ).replace(' aria-hidden="true"', "", 1)


# ---------------------------------------------------------------------------
# CSS dos componentes
# ---------------------------------------------------------------------------

_COMPONENTES = """
*,*::before,*::after{box-sizing:border-box}
body{margin:0;background:var(--layout-bg);color:var(--layout-content-default);
  font:400 14px/24px var(--font-ui);letter-spacing:.2px;-webkit-font-smoothing:antialiased}
.fz-app{display:flex;align-items:flex-start;min-height:100vh}
.fz-main{flex:1 1 auto;min-width:0}
.fz-page{max-width:1280px;margin:0 auto;padding:48px 48px 96px}
@media (max-width:767px){.fz-page{padding:32px 24px 64px}}

/* Menu Sidebar — mapeamento do deck; plano, o degrau tonal separa */
.fz-sb{position:sticky;top:0;height:100vh;width:280px;flex:0 0 280px;display:flex;flex-direction:column;gap:24px;
  padding:24px;background:var(--layout-surface-default);overflow-x:hidden;overflow-y:auto;z-index:40}
:root[data-sb="fechada"] .fz-sb{width:64px;flex-basis:64px;padding:24px 12px}
.fz-sb-topo{display:flex;align-items:center;justify-content:space-between;gap:8px;min-height:36px}
.fz-sb-acoes{display:flex;gap:4px}
.fz-sb-logo{color:var(--layout-content-default);display:flex}
.fz-sb-logo .fz-wordmark{height:19px}
:root[data-sb="fechada"] .fz-sb-topo,:root[data-sb="fechada"] .fz-sb-acoes{flex-direction:column;gap:4px}
:root[data-sb="fechada"] [data-fz-sb-toggle]{order:-1}
:root[data-sb="fechada"] .fz-sb-logo,:root[data-sb="fechada"] .fz-sb-ctx,
:root[data-sb="fechada"] .fz-sb-rotulo,:root[data-sb="fechada"] .fz-sb-item .rot{display:none}
.fz-sb-ctx{display:flex;flex-direction:column;gap:4px}
.fz-sb-ctx-over{font:600 12px/12px var(--font-ui);letter-spacing:.48px;text-transform:uppercase;color:var(--layout-content-subtle)}
.fz-sb-ctx-title{font:400 20px/32px var(--font-ui);letter-spacing:.1px;color:var(--layout-content-default)}
.fz-sb-ctx-line{font:400 12px/16px var(--font-ui);letter-spacing:.4px;color:var(--layout-content-subtle);overflow-wrap:anywhere}
.fz-divider{height:1px;background:var(--layout-stroke-subtle);flex:0 0 1px}
.fz-sb-lista{display:flex;flex-direction:column;gap:2px}
.fz-sb-grupo{display:flex;flex-direction:column;gap:2px}
.fz-sb-grupo+.fz-sb-grupo{margin-top:16px}
.fz-sb-rotulo{font:600 12px/12px var(--font-ui);letter-spacing:.48px;text-transform:uppercase;color:var(--layout-content-subtle);margin:0 0 12px}
.fz-sb-item{position:relative;display:flex;align-items:center;gap:12px;width:100%;height:44px;padding:0 12px;border:0;
  border-radius:8px;background:transparent;cursor:pointer;text-align:left;color:var(--layout-content-subtle);
  font:400 14px/14px var(--font-ui);letter-spacing:.4px}
.fz-sb-item .num{font:600 12px/12px var(--font-ui);letter-spacing:.48px;font-variant-numeric:tabular-nums;flex:none;white-space:pre}
.fz-sb-item .rot{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-variant-caps:small-caps}
.fz-sb-item .fz-icon{width:16px;height:16px}
:root[data-sb="fechada"] .fz-sb-item{justify-content:center;padding:0}
.fz-sb-item.sel{background:var(--layout-surface-accent-subtle);color:var(--layout-content-accent)}
.fz-sb-item.sel::before{content:"";position:absolute;left:0;top:8px;bottom:8px;width:4px;border-radius:4px;background:var(--layout-surface-accent)}
.fz-sb-pe{margin-top:auto;display:flex;flex-direction:column;gap:4px}
:root[data-sb="fechada"] .fz-sb-item .num{font-size:11px;letter-spacing:0}

/* blocos */
.fz-bloco{display:none}
.fz-bloco.ativo,:root[data-tudo] .fz-bloco{display:block}
:root[data-tudo] .fz-bloco+.fz-bloco,:root[data-tudo] .fz-fpainel .fz-bloco:first-child{margin-top:64px}
.fz-bloco-head{margin:0 0 24px}
.fz-bloco-pe{display:flex;justify-content:space-between;gap:16px;margin-top:32px;padding-top:24px;border-top:1px solid var(--layout-stroke-subtle)}
:root[data-tudo] .fz-bloco-pe{display:none}
.fz-btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;height:32px;padding:4px 12px;border:0;
  border-radius:8px;cursor:pointer;font:600 14px/14px var(--font-ui);letter-spacing:.56px;font-variant-caps:small-caps}
.fz-btn .fz-icon{width:16px;height:16px}
.fz-btn-secondary{background:var(--layout-surface-subtle);color:var(--layout-content-default)}
.fz-btn .num{font-variant-caps:normal;font-variant-numeric:tabular-nums}
/* pé do bloco: dois botões na mesma linha até em 390px; o nome encolhe com reticências */
.fz-bloco-pe .fz-btn{min-width:0;max-width:calc(50% - 8px);white-space:nowrap}
.fz-bloco-pe .fz-btn>span:not(.num){min-width:0;overflow:hidden;text-overflow:ellipsis}
@media print{
  .fz-sb,.fz-bloco-pe,.fz-scope{display:none!important}
  .fz-bloco{display:block!important;break-inside:auto}
  .fz-bloco+.fz-bloco{margin-top:48px}
}
.fz-code{font-variant-numeric:tabular-nums;word-break:break-all}

/* scrim de hover/pressed: sobre a superfície, nunca troca a cor de fundo */
.fz-scrim{position:relative;isolation:isolate}
.fz-scrim::after{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;
  background:var(--overlay-hover);opacity:0;mix-blend-mode:var(--overlay-blend)}
.fz-scrim:hover::after{opacity:.32}
.fz-scrim:active::after{background:var(--overlay-pressed);opacity:.32}

/* Icon Button ghost (tema, colapsar) */
.fz-icon-btn{display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;
  padding:8px;border:0;border-radius:8px;background:transparent;color:var(--layout-content-subtle);cursor:pointer}
.fz-icon-btn .fz-icon{width:20px;height:20px}
.fz-icon{display:block;flex:0 0 auto}

/* Text Display */
.fz-td{display:flex;flex-direction:column;gap:4px}
.fz-td-over{font:600 14px/22.4px var(--font-ui);letter-spacing:.35px;text-transform:uppercase;color:var(--layout-content-subtle)}
.fz-td-over-bloco{text-transform:none;font-variant-caps:small-caps}
.fz-td-over-bloco .num{font-variant-caps:normal;font-variant-numeric:tabular-nums}
.fz-td-title{margin:0;color:var(--layout-content-default)}
.fz-td-xlarge .fz-td-title{font:400 40px/48px var(--font-display);letter-spacing:.1px}
.fz-td-large .fz-td-title{font:400 32px/40px var(--font-display);letter-spacing:.1px}
.fz-td-medium .fz-td-title{font:400 24px/32px var(--font-ui);letter-spacing:.1px}
.fz-td-small .fz-td-title{font:400 20px/32px var(--font-ui);letter-spacing:.1px}
.fz-td-sub{margin:0;font:400 16px/24px var(--font-ui);letter-spacing:.1px;color:var(--layout-content-subtle)}
.fz-td-small .fz-td-sub{font:400 14px/22.4px var(--font-ui);letter-spacing:.14px}

/* Tile Big Number */
.fz-tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}
.fz-tiles-sm{grid-template-columns:repeat(auto-fill,minmax(170px,1fr))}
.fz-tiles+*{margin-top:16px}
.fz-section .fz-tile{background:var(--layout-surface-subtle)}
.fz-tile{display:flex;gap:12px;padding:12px;border-radius:8px;background:var(--layout-surface-default)}
.fz-tile-bar{width:4px;border-radius:4px;flex:0 0 4px}
.fz-tile-body{display:flex;flex-direction:column;gap:12px;flex:1 1 auto;min-width:0}
.fz-tile-top{display:flex;flex-direction:column;gap:4px}
.fz-tile-bottom{display:flex;flex-direction:column;gap:8px}
.fz-tile-over{font:600 12px/12px var(--font-ui);letter-spacing:.48px;text-transform:uppercase;color:var(--layout-content-subtle)}
.fz-tile-sub{font:400 14px/14px var(--font-ui);letter-spacing:.4px;color:var(--layout-content-default)}
.fz-tile-value{font:400 32px/48px var(--font-ui);letter-spacing:.1px;font-variant-numeric:tabular-nums;
  color:var(--layout-content-default);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fz-tile-desc{display:flex;align-items:center;gap:4px;height:20px}
.fz-tile .fz-help{align-self:flex-start}

/* Ajuda: botão ? + popover único da página (ver docstring) */
.fz-help{display:inline-flex}
.fz-help-btn{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;
  padding:0;border:0;border-radius:999px;background:transparent;color:var(--layout-content-subtle);cursor:pointer}
.fz-help-btn[aria-expanded="true"]{color:var(--layout-content-accent)}
.fz-help-trigger{display:inline-flex;align-items:center;padding:0;margin:0;border:0;background:transparent;
  color:inherit;font:inherit;letter-spacing:inherit;cursor:pointer;border-radius:999px}
.fz-help-trigger[aria-expanded="true"] .fz-status{box-shadow:inset 0 0 0 1px currentColor}
th .fz-help{margin-left:4px;vertical-align:middle}
.fz-pop{position:fixed;z-index:100;max-width:320px;padding:12px 16px;border-radius:8px;
  background:var(--layout-surface-accent-subtle);color:var(--layout-content-default);
  border-left:4px solid var(--layout-stroke-accent);box-shadow:0 12px 64px 0 var(--fz-shadow);
  font:400 14px/24px var(--font-ui);letter-spacing:.2px;text-align:left;white-space:normal}
.fz-pop[hidden]{display:none}
.fz-pop p{margin:0 0 8px}
.fz-pop p:last-child,.fz-pop ul:last-child{margin-bottom:0}
.fz-pop ul{margin:0 0 8px;padding-left:18px}
.fz-pop b{font-weight:600}

/* Section */
.fz-section{background:var(--layout-surface-default);border-radius:8px;margin:0 0 32px}
.fz-section-header{display:flex;align-items:center;justify-content:space-between;height:56px;padding:8px 16px}
.fz-section-over{font:600 14px/14px var(--font-ui);letter-spacing:.56px;text-transform:uppercase;color:var(--layout-content-subtle)}
.fz-section-toolbar{display:flex;align-items:flex-start;gap:8px;padding:4px 8px}
.fz-section-titles{flex:1 1 auto;min-width:0}
.fz-section-title{margin:0;font:400 24px/32px var(--font-ui);letter-spacing:.1px;color:var(--layout-content-default)}
.fz-section-desc{margin:0;font:400 14px/16px var(--font-ui);letter-spacing:.4px;color:var(--layout-content-subtle)}
.fz-section-body{display:flex;flex-direction:column;gap:16px;padding:16px;min-width:0}
.fz-grid-2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px;align-items:start}
@media (max-width:1023px){.fz-grid-2{grid-template-columns:1fr}}
.fz-block{min-width:0}
.fz-block .js-plotly-plot,.fz-block .plot-container{max-width:100%}

/* Data Table — sobre as classes que o JS do cv_report ordena */
.sortable-table{min-width:0}
.table-wrap{border:1px solid var(--layout-stroke-default);border-radius:8px;overflow:auto}
table[data-sortable]{width:100%;border-collapse:collapse}
table[data-sortable] th{height:48px;padding:0 16px;background:var(--layout-surface-subtle);
  border-bottom:1px solid var(--layout-stroke-deep);text-align:left;white-space:nowrap;cursor:pointer;
  font:600 14px/14px var(--font-ui);letter-spacing:.56px;color:var(--layout-content-subtle)}
table[data-sortable] th>span:not(.fz-help){display:inline-flex;align-items:center;gap:8px}
table[data-sortable] th[data-sort-col]>span:not(.fz-help)::after{content:"";width:16px;height:16px;background:currentColor;
  -webkit-mask:var(--fz-sort) center/16px no-repeat;mask:var(--fz-sort) center/16px no-repeat}
table[data-sortable] th{--fz-sort:%UNFOLD%}
table[data-sortable] th.sort-asc{--fz-sort:%UP%}
table[data-sortable] th.sort-desc{--fz-sort:%DOWN%}
table[data-sortable] td{height:48px;padding:0 16px;background:var(--layout-surface-default);white-space:nowrap;
  border-bottom:1px solid var(--layout-stroke-deep);
  font:400 14px/16px var(--font-ui);letter-spacing:.4px;color:var(--layout-content-default)}
table[data-sortable] tbody tr:last-child td{border-bottom:0}
table[data-sortable] .num{text-align:right;font-variant-numeric:tabular-nums}
table[data-sortable] th.num>span:not(.fz-help){flex-direction:row-reverse}
tr[data-midia]{cursor:pointer}
.table-caption{margin-top:8px;font:400 12px/16px var(--font-ui);letter-spacing:.4px;color:var(--layout-content-subtle)}
.fz-cell{display:inline-flex;align-items:center;gap:8px}

/* Filter / Filter Group */
.fz-filter,.tab-btn{display:inline-flex;align-items:center;gap:8px;height:32px;padding:4px 12px;
  border:0;border-radius:8px;font:600 14px/14px var(--font-ui);letter-spacing:.56px;cursor:pointer;
  background:var(--layout-surface-subtle);color:var(--layout-content-default);box-shadow:0 1.5px 0 0 var(--fz-shadow)}
.fz-filter-active,.tab-btn.active{background:var(--layout-surface-accent-subtle);color:var(--layout-content-accent);
  border:1px solid var(--layout-stroke-accent)}
.fz-filter-select{position:relative;padding:0;gap:0}
.fz-filter select{appearance:none;-webkit-appearance:none;border:0;margin:0;background:transparent;align-self:stretch;
  padding:0 36px 0 12px;color:inherit;font:inherit;letter-spacing:inherit;cursor:pointer;border-radius:inherit}
.fz-filter select:focus:not(:focus-visible){outline:none}
.fz-filter-select>.fz-icon{position:absolute;right:12px;top:50%;margin-top:-8px;pointer-events:none}
.fz-filter select option{background:var(--layout-surface-subtle);color:var(--layout-content-default)}
.fz-filter-group{display:flex;align-items:center;gap:20px;min-height:36px;flex-wrap:wrap}
.fz-filter-item{display:flex;align-items:center;gap:8px;min-width:0;max-width:100%}
.fz-filter-item .fz-filter{min-width:0;max-width:100%}
.fz-filter select{min-width:0;max-width:100%;text-overflow:ellipsis}
.fz-filter-caption{font:400 12px/12px var(--font-ui);letter-spacing:.4px;color:var(--layout-content-subtle)}
.fz-scope{position:sticky;top:0;z-index:30;background:var(--layout-bg);padding:12px 0;margin:0 0 24px}
.tab-bar{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 16px}
.tab-panel{display:none}
.tab-panel.active{display:block}

/* Tag Trend e pílula de status */
.fz-trend,.fz-status{display:inline-flex;align-items:center;gap:4px;height:20px;padding:4px 8px;border-radius:999px;
  font:400 12px/12px var(--font-ui);letter-spacing:.4px;font-variant-numeric:tabular-nums;white-space:nowrap}
.fz-trend .fz-icon{width:12px;height:12px}
.fz-trend-positive{background:var(--trend-positive-surface);color:var(--trend-positive-content)}
.fz-trend-negative{background:var(--trend-negative-surface);color:var(--trend-negative-content)}
.fz-trend-neutral{background:var(--trend-neutral-surface);color:var(--trend-neutral-content)}
.fz-status{font-weight:600;letter-spacing:.48px}
.fz-status-ok{background:var(--success-surface-subtle);color:var(--success-content)}
.fz-status-fail{background:var(--error-surface-subtle);color:var(--error-content)}
.fz-status-warn{background:var(--warning-surface-subtle);color:var(--warning-content)}
.fz-status-neutral{background:var(--layout-surface-subtle);color:var(--layout-content-subtle)}
.fz-swatch{display:inline-block;width:8px;height:8px;border-radius:999px;flex:0 0 8px}

/* Accordion */
.fz-acc{border-radius:8px;overflow:hidden}
.fz-acc>summary{display:flex;align-items:center;justify-content:space-between;gap:8px;min-height:48px;padding:0 16px;
  list-style:none;cursor:pointer;background:var(--layout-surface-subtle);
  font:600 14px/14px var(--font-ui);letter-spacing:.56px;color:var(--layout-content-default)}
.fz-acc>summary::-webkit-details-marker{display:none}
.fz-acc[open]>summary{background:var(--layout-surface-default);border-bottom:1px solid var(--layout-stroke-subtle)}
.fz-acc[open]>summary .fz-icon{transform:rotate(180deg)}
.fz-acc-body{display:flex;flex-direction:column;gap:16px;padding:16px 0 0}

/* notas */
.fz-note{margin:0;font:400 12px/16px var(--font-ui);letter-spacing:.4px;color:var(--layout-content-subtle)}
.fz-notes{margin:0;padding-left:18px;font:400 14px/24px var(--font-ui);letter-spacing:.2px;color:var(--layout-content-subtle)}
.fz-notes li{margin:4px 0}
.fz-notes b{font-weight:600;color:var(--layout-content-default)}

/* painel de filtro (mídia, fold…) */
.fz-fpainel{display:none}
.fz-fpainel.ativo{display:block}
.fz-stack{display:flex;flex-direction:column;gap:16px}
"""


@lru_cache(maxsize=1)
def css() -> str:
    comp = (
        _COMPONENTES.replace("%UNFOLD%", _mascara("unfold_more"))
        .replace("%UP%", _mascara("arrow_upward"))
        .replace("%DOWN%", _mascara("arrow_downward"))
    )
    return _fontes_css() + "\n" + _tokens_css() + comp


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------

def layout_figura(fig, height: int = 400):
    """Layout sem cor de chrome — o `THEME_JS` preenche pelo tema ativo."""
    fig.update_layout(
        height=height,
        autosize=True,
        width=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=60, r=24, t=32, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Marcação
# ---------------------------------------------------------------------------

def _esc(v) -> str:
    return _html.escape("" if v is None else str(v))


def rotulo_cliente() -> str:
    """Sem globals.yml neste repo: o overline do sidebar vem do chamador."""
    return ""


def num_bloco(n: int) -> str:
    return f"[ {n:02d} ]"


def sidebar(contexto: list[str], grupos: list[tuple[str, list[tuple[str, int, str]]]],
            titulo: str, overline: str = "") -> str:
    """Menu Sidebar. ``grupos`` é ``[(rótulo, [(id_bloco, número, nome)])]`` na
    ordem de leitura — a mesma ordem do voltar/avançar. ``contexto`` são linhas
    curtas (run, data), escapadas aqui."""
    over = f'<span class="fz-sb-ctx-over">{_esc(overline)}</span>' if overline else ""
    linhas = "".join(f'<span class="fz-sb-ctx-line">{_esc(t)}</span>' for t in contexto)
    lista = "".join(
        '<div class="fz-sb-grupo">'
        + (f'<span class="fz-sb-rotulo">{_esc(rot)}</span>' if rot else "")
        + "".join(
            f'<button type="button" class="fz-sb-item fz-scrim" data-fz-ir="{_esc(bid)}" '
            f'data-num="{_esc(num_bloco(n))}" data-nome="{_esc(nome)}" title="{_esc(nome)}">'
            f'<span class="num">{_esc(num_bloco(n))}</span><span class="rot">{_esc(nome)}</span></button>'
            for bid, n, nome in itens
        )
        + "</div>"
        for rot, itens in grupos
    )
    return (
        '<aside class="fz-sb" aria-label="Blocos do relatório">'
        '<div class="fz-sb-topo">'
        f'<span class="fz-sb-logo">{wordmark()}</span>'
        '<span class="fz-sb-acoes">'
        '<button type="button" class="fz-icon-btn fz-scrim" data-fz-tema-toggle '
        'aria-label="Alternar tema claro/escuro"></button>'
        '<button type="button" class="fz-icon-btn fz-scrim" data-fz-sb-toggle '
        f'aria-label="Recolher ou expandir o menu">{icone("menu", 20)}</button>'
        "</span></div>"
        f'<div class="fz-sb-ctx">{over}<span class="fz-sb-ctx-title">{_esc(titulo)}</span>{linhas}</div>'
        '<div class="fz-divider"></div>'
        f'<nav class="fz-sb-lista">{lista}</nav>'
        '<div class="fz-sb-pe"><div class="fz-divider"></div>'
        '<button type="button" class="fz-sb-item fz-scrim" data-fz-tudo title="Mostrar tudo">'
        f'<span class="num">{icone("unfold_more", 16)}</span><span class="rot">Mostrar tudo</span></button>'
        "</div></aside>"
    )


def bloco(bid: str, conteudo: str, escopo: str = "geral") -> str:
    """Um bloco navegável. O pé (voltar/avançar) é montado pelo `NAV_JS` a partir
    da ordem do sidebar. ``escopo`` liga as barras de filtro daquele escopo."""
    return (
        f'<div class="fz-bloco" data-bloco="{_esc(bid)}" data-escopo="{_esc(escopo)}">'
        f'{conteudo}<nav class="fz-bloco-pe" data-fz-pe aria-label="Navegar entre blocos"></nav></div>'
    )


def blocos(itens: list[dict], inicio: int = 0) -> tuple[list, str]:
    """Numera e monta os blocos. Cada item: ``bid``, ``nome`` (rótulo do
    sidebar), ``titulo``, ``corpo`` e, opcionais, ``grupo``, ``sub``
    (``sub_html=True`` para HTML), ``escopo``, ``size``. Item com ``corpo`` vazio
    sai, e a numeração segue sem buraco. Devolve ``(grupos, html)`` — ``grupos``
    no formato do `sidebar`."""
    grupos: list[tuple[str, list]] = []
    html = []
    n = inicio
    for it in itens:
        if not it.get("corpo"):
            continue
        g = it.get("grupo", "")
        if not grupos or grupos[-1][0] != g:
            grupos.append((g, []))
        grupos[-1][1].append((it["bid"], n, it["nome"]))
        head = bloco_head(n, it["nome"], it["titulo"], it.get("sub", ""),
                          size=it.get("size", "large"), subtitle_html=it.get("sub_html", False))
        html.append(bloco(it["bid"], head + it["corpo"], it.get("escopo", "geral")))
        n += 1
    return grupos, "".join(html)


def barra_filtro(nome: str, caption: str, opcoes: list[tuple[str, str]], escopo: str) -> str:
    """Barra sticky do topo com um filtro. ``opcoes`` = ``[(valor, rótulo)]``;
    aparece só nos blocos de ``escopo`` (lista separada por espaço)."""
    opts = "".join(
        f'<option value="{_esc(v)}"{" selected" if i == 0 else ""}>{_esc(r)}</option>'
        for i, (v, r) in enumerate(opcoes)
    )
    return (
        f'<div class="fz-scope" data-fz-escopo="{_esc(escopo)}" hidden>'
        + filter_select(caption, f"fz-filtro-{nome}", opts, f'data-fz-filtro="{_esc(nome)}"')
        + "</div>"
    )


def painel_filtro(nome: str, valor: str, conteudo: str, ativo: bool = False) -> str:
    """Conteúdo de um valor do filtro ``nome``; só o do valor ativo aparece."""
    return (
        f'<div class="fz-fpainel{" ativo" if ativo else ""}" data-fz-filtro-painel="{_esc(nome)}" '
        f'data-valor="{_esc(valor)}">{conteudo}</div>'
    )


def bloco_over(n: int, nome: str) -> str:
    """Overline de bloco: ``[ NN ] Nome`` em versalete, o mesmo rótulo do sidebar."""
    return (
        f'<div class="fz-td-over fz-td-over-bloco"><span class="num">{_esc(num_bloco(n))}</span> '
        f"<span>{_esc(nome)}</span></div>"
    )


def bloco_head(n: int, nome: str, title: str, subtitle: str = "", size: str = "large",
               subtitle_html: bool = False) -> str:
    """Cabeçalho do bloco: overline ``[ NN ] Nome`` + título (nos blocos de mídia,
    o nome da mídia do filtro) + descrição."""
    sub = ""
    if subtitle:
        sub = f'<p class="fz-td-sub">{subtitle if subtitle_html else _esc(subtitle)}</p>'
    tag = "h1" if size == "xlarge" else "h2"
    return (
        f'<div class="fz-td fz-td-{size} fz-bloco-head">{bloco_over(n, nome)}'
        f'<{tag} class="fz-td-title">{_esc(title)}</{tag}>{sub}</div>'
    )


def text_display(title: str, overline: str = "", subtitle: str = "", size: str = "xlarge",
                 subtitle_html: bool = False, cls: str = "") -> str:
    """Bloco de título. ``xlarge``/``large`` usam Machina; ``medium``/``small``, Inter."""
    tag = "h1" if size == "xlarge" else "h2"
    over = f'<div class="fz-td-over">{_esc(overline)}</div>' if overline else ""
    sub = ""
    if subtitle:
        sub = f'<p class="fz-td-sub">{subtitle if subtitle_html else _esc(subtitle)}</p>'
    return (
        f'<div class="fz-td fz-td-{size} {cls}">{over}'
        f'<{tag} class="fz-td-title">{_esc(title)}</{tag}>{sub}</div>'
    )


def ajuda(conteudo_html: str, rotulo: str = "Ajuda") -> str:
    """Botão ``?`` que abre o popover. ``conteudo_html`` entra como HTML — escape
    todo dado antes de montar."""
    return (
        '<span class="fz-help"><button type="button" class="fz-help-btn" data-fz-help '
        f'aria-expanded="false" aria-label="{_esc(rotulo)}">{icone("help_outline", 16)}</button>'
        f"<template>{conteudo_html}</template></span>"
    )


def tile(value: str, overline: str, subtitle: str = "", desc: str = "", hint: str = "",
         serie_n: int | None = None, hint_html: str = "", cor_barra: str = "") -> str:
    """Tile Big Number. ``serie_n`` liga a series bar — só quando o tile resume uma
    série de gráfico; ``desc`` é o HTML do slot (uma Tag Trend ou pílula).
    ``hint`` é texto puro (escapado); ``hint_html``, HTML pronto para o popover."""
    # a barra é a legenda do tile: a cor da série que ele resume no gráfico
    # (``serie_n``) ou, quando o gráfico colore por outra regra, a cor dada
    cor_b = serie_var(serie_n) if serie_n else cor_barra
    bar = f'<span class="fz-tile-bar" style="background:{_esc(cor_b)}"></span>' if cor_b else ""
    sub = f'<div class="fz-tile-sub">{_esc(subtitle)}</div>' if subtitle else ""
    desc_html = f'<div class="fz-tile-desc">{desc}</div>' if desc else ""
    conteudo = hint_html or (_esc(hint) if hint else "")
    help_html = ajuda(conteudo, f"Ajuda: {subtitle or overline}") if conteudo else ""
    return (
        f'<div class="fz-tile">{bar}<div class="fz-tile-body">'
        f'<div class="fz-tile-top"><div class="fz-tile-over">{_esc(overline)}</div>{sub}</div>'
        f'<div class="fz-tile-bottom"><div class="fz-tile-value" title="{_esc(value)}">{_esc(value)}</div>'
        f"{desc_html}</div></div>{help_html}</div>"
    )


def tiles(items: list[str], compacto: bool = False) -> str:
    """Linha de tiles. Dentro de uma Section o tile sobe para `surface-subtle`
    (o degrau do painel aninhado), senão teria a cor da própria Section."""
    cls = "fz-tiles fz-tiles-sm" if compacto else "fz-tiles"
    return f'<div class="{cls}">{"".join(items)}</div>'


def section(overline: str, title: str, description: str, body: str, anchor: str) -> str:
    return (
        f'<section class="fz-section" id="{_esc(anchor)}">'
        f'<div class="fz-section-header"><span class="fz-section-over">{_esc(overline)}</span></div>'
        '<div class="fz-section-toolbar"><div class="fz-section-titles">'
        f'<h2 class="fz-section-title">{_esc(title)}</h2>'
        f'<p class="fz-section-desc">{_esc(description)}</p></div></div>'
        f'<div class="fz-section-body">{body}</div></section>'
    )


def painel(body: str, anchor: str = "") -> str:
    """Section sem header nem toolbar (``header``/``toolbar`` = false): o título
    já está no cabeçalho do bloco."""
    ident = f' id="{_esc(anchor)}"' if anchor else ""
    return f'<section class="fz-section"{ident}><div class="fz-section-body">{body}</div></section>'


def block(content: str) -> str:
    return f'<div class="fz-block">{content}</div>'


def tag_trend(text: str, status: str) -> str:
    """Tag Trend: ``positive`` · ``negative`` · ``neutral``."""
    icon = {"positive": "arrow_upward", "negative": "arrow_downward", "neutral": "remove"}[status]
    return f'<span class="fz-trend fz-trend-{status}">{icone(icon, 12)}<span>{_esc(text)}</span></span>'


def status_pill(text: str, status: str, title: str = "", ajuda_html: str = "") -> str:
    """Pílula de status (composta, ver docstring do módulo): ``ok`` · ``warn`` ·
    ``fail`` · ``neutral``. Com ``ajuda_html`` (ou ``title``, texto puro) a própria
    pílula abre o popover de ajuda — sem ``?`` ao lado, para caber em célula."""
    conteudo = ajuda_html or (_esc(title) if title else "")
    if not conteudo:
        return f'<span class="fz-status fz-status-{status}">{_esc(text)}</span>'
    return ajuda_em(
        conteudo, f'<span class="fz-status fz-status-{status}">{_esc(text)}</span>',
        rotulo=f"Detalhe: {text}",
    )


def ajuda_em(conteudo_html: str, gatilho_html: str, rotulo: str = "Ajuda") -> str:
    """Popover de ajuda aberto por um gatilho qualquer (pílula, ícone, texto)."""
    return (
        '<span class="fz-help"><button type="button" class="fz-help-trigger" data-fz-help '
        f'aria-expanded="false" aria-label="{_esc(rotulo)}">{gatilho_html}</button>'
        f"<template>{conteudo_html}</template></span>"
    )


def swatch(serie_n: int) -> str:
    return f'<span class="fz-swatch" style="background:{serie_var(serie_n)}"></span>'


def filter_select(caption: str, select_id: str, options: str, attrs: str = "") -> str:
    """Filter Group com um item: caption + chip `active` hospedando um ``<select>``."""
    return (
        '<div class="fz-filter-group"><div class="fz-filter-item">'
        f'<label class="fz-filter-caption" for="{_esc(select_id)}">{_esc(caption)}</label>'
        '<span class="fz-filter fz-filter-active fz-filter-select fz-scrim">'
        f'<select id="{_esc(select_id)}" {attrs}>{options}</select>'
        f'{icone("expand_more", 16)}</span></div></div>'
    )


def accordion(label: str, content: str) -> str:
    return (
        f'<details class="fz-acc"><summary class="fz-scrim"><span>{_esc(label)}</span>'
        f'{icone("expand_more", 16)}</summary><div class="fz-acc-body">{content}</div></details>'
    )


# ---------------------------------------------------------------------------
# JS
# ---------------------------------------------------------------------------

# Roda no <head>, antes do primeiro paint: tema e sidebar recolhido sem piscar.
# Sem escolha salva, o sidebar nasce recolhido abaixo de 768px e sempre em
# `embed` (iframe de notebook).
_TEMA_HEAD_JS = (
    "(function(){var r=document.documentElement,sb=null;try{var t=localStorage.getItem('fz-tema');"
    "if(t==='dark'||t==='light')r.setAttribute('data-tema',t);sb=localStorage.getItem('fz-sb');}catch(e){}"
    "if(sb==='fechada'||r.hasAttribute('data-fz-embed')||(sb!=='aberta'&&window.innerWidth<768))"
    "r.setAttribute('data-sb','fechada');})();"
)

# Motor sem aparência, comum a todos os relatórios: tokens e chrome do tema,
# registro de figuras (pool do `_payload`, montadas quando ficam visíveis),
# abas, tabela ordenável, seletor sincronizado por grupo e o gancho `FZ_REDRAW`,
# onde widgets que desenham com Plotly.react se registram para serem refeitos
# ao aparecer e ao trocar de tema. Tudo por delegação no document, para servir
# também conteúdo montado depois (painel em gzip). Exige o HYDRATE_JS antes.
MOTOR_JS = """
// ---- tokens e chrome -------------------------------------------------------
function fzToken(nome) { return getComputedStyle(document.documentElement).getPropertyValue(nome).trim(); }
var FZ_SENT = %SENT%;
function fzRgb(hex) {
  var h = hex.replace('#', '');
  if (h.length === 3) h = h.split('').map(function (c) { return c + c; }).join('');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
// cor sentinela (#fe00NN / rgba(254,0,N,a), ver cor() no Python) -> token do tema ativo
function fzCores(node) {
  if (typeof node === 'string') {
    var m = /^#fe00([0-9a-f]{2})$/i.exec(node);
    if (m) { var t = FZ_SENT[m[1].toLowerCase()]; return t ? fzToken(t) : node; }
    m = /^rgba\\(\\s*254\\s*,\\s*0\\s*,\\s*(\\d+)\\s*,\\s*([\\d.]+)\\s*\\)$/.exec(node);
    if (m) {
      var t2 = FZ_SENT[('0' + parseInt(m[1], 10).toString(16)).slice(-2)];
      if (t2) { var c = fzRgb(fzToken(t2)); return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + m[2] + ')'; }
    }
    return node;
  }
  if (Array.isArray(node)) {
    if (node.length && typeof node[0] === 'number') return node;
    return node.map(fzCores);
  }
  if (node && typeof node === 'object') {
    var o = {};
    for (var k in node) o[k] = fzCores(node[k]);
    return o;
  }
  return node;
}
// chrome do layout pelo tema ativo; cor de série não muda (é invariante de modo)
function fzChrome(layout) {
  layout = fzCores(layout || {});
  var texto = fzToken('--layout-content-subtle'), forte = fzToken('--layout-content-default');
  var grade = fzToken('--layout-stroke-subtle'), eixo = fzToken('--layout-stroke-default');
  var out = {};
  for (var k in layout) out[k] = layout[k];
  out.font = Object.assign({}, layout.font, { color: texto });
  Object.keys(layout).concat(['xaxis', 'yaxis']).forEach(function (ax) {
    if (!/^[xy]axis\\d*$/.test(ax)) return;
    out[ax] = Object.assign({}, layout[ax], { gridcolor: grade, zerolinecolor: eixo, linecolor: eixo });
  });
  out.hoverlabel = Object.assign({}, layout.hoverlabel, {
    bgcolor: fzToken('--layout-surface-subtle'), bordercolor: eixo,
    font: { color: forte, family: 'Inter, system-ui, sans-serif' } });
  out.legend = Object.assign({}, layout.legend, {
    font: Object.assign({}, (layout.legend || {}).font, { color: forte }) });
  if (layout.shapes) out.shapes = layout.shapes.map(function (s) {
    var line = Object.assign({}, s.line);
    if (!line.color) line.color = fzToken('--layout-stroke-strong');
    return Object.assign({}, s, { line: line });
  });
  if (layout.annotations) out.annotations = layout.annotations.map(function (a) {
    var font = Object.assign({}, a.font);
    if (!font.color) font.color = texto;
    return Object.assign({}, a, { font: font });
  });
  return out;
}
window.fzToken = fzToken; window.fzCores = fzCores; window.fzChrome = fzChrome;

// ---- redesenho de widgets ---------------------------------------------------
window.FZ_REDRAW = window.FZ_REDRAW || [];
function fzRedraw(root) {
  window.FZ_REDRAW.forEach(function (fn) { fn(root || document); });
}

// ---- figuras ----------------------------------------------------------------
var FZ_FIGS = {};
function fzAddFigs(pack) {
  if (!pack || !pack.specs) return;
  var base = payloadHydrate(pack, pack.base);
  for (var fid in pack.specs) FZ_FIGS[fid] = { spec: pack.specs[fid], pack: pack, base: base };
}
function fzRenderFig(el) {
  if (el.dataset.figReady || !window.Plotly) return;
  var entry = FZ_FIGS[el.dataset.fig];
  if (!entry) return;
  el.dataset.figReady = '1';
  el.style.height = '';  // a altura do layout assume o comando após o render
  var data = payloadHydrate(entry.pack, entry.spec.d);
  var layout = payloadMerge(entry.base, payloadHydrate(entry.pack, entry.spec.l));
  el.__fzSpec = { d: data, l: layout };  // com as sentinelas, para a troca de tema
  Plotly.newPlot(el, fzCores(data), fzChrome(layout), { displayModeBar: false, responsive: true });
}
document.querySelectorAll('script[data-fz-figs]').forEach(function (h) {
  try { fzAddFigs(JSON.parse(h.textContent)); } catch (e) {}
});
// FZ_LAZY: montadores de placeholder próprios do relatório (fn(root) -> true se
// montou algo). Quem guarda estado sobre uma figura (o limiar de atividade do EDA)
// escuta o 'fz:figs-rendered' para reaplicá-lo nas que acabaram de nascer.
window.FZ_LAZY = window.FZ_LAZY || [];
function resizePlotsIn(root) {
  if (!root) return;
  var novas = false;
  root.querySelectorAll('.pfig').forEach(function (el) {
    if (el.offsetParent === null || el.dataset.figReady) return;  // oculto: será montado ao abrir
    fzRenderFig(el);
    if (el.dataset.figReady) novas = true;
  });
  window.FZ_LAZY.forEach(function (fn) { try { if (fn(root)) novas = true; } catch (e) { console.error(e); } });
  if (novas) document.dispatchEvent(new CustomEvent('fz:figs-rendered'));
  if (!window.Plotly) return;
  root.querySelectorAll('.js-plotly-plot').forEach(function (plot) {
    if (plot.offsetParent === null) return;
    try { window.Plotly.Plots.resize(plot); } catch (e) {}
  });
}
function fzMostrar(root) { resizePlotsIn(root); fzRedraw(root); }
window.fzAddFigs = fzAddFigs; window.fzMostrar = fzMostrar; window.resizePlotsIn = resizePlotsIn;

// ---- abas (markup do _tabs) -------------------------------------------------
document.addEventListener('click', function (e) {
  var btn = e.target.closest ? e.target.closest('.tab-bar > .tab-btn[data-target]') : null;
  if (!btn) return;
  var widget = btn.closest('.tabs');
  var bar = widget.querySelector(':scope > .tab-bar');
  var panels = widget.querySelector(':scope > .tab-panels');
  bar.querySelectorAll(':scope > .tab-btn').forEach(function (b) { b.classList.toggle('active', b === btn); });
  panels.querySelectorAll(':scope > .tab-panel').forEach(function (p) { p.classList.remove('active'); });
  var panel = document.getElementById(btn.dataset.target);
  if (panel) { panel.classList.add('active'); fzMostrar(panel); }
});

// ---- seletor sincronizado por grupo ([data-window-*]) -----------------------
document.addEventListener('click', function (e) {
  var btn = e.target.closest ? e.target.closest('[data-window-btn]') : null;
  if (!btn) return;
  var group = btn.dataset.windowGroup, value = btn.dataset.windowBtn;
  document.querySelectorAll('[data-window-group="' + group + '"]').forEach(function (el) {
    if (el.hasAttribute('data-window-btn')) {
      el.classList.toggle('active', el.dataset.windowBtn === value);
    } else if (el.hasAttribute('data-window-panel')) {
      el.classList.toggle('active', el.dataset.windowPanel === value);
    }
  });
  document.querySelectorAll('[data-window-group="' + group + '"].active').forEach(function (el) {
    if (!el.hasAttribute('data-window-btn')) fzMostrar(el);
  });
});

// ---- tabela ordenável ([data-sortable]) -------------------------------------
// data-sort carrega o valor cru; sem ele, o texto exibido. Vazio vai sempre para
// o fim. Só o <tbody> é reordenado: a média fica no <tfoot>.
(function () {
  function sortKey(cell) {
    var raw = cell ? cell.getAttribute('data-sort') : '';
    if (raw === null) return { num: null, txt: cell.textContent.trim().toLowerCase() };
    if (raw === '') return { num: null, txt: '' };
    var n = parseFloat(raw);
    return isNaN(n) ? { num: null, txt: raw.trim().toLowerCase() } : { num: n, txt: raw.trim().toLowerCase() };
  }
  function linhas(tbody) { return Array.prototype.slice.call(tbody.querySelectorAll(':scope > tr')); }
  function legenda(root, tbody) {
    var cap = root && root.querySelector('[data-table-caption]');
    if (!cap) return;
    var n = linhas(tbody).filter(function (r) { return r.style.display !== 'none'; }).length;
    cap.textContent = cap.dataset.tableCaption.replace('{n}', n);
  }
  document.addEventListener('click', function (e) {
    var th = e.target.closest ? e.target.closest('table[data-sortable] th[data-sort-col]') : null;
    if (!th) return;
    var table = th.closest('table'), tbody = table.querySelector('tbody');
    if (!tbody) return;
    var col = parseInt(th.dataset.sortCol, 10);
    var asc = th.classList.contains('sort-desc');  // primeiro clique: descendente
    table.querySelectorAll('th').forEach(function (h) { h.classList.remove('sort-asc', 'sort-desc'); });
    th.classList.add(asc ? 'sort-asc' : 'sort-desc');
    var rows = linhas(tbody);
    rows.sort(function (a, b) {
      var ka = sortKey(a.querySelectorAll(':scope > td')[col]), kb = sortKey(b.querySelectorAll(':scope > td')[col]);
      var ea = ka.num === null && !ka.txt, eb = kb.num === null && !kb.txt;
      if (ea !== eb) return ea ? 1 : -1;
      var cmp = (ka.num !== null && kb.num !== null) ? ka.num - kb.num
        : ka.txt.localeCompare(kb.txt, 'pt-BR', { numeric: true });
      return asc ? cmp : -cmp;
    });
    rows.forEach(function (r) { tbody.appendChild(r); });
  });
  document.addEventListener('input', function (e) {
    var input = e.target.closest ? e.target.closest('[data-table-filter]') : null;
    if (!input) return;
    var root = input.closest('.sortable-table'), tbody = root && root.querySelector('table[data-sortable] tbody');
    if (!tbody) return;
    var q = input.value.trim().toLowerCase();
    linhas(tbody).forEach(function (r) {
      r.style.display = !q || r.textContent.toLowerCase().indexOf(q) !== -1 ? '' : 'none';
    });
    legenda(root, tbody);
  });
  // legenda com contagem e coluna numérica à direita (valor cru em data-sort)
  function fzInitTabelas(root) {
    (root || document).querySelectorAll('table[data-sortable]').forEach(function (t) {
      if (t.dataset.fzInit) return;
      t.dataset.fzInit = '1';
      var tbody = t.querySelector('tbody');
      if (tbody) legenda(t.closest('.sortable-table'), tbody);
      var primeira = t.querySelector('tbody > tr');
      if (!primeira) return;
      var tds = primeira.querySelectorAll(':scope > td');
      t.querySelectorAll('th[data-sort-col]').forEach(function (th) {
        var i = parseInt(th.dataset.sortCol, 10), td = tds[i];
        if (!td || !td.hasAttribute('data-sort') || isNaN(parseFloat(td.getAttribute('data-sort')))) return;
        th.classList.add('num');
        t.querySelectorAll('tbody > tr, tfoot > tr').forEach(function (tr) {
          var c = tr.querySelectorAll(':scope > td')[i];
          if (c) c.classList.add('num');
        });
      });
    });
  }
  window.fzInitTabelas = fzInitTabelas;
  fzInitTabelas(document);
})();

window.addEventListener('load', function () { fzMostrar(document); });
window.addEventListener('resize', function () { resizePlotsIn(document); });
"""


# Tema: o toggle troca `data-tema` e refaz as figuras com o chrome novo. As do
# registro são refeitas da especificação original (com as sentinelas); os
# widgets se refazem pelo FZ_REDRAW; o resto recebe só o chrome.
THEME_JS = """
(function () {
  var root = document.documentElement;
  var ICONES = { dark: %DARK%, light: %LIGHT% };

  function temaAtivo() {
    var t = root.getAttribute('data-tema');
    if (t === 'dark' || t === 'light') return t;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  // Relatório sem Plotly (o do otimizador desenha SVG à mão) também precisa saber
  // que o tema trocou: o evento `fz:tema` sai sempre, com ou sem Plotly.
  function repintar() {
    if (window.Plotly) {
      document.querySelectorAll('.js-plotly-plot').forEach(function (p) {
        if (p.__fzSpec) Plotly.react(p, fzCores(p.__fzSpec.d), fzChrome(p.__fzSpec.l));
        else if (!p.__fzOwn && p.layout) Plotly.relayout(p, fzChrome(p.layout));
      });
    }
    fzRedraw(document);
    document.dispatchEvent(new CustomEvent('fz:tema', { detail: temaAtivo() }));
  }

  var botao = document.querySelector('[data-fz-tema-toggle]');
  function iconeBotao() {
    if (!botao) return;
    // mostra o modo para onde o clique leva
    botao.innerHTML = temaAtivo() === 'dark' ? ICONES.light : ICONES.dark;
  }
  if (botao) botao.addEventListener('click', function () {
    var novo = temaAtivo() === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-tema', novo);
    try { localStorage.setItem('fz-tema', novo); } catch (e) {}
    iconeBotao();
    repintar();
  });
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var seguir = function () { if (!root.getAttribute('data-tema')) { iconeBotao(); repintar(); } };
    if (mq.addEventListener) mq.addEventListener('change', seguir); else if (mq.addListener) mq.addListener(seguir);
  }
  iconeBotao();
})();
"""


# Popover de ajuda. Um elemento só, em position:fixed no <body>: dentro do
# .table-wrap (overflow:auto) um popover absoluto seria cortado. Delegação no
# document para servir também conteúdo montado depois (painel em gzip). O clique
# é capturado no document para não chegar ao <th> (ordena) nem ao tr[data-midia]
# (troca a mídia).
HELP_JS = """
(function () {
  var pop = document.createElement('div');
  pop.className = 'fz-pop'; pop.id = 'fz-pop'; pop.setAttribute('role', 'tooltip'); pop.hidden = true;
  document.body.appendChild(pop);
  var atual = null, fixo = false, tAbrir = null, tFechar = null;

  function posicionar() {
    if (!atual) return;
    var r = atual.getBoundingClientRect(), vw = window.innerWidth, vh = window.innerHeight;
    var w = pop.offsetWidth, h = pop.offsetHeight;
    var left = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), vw - w - 8);
    // acima do ?; abaixo só quando não cabe em cima
    var top = r.top - h - 8;
    if (top < 8 && r.bottom + 8 + h <= vh - 8) top = r.bottom + 8;
    pop.style.left = left + 'px'; pop.style.top = top + 'px';
  }
  function abrir(btn, fixar) {
    clearTimeout(tAbrir); clearTimeout(tFechar);
    if (atual && atual !== btn) atual.setAttribute('aria-expanded', 'false');
    var tpl = btn.parentNode.querySelector('template');
    if (atual !== btn) pop.innerHTML = tpl ? tpl.innerHTML : '';
    atual = btn; fixo = !!fixar;
    pop.setAttribute('role', fixo ? 'dialog' : 'tooltip');
    btn.setAttribute('aria-expanded', fixo ? 'true' : 'false');
    btn.setAttribute('aria-describedby', 'fz-pop');
    pop.hidden = false;
    posicionar();
  }
  function fechar() {
    clearTimeout(tAbrir); clearTimeout(tFechar);
    if (atual) { atual.setAttribute('aria-expanded', 'false'); atual.removeAttribute('aria-describedby'); }
    atual = null; fixo = false; pop.hidden = true;
  }
  function botao(el) { return el && el.closest ? el.closest('[data-fz-help]') : null; }

  document.addEventListener('click', function (e) {
    var btn = botao(e.target);
    if (btn) {
      e.preventDefault(); e.stopPropagation();
      if (atual === btn && fixo) fechar(); else abrir(btn, true);
      return;
    }
    if (fixo && !pop.contains(e.target)) fechar();
  }, true);

  document.addEventListener('mouseover', function (e) {
    var btn = botao(e.target);
    if (btn && !fixo) {
      clearTimeout(tFechar); clearTimeout(tAbrir);
      tAbrir = setTimeout(function () { abrir(btn, false); }, 300);
    } else if (pop.contains(e.target)) {
      clearTimeout(tFechar);
    }
  });
  document.addEventListener('mouseout', function (e) {
    if (fixo) return;
    if (botao(e.target) || pop.contains(e.target)) {
      var para = e.relatedTarget;
      if (para && (pop.contains(para) || botao(para) === atual)) return;
      clearTimeout(tAbrir);
      tFechar = setTimeout(function () { if (!fixo) fechar(); }, 150);
    }
  });
  document.addEventListener('focusin', function (e) {
    var btn = botao(e.target);
    if (btn && !fixo) abrir(btn, false);
  });
  document.addEventListener('focusout', function (e) {
    if (!fixo && botao(e.target)) fechar();
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && atual) { var b = atual; fechar(); b.focus(); }
  });
  window.addEventListener('resize', posicionar);
  window.addEventListener('scroll', function () { if (fixo) posicionar(); else fechar(); }, true);
})();
"""


# Navegação por blocos. Vem depois do MOTOR_JS (usa fzMostrar) e antes do JS do
# relatório, que pode ler e gravar parâmetros no hash via window.FZ_NAV.
NAV_JS = """
(function () {
  var root = document.documentElement;
  var itens = Array.prototype.slice.call(document.querySelectorAll('.fz-sb-lista [data-fz-ir]'));
  var ordem = itens.map(function (b) { return b.dataset.fzIr; });
  var atual = null;
  var ICON_BACK = %BACK%, ICON_FWD = %FWD%;

  function params() {
    var out = {};
    location.hash.replace(/^#/, '').split('&').forEach(function (kv) {
      if (!kv) return;
      var p = kv.split('=');
      out[decodeURIComponent(p[0])] = decodeURIComponent(p[1] || '');
    });
    return out;
  }
  function gravar(p, push) {
    var h = '#' + Object.keys(p).filter(function (k) { return p[k]; })
      .map(function (k) { return encodeURIComponent(k) + '=' + encodeURIComponent(p[k]); }).join('&');
    if (h === location.hash) return;
    if (push) history.pushState(null, '', h); else history.replaceState(null, '', h);
  }
  function setParam(k, v, push) { var p = params(); p[k] = v; gravar(p, push); }

  function item(id) { return itens.filter(function (b) { return b.dataset.fzIr === id; })[0]; }
  function botao(id, dir) {
    var it = item(id);
    if (!it) return '<span></span>';
    var rot = '<span class="num">' + it.dataset.num + '</span><span>' + it.dataset.nome + '</span>';
    return '<button type="button" class="fz-btn fz-btn-secondary fz-scrim" data-fz-ir="' + id + '">'
      + (dir < 0 ? ICON_BACK + rot : rot + ICON_FWD) + '</button>';
  }
  // o nome vem do sidebar, já escapado pelo Python como atributo; aqui volta a texto
  function esc(t) { var d = document.createElement('span'); d.textContent = t; return d.innerHTML; }

  function montarPes() {
    document.querySelectorAll('.fz-bloco[data-bloco]').forEach(function (b) {
      var pe = b.querySelector(':scope > [data-fz-pe]');
      if (!pe) return;
      var i = ordem.indexOf(b.dataset.bloco);
      pe.innerHTML = botao(ordem[i - 1], -1) + botao(ordem[i + 1], 1);
    });
  }

  // barra de filtro ([data-fz-escopo="fold midia"]) aparece nos blocos daquele
  // escopo; no "Mostrar tudo", sempre
  function refrescar() {
    var escopo = 'geral';
    var b = document.querySelector('.fz-bloco[data-bloco="' + atual + '"]');
    if (b) escopo = b.dataset.escopo;
    var tudo = root.hasAttribute('data-tudo');
    document.querySelectorAll('[data-fz-escopo]').forEach(function (el) {
      el.hidden = !tudo && el.dataset.fzEscopo.split(' ').indexOf(escopo) === -1;
    });
    fzMostrar(document);
  }

  // Filtro: <select data-fz-filtro="nome"> troca os painéis
  // [data-fz-filtro-painel="nome"][data-valor]. O valor vive no hash (#…&nome=v)
  // e o bloco aberto não muda.
  function valores(nome) {
    return Array.prototype.map.call(
      document.querySelectorAll('[data-fz-filtro-painel="' + nome + '"]'),
      function (p) { return p.dataset.valor; });
  }
  function filtro(nome, valor, push) {
    var vs = valores(nome);
    if (!vs.length) return;
    if (vs.indexOf(valor) === -1) valor = vs[0];
    document.querySelectorAll('[data-fz-filtro-painel="' + nome + '"]').forEach(function (p) {
      p.classList.toggle('ativo', p.dataset.valor === valor);
    });
    document.querySelectorAll('select[data-fz-filtro="' + nome + '"]').forEach(function (sel) { sel.value = valor; });
    if (push !== undefined) setParam(nome, valor, push);
    fzMostrar(document);
  }
  document.addEventListener('change', function (e) {
    var sel = e.target.closest ? e.target.closest('select[data-fz-filtro]') : null;
    if (sel) filtro(sel.dataset.fzFiltro, sel.value, true);
  });
  function filtrosDoHash(p) {
    document.querySelectorAll('select[data-fz-filtro]').forEach(function (sel) {
      filtro(sel.dataset.fzFiltro, p[sel.dataset.fzFiltro] || sel.value);
    });
  }

  function ir(id, push) {
    if (ordem.indexOf(id) === -1) id = ordem[0];
    atual = id;
    root.removeAttribute('data-tudo');
    itens.forEach(function (b) { b.classList.toggle('sel', b.dataset.fzIr === id); b.setAttribute('aria-current', b.dataset.fzIr === id ? 'true' : 'false'); });
    var tudoBtn = document.querySelector('[data-fz-tudo]');
    if (tudoBtn) tudoBtn.classList.remove('sel');
    document.querySelectorAll('.fz-bloco[data-bloco]').forEach(function (b) {
      b.classList.toggle('ativo', b.dataset.bloco === id);
    });
    var p = params(); p.bloco = id; delete p.tudo; gravar(p, push);
    window.scrollTo(0, 0);
    refrescar();
  }

  function tudo(push) {
    root.setAttribute('data-tudo', '');
    itens.forEach(function (b) { b.classList.remove('sel'); });
    var tudoBtn = document.querySelector('[data-fz-tudo]');
    if (tudoBtn) tudoBtn.classList.add('sel');
    var p = params(); p.tudo = '1'; gravar(p, push);
    refrescar();
  }

  document.addEventListener('click', function (e) {
    var alvo = e.target.closest ? e.target.closest('[data-fz-ir]') : null;
    if (alvo) { ir(alvo.dataset.fzIr, true); return; }
    if (e.target.closest && e.target.closest('[data-fz-tudo]')) { tudo(true); return; }
    if (e.target.closest && e.target.closest('[data-fz-sb-toggle]')) {
      var fechada = root.getAttribute('data-sb') === 'fechada';
      if (fechada) root.removeAttribute('data-sb'); else root.setAttribute('data-sb', 'fechada');
      try { localStorage.setItem('fz-sb', fechada ? 'aberta' : 'fechada'); } catch (err) {}
      fzMostrar(document);
    }
  });

  // janela que encolhe abaixo de 768px recolhe o sidebar (a mesma regra do
  // carregamento, no <head>); voltando a crescer, vale a escolha salva no toggle
  if (window.matchMedia) {
    var estreita = window.matchMedia('(max-width: 767px)');
    var aoRedimensionar = function () {
      if (root.hasAttribute('data-fz-embed')) return;
      var salvo = null;
      try { salvo = localStorage.getItem('fz-sb'); } catch (err) {}
      if (estreita.matches || salvo === 'fechada') root.setAttribute('data-sb', 'fechada');
      else root.removeAttribute('data-sb');
      fzMostrar(document);
    };
    if (estreita.addEventListener) estreita.addEventListener('change', aoRedimensionar);
    else if (estreita.addListener) estreita.addListener(aoRedimensionar);
  }

  document.addEventListener('keydown', function (e) {
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    var t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    if (root.hasAttribute('data-tudo')) return;
    var i = ordem.indexOf(atual);
    if (e.key === 'ArrowRight' && i < ordem.length - 1) { e.preventDefault(); ir(ordem[i + 1], true); }
    if (e.key === 'ArrowLeft' && i > 0) { e.preventDefault(); ir(ordem[i - 1], true); }
  });

  function doHash() {
    var p = params();
    filtrosDoHash(p);
    if (p.tudo) tudo(false); else ir(p.bloco || ordem[0], false);
    if (window.FZ_NAV && window.FZ_NAV.onHash) window.FZ_NAV.onHash(p);
  }
  window.addEventListener('popstate', doHash);
  window.addEventListener('hashchange', doHash);

  itens.forEach(function (b) { b.dataset.nome = esc(b.dataset.nome); });
  montarPes();
  window.FZ_NAV = { ir: ir, params: params, setParam: setParam, refrescar: refrescar, filtro: filtro, onHash: null };
  doHash();
})();
"""


def nav_js(adiado: bool = False) -> str:
    """``adiado=True`` não liga a navegação: define ``window.fzNavInit()``, para o
    relatório chamar depois que os blocos existirem (o corpo do EDA chega em gzip)."""
    import json

    js = NAV_JS.replace("%BACK%", json.dumps(icone("arrow_back", 16))).replace(
        "%FWD%", json.dumps(icone("arrow_forward", 16))
    )
    if adiado:
        corpo = js.strip()
        assert corpo.startswith("(function () {") and corpo.endswith("})();")
        js = "\nwindow.fzNavInit = " + corpo[1:-4] + ";\n"
    return js


def motor_js() -> str:
    """HYDRATE_JS + MOTOR_JS: tem de vir antes de theme_js/nav_js e do JS do relatório."""
    import json

    from ._payload import HYDRATE_JS

    sent = {f"{n:02x}": token for token, n in _SENTINELAS.values()}
    return HYDRATE_JS + MOTOR_JS.replace("%SENT%", json.dumps(sent))


def theme_js() -> str:
    import json

    return THEME_JS.replace("%DARK%", json.dumps(icone("dark_mode", 20))).replace(
        "%LIGHT%", json.dumps(icone("light_mode", 20))
    ) + HELP_JS


def scripts(extra: str = "", nav_adiado: bool = False) -> str:
    """A ordem que funciona: motor → tema/ajuda → navegação → JS do relatório."""
    return motor_js() + theme_js() + nav_js(nav_adiado) + extra


_DOCUMENTO = """<!doctype html>
<html lang="pt-br"{html_attrs}>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script>{tema_head}</script>
{plotly_js}
<style>
{style}
</style>
</head>
<body>
<div class="fz-app">
{sidebar}
<main class="fz-main"><div class="fz-page">
{body}
</div></main>
</div>
{figures}
<script>
{script}
</script>
</body>
</html>
"""


def documento(title: str, sidebar_html: str, body: str, plotly_js: str, figures: str,
              script: str, extra_css: str = "", embed: bool = False) -> str:
    """``embed=True`` é para iframe de célula de notebook: o sidebar nasce recolhido."""
    return _DOCUMENTO.format(
        html_attrs=' data-fz-embed' if embed else "",
        title=_esc(title),
        tema_head=_TEMA_HEAD_JS,
        plotly_js=plotly_js,
        style=css() + extra_css,
        sidebar=sidebar_html,
        body=body,
        figures=figures,
        script=script,
    )
