# Deep Dive — Quebra de Contribuição por Sub-Canal

> Distribui a contribuição agregada de um veículo de mídia entre sub-canais
> usando modelos Hill ancorados em `C_t` com priors calibrados.

---

## 1. Contexto e Problema

Um MMM produz a contribuição total do veículo `C_t` — um único número semanal que representa quanto o canal inteiro contribuiu para o KPI (vendas, transações, receita).

O Deep Dive responde à pergunta que o MMM não responde diretamente:

> **"Quanto dessa contribuição veio de cada praça / ambiente / formato?"**

Para isso, ajusta um modelo Deep Dive Raven por dimensão de quebra, com dois tipos de restrição simultânea:

- **Âncora de proxy**: soma das contribuições estimadas ≈ `C_t` (CoupledExactLikelihood, tolerância ±15%)
- **Prior de share**: shares de contribuição ≈ shares de investimento (ContributionShareLikelihood, softened via `share_prior_scale`)

---

## 2. Estrutura do Repositório

```
deepdive/
├── src/
│   ├── config.py                    # DeepDiveConfig + build_config() — parse YAML + UpgradeResult
│   ├── extraction.py                # load_upgrade_stan/meridian — parquets via MLflow
│   ├── diagnostics.py               # run_diagnostics() — filtra variáveis, cria __outros__
│   ├── pipeline.py                  # run_deep_dive() — orquestrador por dimensão
│   ├── plots.py                     # Plotly dark theme + analyze_deepdive/batch/trees
│   ├── report.py                    # generate_report() — CSVs + HTMLs por cliente
│   ├── batch.py                     # run_deep_dive_batch() + consolidate_results()
│   ├── synthetic_data.py            # generate_synthetic_dim() — validação semi-sintética
│   ├── contrib_share_likelihood.py  # ContributionShareLikelihood — efeito prophetverse
│   └── raven_patch.py               # Raven subclass + CosineScheduleAdamWOptimizer
├── tests/
├── configs/
│   ├── clients_registry.yaml        # Cadastro de clientes
│   ├── bradesco_eletro.yaml
│   ├── hypera_eletro.yaml
│   └── opella_eletro.yaml
├── data/
│   └── vehicle_specs.yaml           # Hierarquias, slugs e rollups por veículo
├── notebooks/
│   ├── deep_dive_eletro.ipynb            # Pipeline single-client
│   ├── deep_dive_batch.ipynb             # Pipeline multi-cliente + meta-análise
│   └── validacao_prior_auxiliar.ipynb    # Benchmark: prior auxiliar de medição
├── benchmarks/
│   └── share_recovery_benchmark.py
└── outputs/
    ├── {cliente}/                    # CSVs + HTMLs por cliente
    └── batch/                        # meta_analysis.csv + sunbursts
```

---

## 3. Pipeline — Visão Geral

```
MMM Base 
        │
        ▼
  C_t = contribuição semanal do canal
        │
        ├──► Dimensão X  ──► shares + ROAS Index
        ├──► Dimensão Y  ──► shares + ROAS Index
        └──► Dimensão Z  ──► shares + ROAS Index
```

Cada dimensão é ajustada **independentemente**, mas todas usam o mesmo `C_t` como âncora. O pipeline opera em três etapas:

1. **Extração** — `load_upgrade_stan` / `load_meridian_upgrade`: carrega `contrib_df` e `spend_df` via parquets MLflow.
2. **Diagnóstico** — `run_diagnostics`: filtra sub-canais com <2% de spend, agrupa em `__outros__`, calcula HHI e semanas ativas.
3. **Deep Dive Raven** — `run_deep_dive`: ajusta modelo Hill por dimensão, ancorado em `C_t`.

---

## 4. Fundamentação Matemática

### 4.1 Função de Resposta Hill

Cada sub-canal `k` tem uma curva de saturação Hill que mapeia o investimento semanal normalizado `x_kt` em contribuição relativa:

```
h_k(x_kt) = me_k · x_kt^sl_k / (hm_k^sl_k + x_kt^sl_k)
```

| Parâmetro | Símbolo | Interpretação |
|---|---|---|
| Efeito máximo | `me_k` | contribuição máxima (unidades de `C_t / max(C_t)`) |
| Half-saturation | `hm_k` | spend normalizado que produz 50% do efeito máximo |
| Slope | `sl_k` | curvatura — valores altos = saturação mais abrupta |

`x_kt = s_kt / max_k(s)` — spend normalizado pelo máximo histórico do sub-canal.

Contribuição total estimada na semana `t`:

```
C_t_hat = sum_k h_k(x_kt)
```

### 4.2 Âncora de Proxy — CoupledExactLikelihood

O MMM base fornece `C_t` como âncora. O proxy é normalizado por `max(C_t)` para ficar na mesma escala das saídas Hill (o Raven usa `target_scale="max"` internamente):

```
proxy_t = C_t / max(C_t)
```

Termo adicionado ao log-posterior:

```
log p(proxy | Hills) = log Normal(proxy_t ; sum_k h_k(x_kt), sigma_proxy)
```

`sigma_proxy` calibrado automaticamente:

```
sigma_proxy = tolerance × mean(C_t[C_t > 0]) / max(C_t)
```

Default: `tolerance = 0.15` (±15%). Sem normalização por `max(C_t)`, as escalas divergem e o gradiente explode nas primeiras iterações.

### 4.3 Prior de Share — ContributionShareLikelihood (CSL)

A âncora de proxy controla o **total**. O CSL controla a **distribuição** entre sub-canais, ancorando shares de contribuição às shares de investimento:

**Shares de referência (métrica):**
```
metric_share_k = sum_t metric_kt / sum_k sum_t metric_kt
```

**Shares estimadas pelo modelo:**
```
model_share_k = sum_t h_k(x_kt) / sum_k sum_t h_k(x_kt)
```

**Likelihood:**
```
log p(shares | Hills) = sum_k log Normal(model_share_k ; metric_share_k, scale)
```

| `scale` | Interpretação |
|---|---|
| 0.005 | prior forte — com dados auxiliares de medição |
| 0.05 | default — spend como referência (prior fraco) |
| 0.10 | prior muito fraco — modelo livre para redistribuir |

Por default, `metric_df = spend_df`. Com dados auxiliares (GRP, brand awareness, impressões), substituir `metric_df` e reduzir `scale` para 0.005.

### 4.5 Prior Auxiliar de Medição

Por padrão `metric_df = spend_df` — shares de spend como referência no CSL. Quando dados de medição direta estão disponíveis (GRP por praça, impressões por formato, brand awareness por região), eles podem substituir o spend como referência, capturando exposição real em vez de apenas investimento.

**Modelo generativo do ruído de medição:**

Dado ground-truth de shares `σ_k^true`, a medição observada é corrompida por ruído log-normal:

```
ℓ_k = log(σ_k^true) + η_k,    η_k ~ N(0, σ_meas)
```

O `metric_df` é construído aplicando softmax sobre as log-medições:

```
metric_share_k = exp(ℓ_k) / Σ_j exp(ℓ_j)
```

No caso noiseless (η=0): `softmax(log(σ^true))_k = σ_k^true`. Com ruído, as shares observadas se desviam das verdadeiras proporcionalmente a `σ_meas`.

**Por que é melhor que spend:**

| Referência | Captura | Limitação |
|---|---|---|
| Spend | Quanto foi investido | Diferenças de CPM entre sub-canais distorcem shares |
| Medição (GRP, impressões) | Exposição real ao público | Ruído de medição; disponibilidade por cliente |

**Trade-off validado (benchmark H1):**

| `scale` | `σ_meas` | MAE | Proxy ratio | Status |
|---|---|---|---|---|
| 0.05 (spend puro) | — | 0.059 | ≈ 1.0 | baseline |
| 0.005 | 0.00 | 0.031 | ≈ 1.0 | **−47% — ponto ótimo** |
| 0.005 | 0.20 | 0.044 | ≈ 1.0 | **−25% mesmo com ruído alto** |
| 0.001 | 0.00 | 0.016 | 1.225 | ⚠️ viola âncora C_t |

`scale=0.001` melhora shares mas viola a âncora proxy em 22% — o modelo passa a "inventar" contribuição além de C_t. Ponto ótimo: `scale=0.005`.

---

### 4.4 Objetivo MAP Completo

```
theta* = argmax_theta [
    log p(proxy | Hills, sigma_proxy)           ← âncora de proxy
  + log p(model_shares | metric_shares, scale)  ← prior de share (CSL)
  + log p(C_t | theta)                          ← likelihood principal
  + log p(theta)                                ← priors Hill
]
```

`theta = {me_k, hm_k, sl_k}` para cada sub-canal `k`.

Otimizador: **AdamW + cosine decay** (`CosineScheduleAdamWOptimizer`):
- Weight decay regulariza `max_effect` — evita que Hill functions concentrem toda a contribuição em um sub-canal com pouco spend.
- Cosine decay estabiliza convergência na fase final (30.000 steps por dimensão, configurável via `num_steps`).

---

## 5. Arquitetura do Sistema

### Dataclasses Tipadas

`DeepDiveConfig`, `UpgradeResult`, `DiagnosisResult`, `DDResult`.

### Diagnósticos Pré-Fit

```python
run_diagnostics(config, upgrade)
  → DiagnosisResult.spend_report   # HHI, % spend, semanas ativas por variável
  → DiagnosisResult.bucketed        # {dim: {var → "__outros__"}}
  → DiagnosisResult.skipped_dims    # dims sem variáveis após filtro
```

Limiares padrão: `min_spend_share=0.02`, `min_active_weeks=2`.

### Rollup Genérico via YAML

```yaml
# vehicle_specs.yaml
breakdowns:
  Ambiente:
    rollups:
      - level: grupo
        groups: grupos
        members_key: ambientes
      - level: vertical
        groups: grupos
        members_key: ambientes
        attr: vertical
  Praca:
    rollups:
      - level: estado
        map: praca_to_estado
      - level: praca
```

Novo veículo = novo YAML. Sem alteração de código Python.

### Extração via Parquets

`_load_from_parquets(run_id, contribution_metric_type, model_type)`:
- Stan: `contribution_metric_type="Contribution Unadstocked"`
- Meridian: `contribution_metric_type="Contribution"` + tolerância de +1 linha (`contrib_df` pode ter 1 linha a mais que o input — trimada automaticamente)

### Validação Semi-Sintética

`synthetic_data.py` gera dados com parâmetros Hill conhecidos, permitindo validar recuperação de shares antes de rodar dados reais.

---

## 6. Hipóteses e Evidências

### H1 — Prior Auxiliar Melhora Recuperação de Shares ✅

Benchmark: `validacao_prior_auxiliar.ipynb` — 40 cenários (K=4, T=52, `scale` ∈ {0.001, 0.005, 0.01, 0.05}, `σ_meas` ∈ {0, 0.05, 0.1, 0.2}):

| Cenário | MAE | Δ vs. baseline |
|---|---|---|
| Baseline (spend puro) | 0.059 | — |
| Prior perfeito (`σ_meas=0`, `scale=0.005`) | 0.031 | **−47%** |
| Prior ruidoso (`σ_meas=0.20`, `scale=0.005`) | 0.044 | **−25%** |
| `scale=0.05` (default, sem aux) | 0.061 | <1% |

⚠️ `scale=0.001` produz MAE excelente mas `proxy_ratio > 1.2` — viola a âncora. Ponto ótimo validado: `scale=0.005`.

### H2 — Normalização do Proxy por `max(y)` Evita Explosão de Gradiente ✅

Sem normalização, o proxy (escala absoluta de `C_t`) domina o gradiente e o otimizador diverge. `X_proxy = C_t / max(C_t)` alinha a escala com as saídas Hill.

### H3 — Sigma do Proxy Auto-calibrado por Sub-canal ✅

```
sigma_proxy_v = tolerance × mean(C_t_v[C_t_v > 0]) / max(C_t_v)
```

(`max(C_t)` cancela no numerador e denominador da forma longa no código.) Sub-canais com magnitudes diferentes recebem tolerância proporcional à própria escala — sem tuning manual de `sigma_proxy` por variável.

---

## 7. Escolhas Metodológicas

| Decisão | Alternativa | Motivo |
|---|---|---|
| Proxy exact (±15%) | Proxy proporcional (escala livre) | Proporcional tem fator de escala não identificado → shares arbitrárias |
| CSL em espaço de shares (Normal) | Log-ratio | Normal evita singularidades em share=0 |
| MAP com CosineScheduleAdamW | Adam (lr fixo) | Weight decay regulariza `max_effect`; cosine decay estabiliza convergência sem tuning manual de lr |
| PiecewiseLinearTrend | FlatTrend | Sub-canais podem ter dinâmicas independentes; piecewise detecta breakpoints locais |
| Rollups declarativos em YAML | Código Python por veículo | Extensível sem mudança de código |

---

## 8. Configuração e Uso

### 8.1 Novo Cliente

**Criar YAML do cliente:**

```yaml
# configs/novo_cliente_eletro.yaml
brand: nome-da-marca
vehicle: eletromidia
vehicle_specs_path: ../data/vehicle_specs.yaml
mlflow_tracking_uri: https://mlflow-dev.cloud.uncover.co
upgrade_run_id: <run_id_do_mmm>
workspace_dd: <workspace_mlflow>
start_date: 2022-01-03
end_date: 2025-12-29
media_var: $metric:investments$vehicle:eletromidia$category:brand:nome-da-marca
```

**Registrar no registry:**

```yaml
# configs/clients_registry.yaml
clients:
  novo_cliente:
    specs_path: novo_cliente_eletro.yaml
    model_type: stan   # ou meridian
    output_subdir: novo_cliente
```

**Novo veículo:** adicionar entrada em `data/vehicle_specs.yaml` com `breakdowns`, `hierarchy` e `rollups`. Sem alteração de código.

### 8.2 Single-Client

```python
upgrade   = load_upgrade_stan(run_id, tracking_uri=mlflow_uri)
config    = build_config(upgrade, specs_path="configs/bradesco_eletro.yaml")
config, _ = run_diagnostics(config, upgrade)
result    = run_deep_dive(config, upgrade)
_         = analyze_deepdive(result)
            generate_report(result, output_dir)
```

### 8.3 Batch Multi-Cliente

```python
registry   = load_registry("configs/clients_registry.yaml")
all_results, diags = run_deep_dive_batch(registry,
                                         registry_path="configs/clients_registry.yaml",
                                         output_base_dir="outputs")
df_meta    = consolidate_results(all_results, vehicle_spec_override=vehicle_spec)
batch_figs = analyze_batch(all_results, df_meta, vehicle_spec_override=vehicle_spec)
tree_figs  = analyze_trees(all_results, vehicle_spec_override=vehicle_spec)
```

### 8.4 Prior Auxiliar de Medição

```python
auxiliary_metric_dfs = {
    "Praca":    df_medicao_por_praca,    # DataFrame (T × K)
    "Ambiente": df_medicao_por_ambiente,
}
result = run_deep_dive(config, upgrade,
                       auxiliary_metric_dfs=auxiliary_metric_dfs)
# ajustar: config.share_prior_scale = 0.005
```

---

## 9. Outputs e Interpretação

### Métricas de Qualidade

| Métrica | Fórmula | Valor OK | Atenção |
|---|---|---|---|
| `proxy_ratio` | `Σ contribs / Σ C_t` | 0.85 – 1.15 | Fora: revisar `proxy_ct_tolerance` ou spend |
| `csl_max_dev` | `max_k \|contrib_share_k − spend_share_k\|` | < 0.20 | > 0.20: desvio forte — sinal real ou ruído |

### Métricas de Output

```
contrib_share_k = sum_t h_k(x_kt) / sum_k sum_t h_k(x_kt)
roas_index_k    = contrib_share_k / spend_share_k
```

ROAS Index é **relativo ao canal** — não é ROAS absoluto. Valor 1.4 = 40% mais eficiente que a média do canal para aquele cliente.

### Arquivos por Cliente

| Arquivo | Conteúdo |
|---|---|
| `outputs/{cliente}/{cliente}_shares.csv` | dim, item, contrib_share, spend_share, proxy_ratio, csl_max_dev |
| `outputs/{cliente}/{cliente}_roas_index.csv` | dim, item, roas_index |
| `outputs/{cliente}/{cliente}_contributions.html` | Barra agrupada: contrib_share vs. spend_share |
| `outputs/{cliente}/{cliente}_roas_index.html` | Heatmap ROAS Index por dimensão × sub-canal |

### Batch

| Arquivo | Conteúdo |
|---|---|
| `outputs/batch/meta_analysis.csv` | Long-form: cliente, dim, rollup, item, share_model, share_spend, roas_index |
| `outputs/batch/report_{Dim}.html` | Tabelas comparativas + sunburst por dimensão |
| `outputs/benchmark_share_recovery.csv` | scale, scenario, seed, mae, rmse, max_err, proxy_ratio |

---

## 10. Testes

```bash
# Testes rápidos
pytest deepdive/tests/ -v

# Incluir integração (MAP 500 steps, ~2min)
pytest deepdive/tests/ -v -m slow

# Benchmark completo (40 cenários, ~20min)
python deepdive/benchmarks/share_recovery_benchmark.py
```

| Arquivo | O Que Cobre |
|---|---|
| `test_config.py` | Parsing de YAML, defaults do dataclass |
| `test_extraction.py` | Mock MLflow, campos do UpgradeResult, parquets |
| `test_diagnostics.py` | Filtro por spend, bucketing `__outros__`, colunas do spend_report |
| `test_plots.py` | Figuras Plotly geradas, template dark |
| `test_report.py` | Criação de arquivos CSV e HTML |
| `test_pipeline_helpers.py` | `_align_to`, `_wmon_norm` (Period e Datetime) |
| `test_synthetic_deepdive.py` | Hill function, SyntheticDimension, recuperação de shares (slow) |

---

## 11. Premissas e Limitações

1. **`C_t` como âncora.** A distribuição entre sub-canais herda tanto os acertos quanto as imprecisões do modelo upstream.
2. **Spend disponível por sub-canal.** Slug ausente no `spend_df` é tratado como spend zero — não erro. Dimensão inteira pulada só se menos de 2 sub-canais têm qualquer spend (`n_active < 2`) ou HHI > threshold. Sub-canal individual sem spend é descartado ou vai para `__outros__`.
3. **Hill assume saturação monotônica crescente.** Formas em U ou com threshold não são capturadas.
4. **Frequência semanal (W-MON).** Séries diárias são agregadas; mensais não são suportadas.
5. **Sub-canais com <2% de spend** são agrupados em `__outros__`. Aumentar `min_share` em `run_diagnostics()` se necessário.
6. **`share_prior_scale`** deve ser calibrado por veículo: 0.05 (default sem dados auxiliares) → 0.005 (com dados de medição).
7. **Alta correlação entre sub-canais** (todos crescem juntos) reduz identificabilidade. O CSL mitiga mas não elimina.
8. **`proxy_ratio` fora de 0.85–1.15** indica pouco sinal em `C_t` para o nível de detalhe solicitado — não necessariamente erro de código.

---

## 12. Dependências

- `mmmverse` — Raven (PiecewiseLinearTrend, MAPInferenceEngine)
- `prophetverse` — BaseEffect (ContributionShareLikelihood)
- `jax / jaxlib` — backend numérico
- `mlflow` — artefatos e parquets
- `plotly` — visualizações (dark theme)
- `pandas / numpy` — manipulação de dados
- `pyyaml` — configuração declarativa

---

## 13. Decisões de Design Futuras

- Mover `ContributionShareLikelihood` para `prophetverse` como efeito nativo
- Mover `extra_effects` + prior dicts para `mmmverse` como API nativa do Raven
- Classe `RavenDeepDive` em `mmmverse` encapsulando o fluxo completo
- `load_raven_upgrade()` para contribuições de modelos Raven como base