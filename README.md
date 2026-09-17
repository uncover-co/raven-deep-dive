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
- **Prior de share**: shares de contribuição ≈ shares de referência — investimento por padrão, ou exposição real (impressões, GRP) quando disponível (ContributionShareLikelihood, softened via `share_prior_scale`)

---

## 2. Estrutura do Repositório

```
deepdive/
├── src/
│   ├── config.py                    # DeepDiveConfig + build_config() — parse YAML + UpgradeResult
│   ├── extraction.py                # load_upgrade_stan/meridian/raven — parquets via MLflow
│   ├── diagnostics.py               # run_diagnostics() — filtra variáveis, cria __others__
│   ├── pipeline.py                  # run_deep_dive() — orquestrador por dimensão
│   ├── plots.py                     # Plotly dark theme + analyze_deepdive/batch/trees
│   ├── report.py                    # generate_report() — CSVs + HTMLs por cliente
│   ├── batch.py                     # run_deep_dive_batch() + consolidate_results()
│   ├── stability.py                 # run_stability_test() — MAP multi-seed
│   ├── synthetic_data.py            # generate_synthetic_dim() — validação semi-sintética
│   ├── contrib_share_likelihood.py  # ContributionShareLikelihood — efeito prophetverse
│   └── raven_patch.py               # Raven subclass + CosineScheduleAdamWOptimizer
├── tests/
├── configs/
│   ├── clients_registry.yaml        # Cadastro de clientes (formato multi-veículo)
│   ├── bradesco_eletro.yaml
│   ├── bradesco_tiktok.yaml
│   ├── casas_bahia_tiktok.yaml
│   ├── alpargatas_tiktok.yaml
│   ├── hypera_eletro.yaml
│   └── opella_eletro.yaml
├── data/
│   └── vehicle_specs.yaml           # Hierarquias, slugs e rollups por veículo
├── notebooks/
│   ├── deep_dive_eletro.ipynb            # Pipeline single-client (Eletromidia)
│   ├── deep_dive_tiktok.ipynb            # Pipeline single-client (TikTok)
│   ├── deep_dive_batch.ipynb             # Pipeline multi-cliente + meta-análise
│   └── validacao_prior_auxiliar.ipynb    # Benchmark: prior auxiliar de medição
├── benchmarks/
│   └── share_recovery_benchmark.py
└── outputs/
    ├── {cliente}_{vehicle}/          # CSVs + HTMLs por cliente × veículo
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

1. **Extração** — `load_upgrade_stan` / `load_meridian_upgrade` / `load_raven_upgrade`: carrega `contrib_df` e `spend_df` via parquets MLflow.
2. **Diagnóstico** — `run_diagnostics`: usa `auxiliary_metric` (obrigatório) como gate — filtra sub-canais com <2% nessa métrica, agrupa em `__others__`, calcula HHI e semanas ativas. Se o veículo não tem métrica de exposição real, o client YAML aponta `auxiliary_metric` pro mesmo valor de `share_likelihood_metric` (investimento como seu próprio proxy) — a escolha é feita na config, não em runtime. Se a dimensão não tiver dado real na métrica configurada, `run_diagnostics` levanta erro (sem fallback silencioso).
3. **Deep Dive Raven** — `run_deep_dive`: ajusta modelo Hill por dimensão, ancorado em `C_t`. Requer `auxiliary_metric_dfs` (de `diag.auxiliary_metric_dfs`) — não é opcional.

---

## 4. Restrições do Modelo

**Âncora de Proxy (CoupledExactLikelihood):** soma das contribuições por Hill function ≈ `C_t` do MMM (tolerância ±15%).

**Prior de Share (ContributionShareLikelihood):** shares de contribuição de cada sub-canal ≈ shares de referência (por default, share de spend). Com dados auxiliares (impressões, GRP), apontar `auxiliary_metric` e reduzir `share_prior_scale` de 0.05 → 0.005 pra confiar mais no prior baseado em exposição.

Otimizador: **AdamW + cosine decay** (30k steps/dim). Weight decay regulariza concentração em um sub-canal; cosine decay estabiliza convergência final.

---

## 5. Configuração e Uso

### 5.1 Novo Cliente

**Criar YAML do cliente:**

```yaml
# configs/novo_cliente_eletro.yaml
brand: nome-da-marca
vehicle: eletromidia
model_type: stan          # stan, meridian ou raven
model_name: "Nome do Modelo Upstream"  # identificador legível (opcional)
vehicle_specs_path: ../data/vehicle_specs.yaml
mlflow_tracking_uri: https://mlflow-dev.cloud.uncover.co
upgrade_run_id: <run_id_do_mmm>
workspace_dd: <workspace_mlflow>
data_version: <nome_do_snapshot>  # opcional -- fixa a leitura num data version do ducks; omitir usa o dado atual da plataforma, não a versão salva
start_date: 2022-01-03
end_date: 2025-12-29
media_var: $metric:investments$vehicle:eletromidia$category:brand:nome-da-marca
auxiliary_metric: investments  # obrigatório -- métrica de exposição real (ex: impressions) quando o veículo tiver; senão, o mesmo valor do investimento
```

**Registrar no registry** (formato multi-veículo, recomendado — `model_type` vem do client YAML, não do registry):

```yaml
# configs/clients_registry.yaml
clients:
  novo_cliente:
    output_subdir: "NovoCliente"
    vehicles:
      eletromidia:
        specs_path: novo_cliente_eletro.yaml
```

**Novo veículo:** adicionar entrada em `data/vehicle_specs.yaml` com `breakdowns`, `hierarchy` e `rollups`. Sem alteração de código.

### 5.2 Single-Client

```python
# run_id, mlflow_uri, workspace_dd, start_date, end_date, output_dir: do client YAML.
upgrade        = load_upgrade_stan(run_id, tracking_uri=mlflow_uri)
config         = build_config(upgrade, specs_path="configs/bradesco_eletro.yaml")

# upgrade.spend_df vem vazio -- popula com o breakdown real antes do diagnóstico.
all_vars       = [v for slugs in config.vars_per_dim.values() for v in slugs]
upgrade.spend_df = load_breakdown_spend(workspace_dd, all_vars, start_date, end_date)

config, diag   = run_diagnostics(config, upgrade)
result         = run_deep_dive(config, upgrade, auxiliary_metric_dfs=diag.auxiliary_metric_dfs)
_              = analyze_deepdive(result)
generate_report(result, diag=diag, output_dir=output_dir, client_name="bradesco")
# outputs em: outputs/bradesco_eletromidia/
```

### 5.3 Batch Multi-Cliente

```python
registry = load_registry("configs/clients_registry.yaml")
all_results, diagnostics, errors = run_deep_dive_batch(
    registry, registry_path="configs/clients_registry.yaml", output_base_dir="outputs"
)
df_meta    = consolidate_results(all_results)  # usa result.config.vehicle_spec de cada cliente
batch_figs = analyze_batch(all_results, df_meta)
```

### 5.4 Prior com Dados Auxiliares

Fluxo padrão (5.2): `auxiliary_metric_dfs=diag.auxiliary_metric_dfs`, montado automaticamente por `run_diagnostics` a partir do `auxiliary_metric` do client YAML.

Pra passar medição própria em vez disso, precisa cobrir **todas** as dims em `config.dims` (`run_deep_dive` levanta erro se faltar uma):

```python
auxiliary_metric_dfs = {dim: diag.auxiliary_metric_dfs[dim] for dim in config.dims}
auxiliary_metric_dfs["Praca"] = df_impressoes_praca  # override pontual (T × K, mesmas colunas finais do dim)
result = run_deep_dive(config, upgrade, auxiliary_metric_dfs=auxiliary_metric_dfs)
# Ajustar: config.share_prior_scale = 0.005 (no client YAML)
```

### 5.5 Adstock Customizado

Declare no client YAML via `!instance`/`!params`. Chaves devem ser slugs exatamente como aparecem em `config.vars_per_dim[dim]` **após diagnóstico** (incluindo `__others__<dim>` se houver). Rode `run_diagnostics` primeiro pra saber a lista exata — `_run_raven_dim` (pipeline.py) valida e levanta erro antes de construir o modelo se faltar ou sobrar chave.

```yaml
upper_funnel_adstock_effect_per_dim:
  product_level_4:
    "$metric:w:investments---tiktok-mmm$category:brand:bradesco$category:product-level-1:tiktok$category:product-level-4:app-retargeting":
      '!instance': prophetverse.effects.adstock.WeibullAdstockEffect
      '!params': {max_lag: 2}  # memória curta
    # ... uma entrada por cada slug restante em product_level_4, incluindo __others__product_level_4 se existir
```

Alternativa (mais simples): um único `!instance`/`!params` sem dict aplica o mesmo efeito a todo o dim.

---

## 6. Outputs e Interpretação

### Métricas de Qualidade

| Métrica | Fórmula | Valor OK | Atenção |
|---|---|---|---|
| `proxy_ratio` | `Σ contribs / Σ C_t` | 0.85 – 1.15 | Fora: revisar `proxy_ct_tolerance` ou spend |
| `csl_max_dev` | `max_k \|contrib_share_k − csl_metric_share_k\|` (métrica de referência do prior CSL — `auxiliary_metric`, spend quando não há exposição real) | < 0.20 | > 0.20: desvio forte — sinal real ou ruído |

### Métricas de Output

```
contrib_share_k = sum_t h_k(x_kt) / sum_k sum_t h_k(x_kt)
roas_index_k    = contrib_share_k / spend_share_k
```

ROAS Index é **relativo ao canal** — não é ROAS absoluto. Valor 1.4 = 40% mais eficiente que a média do canal para aquele cliente.

### Arquivos por Cliente

Output dir: `outputs/{cliente}_{vehicle}/` (gerado por `generate_report()`)

| Arquivo | Conteúdo |
|---|---|
| `metadata.csv` | model_name, client, vehicle, upgrade_run_id, dd_date, period_start, period_end |
| `contributions.csv` | dim, item, contrib_share, spend_share, roas_index (+ rollups quando `vehicle_spec` tem hierarquia) |
| `diagnostics.csv` | Saída de `run_diagnostics`: status (kept/discarded_*/others_aggregate), reason, active_weeks, gate_total — só se `diag` for passado |
| `hill_params.csv` | Parâmetros Hill (max_effect, half_max, slope) por variável e dimensão |
| `model_inputs.csv` | dim, variable, date, investment, auxiliary_metric — exatamente o que entrou no fit de cada dim (já pós-diagnóstico: `__others__<dim>` no lugar dos membros absorvidos) |
| `model_inputs_bucketed_detail.csv` | Mesmas colunas + `dim`, mas com as séries originais (pré-agregação) de cada membro absorvido em `__others__` — só para auditoria, nunca alimentou o modelo. Gerado só se `diag` for passado e houver bucketing |
| `contributions.html` | Barra agrupada: contrib_share vs. spend_share |
| `roas_index.html` | Heatmap ROAS Index por dimensão × sub-canal |
| `weekly_{dim}.html` | Área empilhada: contribuição semanal por sub-canal, um por dimensão |
| `weekly_{dim}_{level}.html` | Mesmo, no nível de rollup (quando `vehicle_spec` define rollups) |

### Batch

| Arquivo | Conteúdo |
|---|---|
| `outputs/batch/meta_analysis.csv` | Long-form: cliente, dim, rollup, item, share_model, share_spend, roas_index |
| `outputs/batch/report_{Dim}.html` | Tabelas comparativas + sunburst por dimensão |
| `outputs/benchmark_share_recovery.csv` | scale, scenario, seed, mae, rmse, max_err, proxy_ratio |

---

## 7. Testes

```bash
# Testes rápidos
pytest deepdive/tests/ -v

# Incluir integração
pytest deepdive/tests/ -v -m slow

# Benchmark completo
python deepdive/benchmarks/share_recovery_benchmark.py
```

| Arquivo | O Que Cobre |
|---|---|
| `test_config.py` | Parsing de YAML, defaults do dataclass |
| `test_extraction.py` | Mock MLflow, campos do UpgradeResult, parquets, cache completo vs. parcial |
| `test_diagnostics.py` | Filtro por spend, bucketing `__others__`, colunas do spend_report |
| `test_slug_genericity.py` | `build_config`/`_build_vars_per_dim`: templates, placeholders, cross-product de métricas |
| `test_funnel_split.py` | `lower_funnel_vars_per_dim`: fit sem adstock por sub-canal, herança em `__others__` |
| `test_adstock_per_variable.py` | `upper_funnel_adstock_effect_per_dim`: adstock customizado por variável, chaves obrigatórias |
| `test_raven_patch.py` | Patch de Hill priors no `Raven` (duck typing, no-op quando vazio) |
| `test_plots.py` | Figuras Plotly geradas, template dark |
| `test_report.py` | Criação de arquivos CSV e HTML |
| `test_pipeline_helpers.py` | `align_to`, `wmon_norm` (Period e Datetime) |
| `test_run_deep_dive_validation.py` | `auxiliary_metric_dfs` sem entry pro dim ou sem coluna obrigatória — levanta erro, sem fit |
| `test_synthetic_deepdive.py` | Hill function, SyntheticDimension, recuperação de shares (slow) |

---

## 8. Premissas e Limitações

1. **`C_t` como âncora.** A distribuição entre sub-canais herda tanto os acertos quanto as imprecisões do modelo upstream.
2. **`auxiliary_metric` é obrigatório e sempre decide o gate.** Não há fallback automático em runtime: se a dimensão não tiver dado real na métrica configurada, `run_diagnostics` levanta `ValueError` (não faz skip silencioso, nem cai pro investimento sozinho). O fallback pra investimento-como-proxy é uma decisão explícita no client YAML (`auxiliary_metric` apontando pro mesmo valor de `share_likelihood_metric`), não algo que o sistema escolhe sozinho.
3. **Investimento disponível por sub-canal.** Slug ausente no `spend_df` → sem série de investimento → descartado silenciosamente (sem erro, sem entrar em `__others__`), mesmo que passe no gate de exposição. Sub-canal com < `min_spend_share` (default 2%) na métrica de gate vai pra `__others__`. Dimensão inteira pulada se `n_active < 2` ou HHI > `hhi_threshold` (calculados na métrica de gate).
4. **Frequência semanal (W-MON).** Séries diárias são agregadas; mensais não são suportadas.
5. **`share_prior_scale`** deve ser calibrado por veículo: 0.05 quando `auxiliary_metric` aponta pro próprio investimento (sem exposição real) → 0.005 com exposição real (ex: impressions).
6. **Alta correlação entre sub-canais** (todos crescem juntos) reduz identificabilidade. O CSL mitiga mas não elimina.
7. **`proxy_ratio` fora de 0.85–1.15** pode indicar sinal ruidoso em `C_t` para o nível de detalhe solicitado (ver §6).
8. **Classificação funil do `__others__`** herda `lower_funnel_vars_per_dim` só quando todos os membros agrupados são lower funnel (caso homogêneo). Se o bucket for misto (alguns lower, alguns upper), não há classificação inequívoca — o agregado fica upper funnel (adstocked) por padrão. Limitação conhecida.
9. **Adstock per-variável é tudo ou nada por dimensão.** `upper_funnel_adstock_effect_per_dim[dim]` no modo dict exige uma entrada pra cada variável upper funnel daquele dim — sem meio-termo (algumas customizadas, outras no default automático). Cobrir todas com o mesmo efeito, ou usar um único `!instance`/`!params` (sem dict) pra aplicar a todo o grupo, quando não precisar de granularidade por variável.

---

## 9. Dependências

- `mmmverse` — Raven (PiecewiseLinearTrend, MAPInferenceEngine)
- `prophetverse` — BaseEffect (ContributionShareLikelihood)
- `jax / jaxlib` — backend numérico
- `mlflow` — artefatos e parquets. Acesso ao bucket S3 do artifact store varia por run: alguns usam chaves estáticas (`.env`), outros exigem AWS SSO (`aws sso login --profile <perfil>`) — se `download_artifacts`/`list_artifacts` falhar com erro de credencial/token expirado, checar qual dos dois mecanismos o run em questão usa antes de assumir bug.
- `plotly` — visualizações (dark theme)
- `pandas / numpy` — manipulação de dados
- `pyyaml` — configuração declarativa

---

## 10. Decisões de Design Futuras

- Mover `ContributionShareLikelihood` para `prophetverse` como efeito nativo
- Mover `extra_effects` + prior dicts para `mmmverse` como API nativa do Raven
- Classe `RavenDeepDive` em `mmmverse` encapsulando o fluxo completo
- `load_raven_upgrade()` para contribuições de modelos Raven como base