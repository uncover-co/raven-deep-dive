# Deep Dive

> Quebra a contribuição agregada de um veículo de mídia em sub-canais
> usando modelos Deep Dive Raven com âncoras e priors calibrados.

---

## Contexto do Projeto

Um MMM (Raven / Stan / Meridian) produz a contribuição total do veículo `C_t`. O Deep Dive responde a pergunta:
> **"Quanto dessa contribuição veio de cada ambiente / praça / formato?"**

Para isso, ajusta um modelo Deep Dive Raven por dimensão de quebra, com dois tipos de restrição simultânea:

- **Âncora de proxy**: soma das contribuições estimadas ≈ `C_t` (CoupledExactLikelihood, tolerância ±15%)
- **Prior de share**: shares de contribuição ≈ shares de investimento (ContributionShareLikelihood, softened via `share_prior_scale`)

Os parâmetros Hill resultantes permitem calcular ROAS index e curvas de saturação por sub-canal.

---

## Estrutura do Repositório

```
deepdive/
├── src/
│   ├── config.py                 # DeepDiveConfig + build_config() — parse YAML + UpgradeResult
│   ├── extraction.py             # load_upgrade_auto() — Stan, Meridian, Raven via MLflow
│   ├── diagnostics.py            # run_diagnostics() — filtra variáveis, cria __outros__
│   ├── pipeline.py               # run_deep_dive_e1() — orquestrador Deep Dive Raven por dimensão
│   ├── plots.py                  # Plotly dark theme + analyze_deepdive/analyze_batch/analyze_trees
│   ├── report.py                 # generate_report() — CSVs + HTMLs por cliente
│   ├── batch.py                  # run_deep_dive_batch() + consolidate_results() — multi-cliente
│   ├── synthetic_data.py         # generate_synthetic_dim() — validação semi-sintética
│   ├── contrib_share_likelihood.py  # ContributionShareLikelihood — efeito prophetverse
│   └── raven_patch.py            # Raven subclass + CosineScheduleAdamWOptimizer
├── tests/                        # Testes unitários e de integração
├── configs/
│   ├── clients_registry.yaml     # Cadastro de clientes
│   ├── bradesco_eletro.yaml
│   ├── hypera_eletro.yaml
│   └── opella_eletro.yaml
├── data/
│   └── vehicle_specs.yaml        # Hierarquias, slugs e rollups por veículo
├── notebooks/
│   ├── deep_dive_eletro.ipynb    # Pipeline single-client
│   └── deep_dive_batch.ipynb     # Pipeline multi-cliente + meta-análise
├── benchmarks/
│   └── share_recovery_benchmark.py
└── outputs/
    ├── {cliente}/                # CSVs + HTMLs por cliente
    └── batch/                    # meta_analysis.csv + sunbursts
```

---

## Problemas Resolvidos

| Problema | Solução |
|---|---|
| Modelos por dimensão sem âncora comum → shares inconsistentes | Proxy exact likelihood fixa soma em `C_t` por modelo |
| Colunas correlacionadas por falta de sinal → divergência | `run_diagnostics()` buketiza vars com <2% de spend em `__outros__` |
| Análise manual por cliente, sem visão cross-client | `run_deep_dive_batch()` + `consolidate_results()` + `analyze_batch()` |
| Visualização hierárquica inexistente | `analyze_trees()` gera sunburst/treemap/icicle por dimensão, genérico por veículo |

---

## Melhorias Implementadas

### Arquitetura Modular

Cada notebook chama funções importadas de `src/`. Dez módulos com responsabilidades separadas (config, extração, diagnósticos, pipeline, plots, report, batch, dados sintéticos, efeito CSL, patch Raven).

### Dataclasses Tipadas

`DeepDiveConfig`, `UpgradeResult`, `DiagnosisResult`, `DDResult` — sem dicts volantes.

### Diagnósticos Pré-Fit

```
run_diagnostics(config, upgrade)
  → DiagnosisResult.spend_report   # HHI, % spend, semanas ativas por variável
  → DiagnosisResult.bucketed        # {dim: {var → __outros__}}
  → DiagnosisResult.skipped_dims    # dims sem variáveis após filtro
```

Limiares padrão: `min_share=0.02`, `min_active_weeks=4`. Variáveis abaixo são agrupadas em `__outros__ambiente`, `__outros__praca`, etc.

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
      - level: tipo
        groups: grupos
        members_key: ambientes
        attr: tipo
  Praca:
    rollups:
      - level: estado
        map: praca_to_estado
      - level: praca       # identity rollup
```

Para adicionar um novo veículo com hierarquia diferente, só alterar o YAML — sem mudança de código Python.

### Validação Semi-Sintética

`synthetic_data.py` gera dados com parâmetros Hill conhecidos, permitindo validar recuperação de shares antes de rodar dados reais. O benchmark varia `share_prior_scale` e qualidade da medição auxiliar (`σ_meas`).

### Tema Visual Uncover

`UNCOVER_DARK_TEMPLATE` (fundo `#141414`, papel `#1E1E1E`, fonte `#E0E0E0`) aplicado por default. Paleta de 8 cores consistente em todos os plots.

---

## Hipóteses e Fundamentação Metodológica

### H1 — Prior Auxiliar Melhora Recuperação de Shares ✅

Notebook de teste: `deep_dive_eletro.ipynb`: Benchmark sintético com 40 cenários (K=4 sub-canais, T=52 semanas, `share_prior_scale` ∈ {0.001, 0.005, 0.01, 0.05}, `σ_meas` ∈ {0, 0.05, 0.1, 0.2}):

- Sem prior auxiliar (baseline spend): MAE ≈ 0.05–0.07 (`scale=0.005`)
- Com brand study perfeito (`σ_meas=0`): MAE ≈ 0.03 → menor variância e estimativas mais concentradas
- Ponto ótimo prático (`σ_meas=0.1`, `scale=0.005`): MAE ≈ 0.04

Conclusão: prior auxiliar consistentemente reduz MAE, especialmente quando `σ_meas ≤ 0.10`. Ver `outputs/benchmark_share_recovery.csv` para resultados completos por cenário.

### H2 — Normalização do Proxy por `max(y)` Evita Explosão de Gradiente

O modelo Deep Dive Raven combina saídas Hill (escala [0, 1]) com a âncora de proxy (escala absoluta, ex: R$ 10M). Sem normalização, o proxy domina o gradiente e o otimizador diverge. A normalização `X_proxy = C_t / max(C_t)` coloca o proxy na mesma escala dos inputs Hill.

### H3 — CSL com Escala por Canal Evita Tuning Manual

A escala do prior CSL é calibrada por canal como:

```
proxy_scale_v = tolerance × mean(C_t_v≠0) / max(C_t) / (max(C_t_v) / max(C_t))
```

Isso garante que variáveis com contribuições maiores tenham prior mais rígido — sem exigir ajuste manual por cliente.

---

## Premissas para o Deep Dive Funcionar

1. **`C_t` é verdade**: a contribuição total do veículo calculada pelo MMM original é tratada como âncora fixa. Erros no modelo base propagam para o Deep Dive.

2. **Spend disponível por sub-canal na granularidade da quebra**: `load_breakdown_spend()` precisa encontrar as colunas com slugs correspondentes às variáveis de cada dimensão. Se o dado de spend não existir, a dimensão é pulada.

3. **Modelo base convergiu**: proxy_ratio deve ficar entre 0.85–1.15 para confiar nos shares. Fora desse intervalo, revisar `proxy_ct_tolerance` ou a qualidade de `C_t`.

4. **Frequência semanal**: todos os índices são normalizados para W-MON (semana terminada na segunda-feira). Séries diárias ou mensais não são suportadas nativamente.

5. **Hill é adequado para a relação spend→contribuição**: assume saturação monotônica crescente. Formas em U ou com threshold precisariam de modificação na arquitetura.

6. **`share_prior_scale` deve ser calibrado por veículo**: o default 0.05 é conservador (prior fraco). Para veículos com dados de brand study confiáveis, reduzir para 0.005–0.01.

7. **Variáveis com <2% de spend são descartadas ou agrupadas**: se um sub-canal muito pequeno for crítico para análise, aumentar `min_share` em `run_diagnostics()` ou removê-lo do filtro.

---

## Escolhas Metodológicas

| Decisão | Alternativa Considerada | Motivo da Escolha |
|---|---|---|
| Proxy exact (tolerância ±15%) | Proxy proporcional (escala livre) | Proporcional tem fator de escala não identificado → pode tornar shares arbitrárias |
| CSL em espaço de shares (Normal) | Log-ratio | Normal evita singularidades em share=0 |
| MAP com CosineScheduleAdamW | Adam (lr fixo) | Weight decay regulariza `max_effect` e evita explosão em canais com pouco spend; cosine decay estabiliza convergência na fase final sem tuning manual de lr |
| PiecewiseLinearTrend no Deep Dive Raven | FlatTrend | Sub-canais podem ter dinâmicas independentes; piecewise detecta breakpoints locais |
| Rollups declarativos em YAML | Código Python por veículo | Extensível sem mudança de código; YAML novos por veículo e clietes são suficientes |

---

## Fluxo de Execução do Teste - Exemplos

### Single-Client (`deep_dive_eletro.ipynb`)

```python
upgrade   = load_upgrade_auto(run_id, workspace, mlflow_uri, model_type)
config    = build_config(upgrade, specs_path="configs/bradesco_eletro.yaml")
config, _ = run_diagnostics(config, upgrade)   # filtra variáveis, cria __outros__
result    = run_deep_dive_e1(config, upgrade)  # Deep Dive Raven por dimensão
_         = analyze_deepdive(result)           # tabelas ASCII + plots
            generate_report(result, output_dir)
```

### Multi-Client (`deep_dive_batch.ipynb`)

```python
all_results, diags, errors = run_deep_dive_batch("configs/clients_registry.yaml")
df_meta   = consolidate_results(all_results, vehicle_spec_override=vehicle_spec)
batch_figs = analyze_batch(all_results, df_meta, vehicle_spec_override=vehicle_spec)
tree_figs  = analyze_trees(all_results, vehicle_spec_override=vehicle_spec)
```

---

## Configuração de um Novo Cliente

### 1. Criar YAML do cliente

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

### 2. Registrar no registry

```yaml
# configs/clients_registry.yaml
clients:
  novo_cliente:
    specs_path: novo_cliente_eletro.yaml
    model_type: stan   # ou meridian
    output_subdir: novo_cliente
```

### 3. Para um novo veículo (não-Eletromídia)

Adicionar entrada em `data/vehicle_specs.yaml` com `breakdowns`, `hierarchy` e `rollups`. O código Python não precisa de alteração.

---

## Entregáveis

### Por Cliente

| Arquivo | Conteúdo |
|---|---|
| `outputs/{cliente}/{cliente}_shares_e1.csv` | dim, item, contrib_share, spend_share, proxy_ratio, csl_max_dev |
| `outputs/{cliente}/{cliente}_roas_index.csv` | dim, item, roas_index (contrib_share / spend_share) |
| `outputs/{cliente}/{cliente}_contributions.html` | Barra agrupada: share de contribuição vs. share de spend |
| `outputs/{cliente}/{cliente}_roas_index.html` | Heatmap ROAS index por dimensão × sub-canal |

### Batch (Multi-Cliente)

| Arquivo | Conteúdo |
|---|---|
| `outputs/batch/meta_analysis.csv` | Long-form: cliente, dim, rollup, item, share_model, share_spend, roas_index |
| `outputs/batch/report_{Dim}.html` | Tabelas comparativas + sunburst por dimensão |

### Benchmark

| Arquivo | Conteúdo |
|---|---|
| `outputs/benchmark_share_recovery.csv` | scale, scenario (baseline / s=0.05 / ...), seed, mae, rmse, max_err, proxy_ratio |

---

## Testes

```bash
# Testes rápidos
pytest deepdive/tests/ -v

# Incluir teste de integração (MAP 500 steps, ~2min)
pytest deepdive/tests/ -v -m slow

# Benchmark completo (40 cenários, ~20min)
python deepdive/benchmarks/share_recovery_benchmark.py
```

| Arquivo de Teste | O Que Cobre |
|---|---|
| `test_config.py` | Parsing de YAML, defaults do dataclass |
| `test_extraction.py` | Mock MLflow, campos do UpgradeResult |
| `test_diagnostics.py` | Filtro por spend, bucketing __outros__, colunas do spend_report |
| `test_plots.py` | Figuras Plotly geradas, template dark |
| `test_report.py` | Criação de arquivos CSV e HTML |
| `test_pipeline_helpers.py` | _align_to, _wmon_norm (Period e Datetime) |
| `test_synthetic_deepdive.py` | Hill function, SyntheticDimension, recuperação de shares (slow) |

---

## Dependências Principais

- `mmmverse` — Raven (incluindo PiecewiseLinearTrend, MAPInferenceEngine)
- `prophetverse` — BaseEffect (usado pelo ContributionShareLikelihood)
- `jax / jaxlib` — backend numérico do Raven
- `mlflow` — carregamento de modelos e artefatos
- `plotly` — visualizações (dark theme)
- `pandas / numpy` — manipulação de dados
- `pyyaml` — configuração declarativa

---

## Decisões de Design Futuras

- Mover `ContributionShareLikelihood` para `prophetverse` como efeito nativo
- Mover `extra_effects` + `prior dicts` para `mmmverse` como API nativa do Raven
- Classe `RavenDeepDive` em `mmmverse` que encapsula o fluxo completo do Deep Dive
- Implementar `load_raven_upgrade()` para carregar contribuições de upgrade de modelos Raven para o Deep Dive
