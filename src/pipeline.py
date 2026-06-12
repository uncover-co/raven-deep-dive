from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote
import jax
import numpy as np
import pandas as pd

from contrib_share_likelihood import ContributionShareLikelihood
from raven_patch import Raven, PROXY_SCALE_FALLBACK, CosineScheduleAdamWOptimizer
from config import DeepDiveConfig
from extraction import UpgradeResult

from mmmverse.models.raven import PiecewiseLinearTrend, MAPInferenceEngine
from prophetverse.effects.trend import FlatTrend


@dataclass
class DDResult:
    models: dict[str, Any]
    contribs: dict[str, pd.DataFrame]
    shares_model: dict[str, pd.Series]
    shares_spend: dict[str, pd.Series]
    proxy_ratios: dict[str, float]
    csl_devs: dict[str, float]
    eletro_contrib: pd.Series
    config: DeepDiveConfig
    features_raw: dict[str, pd.DataFrame] = field(default_factory=dict)
    col_maxes: dict[str, pd.Series] = field(default_factory=dict)


def _wmon_norm(idx) -> pd.DatetimeIndex:
    if isinstance(idx, pd.PeriodIndex):
        return idx.to_timestamp().to_period("W-MON").end_time.normalize()
    return pd.DatetimeIndex(idx).to_period("W-MON").end_time.normalize()


def _align_to(src: pd.Series, target_idx) -> pd.Series:
    s = src.copy()
    s.index = _wmon_norm(s.index)
    s = s.reindex(_wmon_norm(target_idx), fill_value=0)
    s.index = target_idx
    return s


def _apply_adstock_df(df: pd.DataFrame, decay: float) -> pd.DataFrame:
    result = df.copy()
    for col in df.columns:
        s = df[col].values.astype(float)
        adst = np.zeros(len(s))
        for t in range(len(s)):
            adst[t] = decay * adst[t - 1] + (1 - decay) * s[t] if t > 0 else s[t]
        mx_orig, mx_adst = s.max(), adst.max()
        if mx_adst > 0:
            adst = adst * mx_orig / mx_adst
        result[col] = adst
    return result


def _run_raven2_eletro(
    dim_name: str,
    features_df: pd.DataFrame,
    eletro_contrib: pd.Series,
    share_prior_scale: float = 0.05,
    proxy_ct_tolerance: float = 0.15,
    num_steps: int = 30_000,
    learning_rate: float = 0.001,
    use_piecewise_trend: bool = True,
    adstock_decay: float | None = None,
    auxiliary_metric_df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> dict:
    """Raven per dimension.
    Approach: raw features normalized per-column → Hill in [0,1].
    Target = eletro_contrib (not full KPI) → clean identification.
    """
    eletro_contrib = eletro_contrib.copy()
    eletro_contrib.index = _wmon_norm(eletro_contrib.index)
    features_df = features_df.copy()
    features_df.index = _wmon_norm(features_df.index)

    if eletro_contrib.sum() == 0:
        raise ValueError(f"[{dim_name}] eletro_contrib is all zeros.")
    _first_active = (eletro_contrib > 0).idxmax()
    n_dropped = int((eletro_contrib.index < _first_active).sum())
    eletro_contrib = eletro_contrib.loc[_first_active:]

    if verbose:
        _nz = int((eletro_contrib == 0).sum())
        print(f"  [{dim_name}] {eletro_contrib.index[0].date()} → "
              f"{eletro_contrib.index[-1].date()} "
              f"({len(eletro_contrib)}w, {n_dropped} dropped, {_nz} internal zeros)")

    variaveis = list(features_df.columns)
    y2 = eletro_contrib.to_frame(name="eletro")

    features_raw = features_df.reindex(eletro_contrib.index, fill_value=0)
    if adstock_decay is not None and adstock_decay > 0:
        features_raw = _apply_adstock_df(features_raw, adstock_decay)

    col_maxes = features_raw[variaveis].max(axis=0).replace(0, 1.0)
    features_norm = features_raw[variaveis].div(col_maxes)

    _y2_max = float(y2.values.max())
    _proxy_col = f"anchor_{dim_name.replace(' ', '_')}"
    _ct = eletro_contrib.reindex(y2.index, fill_value=0)
    _ct_nz = _ct[_ct > 0]

    X2 = features_norm.copy()
    X2[_proxy_col] = _ct.values / (_y2_max + 1e-12)

    _proxy_scale = (
        proxy_ct_tolerance * float(_ct_nz.mean()) / _y2_max
        / (float(_ct.abs().max()) / _y2_max + 1e-12)
        if len(_ct_nz) > 0 else PROXY_SCALE_FALLBACK
    )

    _csl = ContributionShareLikelihood(
        target_effect_names=[
            f"latent/contribution/media/{quote(v, safe='')}" for v in variaveis
        ],
        metric_df=(
            auxiliary_metric_df.reindex(y2.index).fillna(0)
            if auxiliary_metric_df is not None
            else features_raw[variaveis]
        ),
        scale=share_prior_scale,
        name=dim_name.replace(" ", "_"),
    )

    _trend = PiecewiseLinearTrend(changepoint_interval=52) if use_piecewise_trend else FlatTrend()

    raven2 = Raven(
        upper_funnel_variables=variaveis,
        lower_funnel_variables=[],
        proxy_variable_mapping={_proxy_col: variaveis},
        proxy_type={_proxy_col: "exact"},
        proxy_likelihood_scale=_proxy_scale,
        expected_roi=None,
        trend=_trend,
        seasonality_terms={"YE": 4, "ME": 2},
        seasonality_prior_scale={"YE": 0.2, "ME": 0.1},
        seasonality_mode="multiplicative",
        target_scale="max",
        extra_effects=[(f"share_prior_{dim_name.replace(' ', '_')}", _csl, None)],
        inference_engine=MAPInferenceEngine(
            optimizer=CosineScheduleAdamWOptimizer(
                init_value=learning_rate,
                decay_steps=num_steps,
                weight_decay=1e-4,
            ),
            num_steps=num_steps,
            stable_update=True,
            rng_key=jax.random.PRNGKey(0),
        ),
    )

    raven2.fit(y=y2, X=X2)
    _comps = raven2.predict_components(fh=y2.index, X=X2)

    contribs = pd.DataFrame(
        {v: _comps[f"latent/contribution/media/{v}"] for v in variaveis},
        index=y2.index,
    )

    _proxy_ratio = contribs.sum(axis=1).sum() / (_ct.sum() + 1e-12)
    _sh_mod = contribs.sum() / (contribs.sum().sum() + 1e-12)
    _sh_spend = features_raw[variaveis].sum() / (features_raw[variaveis].sum().sum() + 1e-12)
    _csl_max_dev = (_sh_mod - _sh_spend).abs().max()

    if verbose:
        print(f"  [{dim_name}] proxy_ratio={_proxy_ratio:.4f}  CSL_max_dev={_csl_max_dev:.3f}")

    return {
        "model": raven2,
        "components": _comps,
        "contribs": contribs,
        "proxy_ratio": float(_proxy_ratio),
        "csl_max_dev": float(_csl_max_dev),
        "shares_model": _sh_mod,
        "shares_spend": _sh_spend,
        "y2": y2,
        "X2": X2,
        "variaveis": variaveis,
        "features_raw": features_raw,
        "col_maxes": col_maxes,
    }


def run_deep_dive_e1(
    config: DeepDiveConfig,
    upgrade: UpgradeResult,
    auxiliary_metric_dfs: dict[str, pd.DataFrame] | None = None,
    strategy: Literal["e1", "y_adj"] = "e1",
    learning_rate: float = 0.001,
    verbose: bool = True,
) -> DDResult:
    """Run E1 or y_adj per dimension; collect into DDResult.

    strategy="e1"    : target = C_t_hat (contribuição do veículo no R1)
    strategy="y_adj" : target = residual_R1 + C_t_hat = y - (y_hat - C_t_hat)
                       Requer upgrade.y_actual preenchido no loader.
    """
    if config.media_var not in upgrade.contrib_df.columns:
        available = list(upgrade.contrib_df.columns)[:10]
        raise KeyError(
            f"media_var='{config.media_var}' not in contrib_df. "
            f"Available columns (first 10): {available}"
        )
    eletro_contrib = upgrade.contrib_df[config.media_var]

    if strategy not in ("e1", "y_adj"):
        raise ValueError(f"strategy deve ser 'e1' ou 'y_adj', recebido: '{strategy}'")

    if strategy == "y_adj":
        if upgrade.y_actual is None:
            raise ValueError(
                "strategy='y_adj' requer upgrade.y_actual. "
                "Verifique se o loader preencheu y_actual (Stan: stan_model.y / Meridian: y_true)."
            )
        y_adj_raw = upgrade.y_actual - upgrade.y_hat + eletro_contrib
        target = y_adj_raw.clip(lower=0)
        if verbose:
            ratio = target.sum() / (eletro_contrib.sum() + 1e-12)
            neg_weeks = int((y_adj_raw < 0).sum())
            print(f"[y_adj] ratio={ratio:.4f} (esperado ≈ 1.0)  semanas_negativas_clipadas={neg_weeks}")
    else:
        target = eletro_contrib

    models, contribs, shares_model, shares_spend = {}, {}, {}, {}
    proxy_ratios, csl_devs, features_raw_all, col_maxes_all = {}, {}, {}, {}

    for dim in config.dims:
        slugs = config.vars_per_dim.get(dim, [])
        if not slugs:
            print(f"[SKIP] {dim} — no variables after diagnostics")
            continue
        available = [s for s in slugs if s in upgrade.spend_df.columns]
        if not available:
            print(f"[SKIP] {dim} — no columns in spend_df")
            continue

        print(f"▶ [{dim}]  ({len(available)} vars)")
        _aux = (auxiliary_metric_dfs or {}).get(dim)
        r = _run_raven2_eletro(
            dim_name=dim,
            features_df=upgrade.spend_df[available].copy(),
            eletro_contrib=target,
            share_prior_scale=config.share_prior_scale,
            proxy_ct_tolerance=config.proxy_ct_tolerance,
            num_steps=config.num_steps,
            learning_rate=learning_rate,
            verbose=verbose,
            auxiliary_metric_df=_aux,
        )

        models[dim] = r["model"]
        contribs[dim] = r["contribs"]
        shares_model[dim] = r["shares_model"]
        shares_spend[dim] = r["shares_spend"]
        proxy_ratios[dim] = r["proxy_ratio"]
        csl_devs[dim] = r["csl_max_dev"]
        features_raw_all[dim] = r["features_raw"]
        col_maxes_all[dim] = r["col_maxes"]

    return DDResult(
        models=models,
        contribs=contribs,
        shares_model=shares_model,
        shares_spend=shares_spend,
        proxy_ratios=proxy_ratios,
        csl_devs=csl_devs,
        eletro_contrib=eletro_contrib,
        config=config,
        features_raw=features_raw_all,
        col_maxes=col_maxes_all,
    )


def extract_hill_params(
    raven2_model,
    variaveis: list[str],
    features_raw: pd.DataFrame | None = None,
    col_maxes: pd.Series | None = None,
    y_max: float | None = None,
) -> pd.DataFrame:
    """Extract MAP Hill parameters per variable.

    Source: Deep_Dive_Eletromidia_Raven2.ipynb cell 34 — extract_hill_params.
    half_max_abs = half_max_norm * col_maxes[v]  → BRL/semana.
    """
    import jax.numpy as jnp
    posterior = raven2_model.model_.inference_engine_.posterior_samples_
    records = []
    for var in variaveis:
        _q = quote(var, safe="")
        _keys = {k for k in posterior if _q in k}

        def _mean(suffix, _k=_keys):
            key = next((k for k in _k if k.endswith(suffix)), None)
            return float(jnp.mean(posterior[key])) if key else None

        me, hm, sl = _mean("/max_effect"), _mean("/half_max"), _mean("/slope")
        raw_max = float(col_maxes[var]) if col_maxes is not None and var in col_maxes else None
        hm_abs = hm * raw_max if hm is not None and raw_max else None
        active = (
            features_raw[var][features_raw[var] > 0]
            if features_raw is not None and var in features_raw.columns else None
        )
        mean_spend = float(active.mean()) if active is not None and len(active) > 0 else None
        mean_norm = (mean_spend / raw_max) if mean_spend and raw_max else None
        sat = (
            mean_norm**sl / (hm**sl + mean_norm**sl)
            if all(v is not None for v in [hm, sl, mean_norm]) and hm > 0 else None
        )
        rec: dict[str, Any] = {
            "variable": var, "max_effect": me,
            "half_max_norm": hm, "half_max_spend": hm_abs,
            "slope": sl, "mean_spend": mean_spend, "saturation_at_mean": sat,
        }
        if y_max and me:
            rec["max_effect_kpi"] = me * y_max
        records.append(rec)
    return pd.DataFrame(records).set_index("variable")
