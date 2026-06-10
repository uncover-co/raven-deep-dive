"""
Patch local sobre mmmverse.models.raven.Raven — extensões para o pipeline de Deep Dive.

Mantém a assinatura completa de _BaseRaven.__init__ (sem *args/**kwargs) para
compatibilidade com skbase._get_init_signature() e o mixin pydantic de spec.
Atributos extras são gravados em self ANTES de super().__init__() porque validate()
é chamado internamente pelo super e precisa acessá-los.

════════════════════════════════════════════════════════════════════════════════
1. NORMALIZAÇÃO DOS PROXIES POR y2_max (E2 / multi-proxy)
════════════════════════════════════════════════════════════════════════════════

Problema original
-----------------
Com target_scale="max" o Raven escala internamente y → y / max(y). Todos os
efeitos Hill produzem valores em unidades de y_scaled = y / y2_max. O canal de
proxy precisa estar na mesma escala para que CoupledExactLikelihood compare
grandezas compatíveis:

    Normal(sum_Hills_t, prior_scale).log_prob(proxy_col_t)

Se proxy_col_v é normalizado pelo seu próprio máximo (→ max = 1) mas o Hill
correspondente opera em unidades de y_scaled (≈ c_v / y2_max ≈ 0.002 para um
canal que representa 0.2 % do KPI), o mismatch de escala é da ordem de
1 / 0.002 = 500×. Isso gera gradientes explosivos → NaN nas primeiras iterações.

Solução
-------
Normalizar TODOS os proxies por y2_max = max(KPI no período):

    proxy_col_v_t = c_v_t / y2_max

Hills em y_scaled e proxies ficam na mesma escala por construção. Qualquer
normalização consistente funciona — y2_max é a escolha natural porque é o fator
que o próprio Raven usa internamente via target_scale="max".

════════════════════════════════════════════════════════════════════════════════
2. proxy_likelihood_scale POR CANAL
════════════════════════════════════════════════════════════════════════════════

Problema com scalar único
-------------------------
O parâmetro original proxy_likelihood_scale é um único float aplicado a todos os
proxies. O prior_scale efetivo de cada canal é:

    prior_scale_v = proxy_likelihood_scale * max(|proxy_col_v|)
                  = pls * max(c_v) / y2_max

Para que a tolerância relativa (1-sigma como fração da contribuição típica) seja
equal para canais de escalas radicalmente diferentes seria necessário:

    pls_v = tolerance * mean(c_v_nz) / max(c_v)

Como pls_v varia por canal, um scalar único não pode satisfazer todos
simultaneamente — canais pequenos ficam sub-constrangidos, canais grandes podem
ter restrição impossível de cumprir.

Derivação da fórmula de pls_v
------------------------------
Queremos prior_scale_v tal que 1-sigma = tolerance * mean contribuição semanal:

    prior_scale_v  = tolerance * mean(c_v_nz) / y2_max        ... (i)
    prior_scale_v  = pls_v * max(|proxy_col_v|)
                   = pls_v * max(c_v) / y2_max                ... (ii)

Igualando (i) e (ii):

    pls_v = tolerance * mean(c_v_nz) / max(c_v)

y2_max cancela. A fórmula é independente de qualquer normalização externa e
funciona para qualquer canal ou cliente.

Implementação
-------------
proxy_likelihood_scale aceita float (comportamento original) OU
dict {proxy_var_name: float}. Quando dict, _create_proxy_likelihood_effects
sobrescreve effect.prior_scale em cada CoupledExactLikelihood após o super()
construí-los:

    effect.prior_scale = pls_dict[proxy_var] * max(|proxy_col|)

Helpers públicos:
  compute_proxy_likelihood_scale(series, tolerance) → pls
  auto_proxy_tolerance(series, alpha, min_tol, max_tol) → tolerance

════════════════════════════════════════════════════════════════════════════════
3. CALIBRAÇÃO AUTOMÁTICA DE TOLERÂNCIA VIA CV (auto_proxy_tolerance)
════════════════════════════════════════════════════════════════════════════════

Problema com tolerância uniforme
---------------------------------
Usar tolerance = t para todos os canais ignora que canais com padrão spiky
(TV com sazonalidade forte, OOH por burst) têm coeficiente de variação alto —
o Hill não consegue replicar picos que não têm sinal de spend correspondente.
Exigir prior_scale muito apertado nesses canais torna a otimização intratável.

Solução baseada em CV
---------------------
    CV_v = std(c_v_nz) / mean(c_v_nz)
    tolerance_v = clip(alpha * CV_v, min_tol, max_tol)

Interpretação: "a tolerância permitida é proporcional à variação natural da
contribuição Stan do canal". Canal flat (TV contínua, CV ≈ 0.3) → ancoragem
forte (~3–6 %). Canal spiky (OOH por burst, CV ≈ 1.5) → ancoragem mais folgada
(~15–20 %), refletindo que o Hill não pode rastrear picos sem sinal de spend.

Parâmetros padrão (alpha=0.5, min_tol=0.02, max_tol=0.20) foram escolhidos para
que a faixa de tolerâncias resultante (~2–20 %) seja numericamente estável com
stable_update=True e ao mesmo tempo informativa o suficiente para ancorar os
canais secundários no pipeline de E2.

════════════════════════════════════════════════════════════════════════════════
4. AnchoredContribEffect — CANAIS SECUNDÁRIOS COM FORMA FIXA (E2)
════════════════════════════════════════════════════════════════════════════════

Problema com CoupledExactLikelihood para canais secundários
------------------------------------------------------------
CoupledExactLikelihood impõe Normal(Hill_t, sigma) ~ proxy_t por step temporal.
Com 20 canais correlacionados, o otimizador faz trade-off entre eles dentro do
budget de sigma — cada canal pode compensar outro. Resultado: drift real >> sigma,
independente de quão apertado seja proxy_likelihood_scale.

Causa raiz: T parâmetros livres por canal (os Hills), T soft-constraints — muito
espaço de solução degenerado.

Solução: AnchoredContribEffect
---------------------------------
Substitui a estimação Hill + proxy por um efeito de FORMA FIXA com UM único
parâmetro de escala global por canal:

    contrib_v_t = alpha_v * c_v_t / y2_max
    alpha_v ~ TruncatedNormal(1.0, sigma, low=0)

Propriedades:
- Drift máximo ≈ 3 * sigma (tail da TruncatedNormal); com sigma=0.10 → max ~30%
  mas o MAP converge para alpha ≈ 1.0 salvo necessidade do KPI fit
- Um parâmetro por canal elimina a degenerescência temporal — sem trade-off entre
  steps, apenas entre escalas globais
- Sem NaN: a prior é suave e os valores escalados são fixos em y_scaled units
- Forma temporal do Stan é preservada; só o nível varia

Uso em run_e2_multiproxy:
  - Eletromídia: Hill + CSL + proxy_exact (igual E1)
  - Outros canais: AnchoredContribEffect (forma Stan ± sigma)
  - y_target: KPI completo (não Y_adj)

════════════════════════════════════════════════════════════════════════════════
5. PER-VARIABLE HILL PRIOR DICTS
════════════════════════════════════════════════════════════════════════════════

upper/lower_funnel_{max_effect,half_max,slope}_prior_dict:
  Aceitam {var_name: numpyro.distributions.Distribution} e sobrescrevem as
  priors de HillEffect instâncias retornadas por
  super()._create_{upper,lower}_funnel_effects(). Variáveis ausentes no dict
  mantêm o prior global definido no construtor do Raven.
"""
import numpy as np
import pandas as pd
import optax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from urllib.parse import unquote
from numpyro.optim import optax_to_numpyro
from prophetverse.engine.optimizer import BaseOptimizer
from prophetverse.effects.hill import HillEffect
from prophetverse.effects.coupled import CoupledExactLikelihood
from mmmverse.models.raven import (
    Raven as _BaseRaven,
    DEFAULT_CONTROL_PRIOR_SCALE,
    DEFAULT_POSITIVE_CONTROL_PRIOR_SCALE,
    DEFAULT_NEGATIVE_CONTROL_PRIOR_SCALE,
    DEFAULT_SEASONALITY_PRIOR_SCALE,
    DEFAULT_SEASONALITY_TERMS,
    DEFAULT_EXPECTED_ROI,
    DEFAULT_EXPECTED_ROI_SCALE,
    DEFAULT_PROXY_LIKELIHOOD_SCALE,
    DEFAULT_TARGET_SCALE,
    DEFAULT_SEASONALITY_MODE,
)

_PROXY_LIKELIHOOD_PREFIX = "latent/proxy_likelihood/"

# Fallback value for proxy_likelihood_scale when a channel's contribution series
# is all-zero (cannot compute tolerance-based scale).  Equals the mmmverse default.
PROXY_SCALE_FALLBACK: float = DEFAULT_PROXY_LIKELIHOOD_SCALE


def auto_proxy_tolerance(
    series: pd.Series,
    alpha: float = 0.5,
    min_tol: float = 0.02,
    max_tol: float = 0.20,
) -> float:
    """CV-based per-channel tolerance for proxy constraints.

    tolerance_v = clip(alpha * CV_v, min_tol, max_tol)
    CV_v = std(nz) / mean(nz)

    Flat/smooth channels (low CV) → tight anchorage.
    Spiky/seasonal channels (high CV) → looser tolerance — the Hill
    function cannot replicate peaks it has no spend signal for.

    Parameters
    ----------
    series : pd.Series
        Raw contribution series (any unit). Zeros excluded.
    alpha : float
        Scaling factor applied to CV (default 0.5).
    min_tol : float
        Floor tolerance (default 0.02 = 2 %).
    max_tol : float
        Ceiling tolerance (default 0.20 = 20 %).

    Returns
    -------
    float
        Tolerance in [min_tol, max_tol].
    """
    nz = series[series > 0]
    if len(nz) < 2:
        return min_tol
    cv = float(nz.std() / nz.mean())
    return float(np.clip(alpha * cv, min_tol, max_tol))


def compute_proxy_likelihood_scale(
    series: pd.Series,
    tolerance: float,
    fallback: float = DEFAULT_PROXY_LIKELIHOOD_SCALE,
) -> float:
    """Compute proxy_likelihood_scale for one channel so that 1-sigma ≈ tolerance.

    Works regardless of how the proxy column is normalised (y2_max, col_max, etc.)
    because the y_scale factor cancels in the ratio mean/max.

    The CoupledExactLikelihood prior_scale is:
        prior_scale = pls * max(|proxy_col|)

    Setting prior_scale = tolerance * mean(series_nz) / y_scale and
    max(proxy_col) = max(series) / y_scale gives:
        pls = tolerance * mean(series_nz) / max(series)   (y_scale cancels)

    Parameters
    ----------
    series : pd.Series
        Raw contribution series for the channel (any unit, e.g. KPI currency).
        Zeros/negatives are excluded when computing the mean.
    tolerance : float
        Desired 1-sigma tolerance as a fraction of the channel's typical contribution
        (e.g. 0.10 for ±10 % of the mean non-zero contribution per time step).
    fallback : float
        Value returned when the series is all-zero or has no positive values.

    Returns
    -------
    float
        proxy_likelihood_scale to pass to Raven.
    """
    col_max = float(series.max())
    if col_max <= 0:
        return fallback
    nz = series[series > 0]
    if len(nz) == 0:
        return fallback
    return tolerance * float(nz.mean()) / col_max


from prophetverse.effects.base import BaseEffect as _BaseEffect


class AnchoredContribEffect(_BaseEffect):
    """Fixed-shape contribution anchored to a Stan estimate with a global scale prior.

    Replaces Hill + CoupledExactLikelihood for secondary channels in E2.

    Model
    -----
        contrib_t = alpha * values_scaled_t
        alpha ~ TruncatedNormal(1.0, sigma, low=0)

    With sigma=0.10 the MAP estimate of alpha will be 1.0 unless the KPI fit
    strongly prefers a different level. Drift is bounded at the tail of the prior
    (~3 sigma ≈ 30 % for sigma=0.10); in practice the MAP stays close to 1.0.

    Key difference from CoupledExactLikelihood
    -------------------------------------------
    CEL has T soft constraints (one per timestep) on T Hill parameters — the
    optimizer can exploit collinearity across channels to escape.  AnchoredContrib
    has ONE parameter (alpha) per channel; there is no per-step degree of freedom
    to trade against other channels.

    Parameters
    ----------
    values_scaled : array-like, shape (n_timesteps,)
        Stan contribution series pre-divided by y2_max (i.e. in y_scaled units).
    sigma : float
        Std dev of the scale prior (0.10 ≈ ±10 % at 1 sigma).
    effect_name : str
        Unique numpyro sample site name (must be unique within the model).
    """

    _tags = {"requires_X": False, "hierarchical_prophet_compliant": False}

    def __init__(
        self,
        values_scaled: np.ndarray,
        sigma: float = 0.10,
        effect_name: str = "anchored_contrib",
    ):
        self.values_scaled = np.asarray(values_scaled, dtype=float)
        self.sigma = sigma
        self.effect_name = effect_name
        super().__init__()

    def _fit(self, y, X, scale=1.0):
        pass

    def _transform(self, X, fh):
        return {}

    def _predict(self, data, predicted_effects, *args, **kwargs):
        alpha = numpyro.sample(
            self.effect_name + "_scale",
            dist.TruncatedNormal(1.0, self.sigma, low=0.0),
        )
        # reshape to (n, 1) to match prophetverse effect convention;
        # returning (n,) triggers (n,1)+(n,) → (n,n) broadcast = shape 18769
        return jnp.array(self.values_scaled).reshape(-1, 1) * alpha


class CosineScheduleAdamWOptimizer(BaseOptimizer):
    """Adam + cosine LR decay + L2 weight decay (AdamW).

    Drop-in for CosineScheduleAdamOptimizer. Weight decay regularises Hill
    parameters and prevents max_effect from exploding on low-spend items.

    Parameters
    ----------
    init_value   : initial learning rate
    decay_steps  : cosine decay horizon (match num_steps)
    weight_decay : L2 penalty coefficient (default 1e-4)
    alpha        : final LR multiplier at end of schedule
    """

    def __init__(self, init_value=0.001, decay_steps=100_000,
                 weight_decay=1e-4, alpha=0.0):
        self.init_value   = init_value
        self.decay_steps  = decay_steps
        self.weight_decay = weight_decay
        self.alpha        = alpha
        super().__init__()

    def create_optimizer(self):
        scheduler = optax.cosine_decay_schedule(
            init_value=self.init_value,
            decay_steps=self.decay_steps,
            alpha=self.alpha,
        )
        return optax_to_numpyro(
            optax.adamw(learning_rate=scheduler, weight_decay=self.weight_decay)
        )


def _patch_hill_priors(effects, me_dict, hm_dict, sl_dict, prefix):
    """Patch HillEffect instances in-place with per-variable priors.

    Parameters
    ----------
    effects : list of (name, effect, selector) tuples from _create_*_funnel_effects
    me_dict : {var: Distribution} for max_effect_prior  (may be empty)
    hm_dict : {var: Distribution} for half_max_prior    (may be empty)
    sl_dict : {var: Distribution} for slope_prior       (may be empty)
    prefix  : str — name prefix used in that funnel ("latent/unadstocked/" for upper,
              "unadstocked/" for lower)
    """
    if not (me_dict or hm_dict or sl_dict):
        return
    for name, effect, _ in effects:
        if name.startswith(prefix) and isinstance(effect, HillEffect):
            var = name[len(prefix):]
            if var in me_dict:
                effect.max_effect_prior  = me_dict[var]
                effect._max_effect_prior = me_dict[var]
            if var in hm_dict:
                effect.half_max_prior  = hm_dict[var]
                effect._half_max_prior = hm_dict[var]
            if var in sl_dict:
                effect.slope_prior  = sl_dict[var]
                effect._slope_prior = sl_dict[var]


class Raven(_BaseRaven):
    def __init__(
        self,
        upper_funnel_variables,
        lower_funnel_variables,
        control_variables=None,
        positive_control_variables=None,
        negative_control_variables=None,
        proxy_variable_mapping=None,
        proxy_type=None,
        upper_funnel_max_effect_prior=None,
        upper_funnel_half_max_prior=None,
        upper_funnel_slope_prior=None,
        lower_funnel_max_effect_prior=None,
        lower_funnel_half_max_prior=None,
        lower_funnel_slope_prior=None,
        upper_funnel_adstock_effect=None,
        control_prior_scale=DEFAULT_CONTROL_PRIOR_SCALE,
        positive_control_prior_scale=DEFAULT_POSITIVE_CONTROL_PRIOR_SCALE,
        negative_control_prior_scale=DEFAULT_NEGATIVE_CONTROL_PRIOR_SCALE,
        seasonality_prior_scale=DEFAULT_SEASONALITY_PRIOR_SCALE,
        seasonality_terms=DEFAULT_SEASONALITY_TERMS,
        trend=None,
        inference_engine=None,
        expected_roi=DEFAULT_EXPECTED_ROI,
        expected_roi_scale=DEFAULT_EXPECTED_ROI_SCALE,
        proxy_likelihood_scale=DEFAULT_PROXY_LIKELIHOOD_SCALE,
        expected_contrib=None,
        expected_contrib_scale=None,
        seasonality_mode=DEFAULT_SEASONALITY_MODE,
        target_scale=DEFAULT_TARGET_SCALE,
        extra_effects=None,
        upper_funnel_max_effect_prior_dict=None,
        upper_funnel_half_max_prior_dict=None,
        upper_funnel_slope_prior_dict=None,
        lower_funnel_max_effect_prior_dict=None,
        lower_funnel_half_max_prior_dict=None,
        lower_funnel_slope_prior_dict=None,
    ):
        # set before super().__init__() — validate() runs inside super and reads these
        self.extra_effects = extra_effects
        self.upper_funnel_max_effect_prior_dict = upper_funnel_max_effect_prior_dict
        self.upper_funnel_half_max_prior_dict   = upper_funnel_half_max_prior_dict
        self.upper_funnel_slope_prior_dict      = upper_funnel_slope_prior_dict
        self.lower_funnel_max_effect_prior_dict = lower_funnel_max_effect_prior_dict
        self.lower_funnel_half_max_prior_dict   = lower_funnel_half_max_prior_dict
        self.lower_funnel_slope_prior_dict      = lower_funnel_slope_prior_dict

        # per-channel proxy_likelihood_scale: stash dict separately so the base class
        # (which expects a scalar) remains unaware of it
        if isinstance(proxy_likelihood_scale, dict):
            self._proxy_likelihood_scale_dict = proxy_likelihood_scale
            _pls_for_super = DEFAULT_PROXY_LIKELIHOOD_SCALE
        else:
            self._proxy_likelihood_scale_dict = None
            _pls_for_super = proxy_likelihood_scale

        super().__init__(
            upper_funnel_variables=upper_funnel_variables,
            lower_funnel_variables=lower_funnel_variables,
            control_variables=control_variables,
            positive_control_variables=positive_control_variables,
            negative_control_variables=negative_control_variables,
            proxy_variable_mapping=proxy_variable_mapping,
            proxy_type=proxy_type,
            upper_funnel_max_effect_prior=upper_funnel_max_effect_prior,
            upper_funnel_half_max_prior=upper_funnel_half_max_prior,
            upper_funnel_slope_prior=upper_funnel_slope_prior,
            lower_funnel_max_effect_prior=lower_funnel_max_effect_prior,
            lower_funnel_half_max_prior=lower_funnel_half_max_prior,
            lower_funnel_slope_prior=lower_funnel_slope_prior,
            upper_funnel_adstock_effect=upper_funnel_adstock_effect,
            control_prior_scale=control_prior_scale,
            positive_control_prior_scale=positive_control_prior_scale,
            negative_control_prior_scale=negative_control_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_terms=seasonality_terms,
            trend=trend,
            inference_engine=inference_engine,
            expected_roi=expected_roi,
            expected_roi_scale=expected_roi_scale,
            proxy_likelihood_scale=_pls_for_super,
            expected_contrib=expected_contrib,
            expected_contrib_scale=expected_contrib_scale,
            seasonality_mode=seasonality_mode,
            target_scale=target_scale,
        )

    # ------------------------------------------------------------------
    # Per-channel proxy_likelihood_scale
    # ------------------------------------------------------------------

    def _create_proxy_likelihood_effects(self, X):
        effects = super()._create_proxy_likelihood_effects(X)

        if not self._proxy_likelihood_scale_dict:
            return effects

        for name, effect, _selector in effects:
            if not (
                name.startswith(_PROXY_LIKELIHOOD_PREFIX)
                and isinstance(effect, CoupledExactLikelihood)
            ):
                continue
            proxy_var_quoted = name[len(_PROXY_LIKELIHOOD_PREFIX):]
            proxy_var = unquote(proxy_var_quoted)
            pls = self._proxy_likelihood_scale_dict.get(
                proxy_var, DEFAULT_PROXY_LIKELIHOOD_SCALE
            )
            proxy_scale = float(X[proxy_var_quoted].abs().max())
            effect.prior_scale = pls * proxy_scale

        return effects

    # ------------------------------------------------------------------
    # Per-variable Hill priors
    # ------------------------------------------------------------------

    def _create_upper_funnel_effects(self, X, y, upper_funnel_variables):
        effects = super()._create_upper_funnel_effects(X, y, upper_funnel_variables)
        _patch_hill_priors(
            effects,
            self.upper_funnel_max_effect_prior_dict or {},
            self.upper_funnel_half_max_prior_dict   or {},
            self.upper_funnel_slope_prior_dict      or {},
            prefix="latent/unadstocked/",
        )
        return effects

    def _create_lower_funnel_effects(self, X, y, lower_funnel_variables):
        effects = super()._create_lower_funnel_effects(X, y, lower_funnel_variables)
        _patch_hill_priors(
            effects,
            self.lower_funnel_max_effect_prior_dict or {},
            self.lower_funnel_half_max_prior_dict   or {},
            self.lower_funnel_slope_prior_dict      or {},
            prefix="unadstocked/",
        )
        return effects

    def _create_model(self, X, y):
        model = super()._create_model(X, y)
        if self.extra_effects:
            model.exogenous_effects = (
                list(model.exogenous_effects or []) + list(self.extra_effects)
            )
        return model
