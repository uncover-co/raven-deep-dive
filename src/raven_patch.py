"""
Patch local sobre mmmverse.models.raven.Raven — extensões para o pipeline de Deep Dive.

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
2. PER-VARIABLE HILL PRIOR DICTS
════════════════════════════════════════════════════════════════════════════════

upper/lower_funnel_{max_effect,half_max,slope}_prior_dict:
  Aceitam {var_name: numpyro.distributions.Distribution} e sobrescrevem as
  priors de HillEffect instâncias retornadas por
  super()._create_{upper,lower}_funnel_effects(). Variáveis ausentes no dict
  mantêm o prior global definido no construtor do Raven.
"""

import optax
from numpyro.optim import optax_to_numpyro
from prophetverse.engine.optimizer import BaseOptimizer
from prophetverse.effects.hill import HillEffect
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

# Fallback for proxy_likelihood_scale when channel contribution series is all-zero.
PROXY_SCALE_FALLBACK: float = DEFAULT_PROXY_LIKELIHOOD_SCALE


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
            proxy_likelihood_scale=proxy_likelihood_scale,
            expected_contrib=expected_contrib,
            expected_contrib_scale=expected_contrib_scale,
            seasonality_mode=seasonality_mode,
            target_scale=target_scale,
        )

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
