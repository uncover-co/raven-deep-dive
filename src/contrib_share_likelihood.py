"""ContributionShareLikelihood — soft prior on contribution shares vs metric shares.

Design rationale
----------------
Volume metrics (impressions, clicks, video-views) are always correlated with spend
because media is bought: more spend → more volume. Using them directly as proxy
targets does not add independent information beyond spend.

Efficiency metrics (CTR = clicks/impressions, VTR = views/impressions) capture
*quality* — how well each campaign type resonates — independently of spend level.

This effect constrains contribution shares to match metric shares:

    actual_share_i = Σ_t contrib_i(t) / Σ_t Σ_j contrib_j(t)
    expected_share_i = Σ_t metric_i(t) / Σ_t Σ_j metric_j(t)

    actual_shares ~ Normal(expected_shares, scale)          (vector, N items)

This is the share-space equivalent of a log-ratio constraint:

    log(contrib_i / contrib_ref) ≈ log(metric_i / metric_ref)

Without requiring a learnable scale factor k (as `proportional` proxy does),
so it does not suffer from k initialization mismatch → NaN losses.

Integration
-----------
Intended to be used alongside the `exact` proxy anchor on C_t:
  1. exact anchor: Σ contrib_j ≈ C_t  (total constrained to Raven 1 output)
  2. ContributionShareLikelihood: shares constrained to match metric shares

Together: contributions are proportional to metric shares AND sum to C_t.
This is equivalent to the per-item exact proxy approach but expressed purely
in share space, decoupled from the absolute scale of C_t.

The `scale` parameter plays the role of ε in the BreakdownModel formulation:
  - small scale (e.g. 0.01): shares tightly follow metrics
  - large scale (e.g. 0.10): model free to deviate based on spend patterns

Implementation note
-------------------
Metrics are passed directly via ``metric_df`` in ``__init__`` rather than being
read from X. This avoids Prophetverse's broadcast-by-column behavior when the
effect is passed via ``extra_effects``, which would otherwise deliver one column
at a time instead of the full (T, N) matrix needed to compute shares.
``requires_X=False`` signals to Prophetverse that this effect claims no X columns.
"""

from typing import Any, Dict, List

import pandas as pd
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from prophetverse.effects.base import BaseEffect


class ContributionShareLikelihood(BaseEffect):
    """Soft prior on contribution shares vs observed metric shares.

    Observes the time-aggregated contribution share of each item against
    the corresponding time-aggregated metric share. No learnable scale factor
    — uses the metric directly as the expected share target.

    Metrics are provided via ``metric_df`` at construction time; the effect
    does not read any columns from X (``requires_X=False``). This makes it
    safe to add via ``extra_effects`` without triggering Prophetverse's
    per-column broadcast.

    Parameters
    ----------
    target_effect_names : list[str]
        Fully-qualified latent effect names to constrain, e.g.
        ``["latent/contribution/media/branding",
           "latent/contribution/media/performance"]``.
        Must be in the same order as columns in ``metric_df``.
    metric_df : pd.DataFrame
        DataFrame with columns = raw metric values per item (same order as
        ``target_effect_names``). Index must align with the model's time index.
        Shares are computed as ``Σ_t metric_i / Σ_t Σ_j metric_j``.
    scale : float, default 0.05
        Standard deviation of the Normal likelihood on shares (0–1 scale).
        Rule of thumb: 0.05 allows ±5 pp deviation from the metric share
        at 1σ. Decrease to tighten, increase to loosen.
    name : str, optional
        Suffix used in the numpyro sample site name to avoid collisions
        when multiple instances are added to the same model.
        Defaults to an empty string.
    """

    _tags = {
        "capability:panel": False,
        "capability:multivariate_input": True,
        "requires_X": False,
        "applies_to": "X",
        "filter_indexes_with_forecating_horizon_at_transform": True,
    }

    def __init__(
        self,
        target_effect_names: List[str],
        metric_df: pd.DataFrame,
        scale: float = 0.05,
        name: str = "",
    ):
        self.target_effect_names = target_effect_names
        self.metric_df = metric_df
        self.scale = scale
        self.name = name
        super().__init__()

    def _transform(self, X: pd.DataFrame, fh: pd.Index) -> Any:
        """Return (T, N) matrix of metric values from stored DataFrame; ignores X."""
        return jnp.array(self.metric_df.values)

    def _predict(
        self,
        data: Any,
        predicted_effects: Dict[str, jnp.ndarray],
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Add share likelihood terms; return zeros."""
        # ── expected shares from metrics ──────────────────────────────────────
        # data: (T, N) raw metric values
        metric_totals = data.sum(axis=0)                       # (N,) sum over time
        grand_total = metric_totals.sum() + 1e-8
        expected_shares = metric_totals / grand_total          # (N,)

        # ── actual shares from latent contributions ───────────────────────────
        # predicted_effects[name] has shape (T, 1) or (T,)
        contribs = jnp.stack(
            [jnp.sum(predicted_effects[name]) for name in self.target_effect_names]
        )                                                       # (N,)
        total_contrib = contribs.sum() + 1e-8
        actual_shares = contribs / total_contrib               # (N,)

        # ── likelihood ────────────────────────────────────────────────────────
        suffix = f"_{self.name}" if self.name else ""
        numpyro.sample(
            f"share_prior{suffix}:ignore",
            dist.Normal(expected_shares, self.scale),
            obs=actual_shares,
        )

        return jnp.zeros((data.shape[0], 1))
