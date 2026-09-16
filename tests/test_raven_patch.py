import os
import sys
import numpyro.distributions as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from raven_patch import _patch_hill_priors


class _FakeHillEffect:
    """Duck-types the real Hill effect classes without needing mmmverse/prophetverse."""

    def __init__(self):
        self.max_effect_prior = "default_me"
        self.half_max_prior = "default_hm"
        self.slope_prior = "default_sl"


class _FakeNonHillEffect:
    """No half_max_prior attribute — e.g. a trend/seasonality effect that shares the prefix."""


def test_patch_hill_priors_matches_by_duck_typing_not_isinstance():
    hill = _FakeHillEffect()
    other = _FakeNonHillEffect()
    prefix = "latent/unadstocked/"
    effects = [
        (f"{prefix}channel_a", hill, None),
        (f"{prefix}channel_b", other, None),
    ]
    hm_prior = dist.TruncatedNormal(loc=1.0, scale=0.5, low=0.0)

    _patch_hill_priors(effects, me_dict={}, hm_dict={"channel_a": hm_prior}, sl_dict={}, prefix=prefix)

    assert hill.half_max_prior is hm_prior
    assert hill._half_max_prior is hm_prior
    assert hill.max_effect_prior == "default_me"
    assert not hasattr(other, "half_max_prior")


def test_patch_hill_priors_noop_when_dicts_empty():
    hill = _FakeHillEffect()
    prefix = "latent/unadstocked/"
    _patch_hill_priors([(f"{prefix}channel_a", hill, None)], me_dict={}, hm_dict={}, sl_dict={}, prefix=prefix)
    assert hill.half_max_prior == "default_hm"


def test_patch_hill_priors_ignores_vars_outside_prefix():
    hill = _FakeHillEffect()
    hm_prior = dist.TruncatedNormal(loc=1.0, scale=0.5, low=0.0)
    _patch_hill_priors(
        [("unadstocked/channel_a", hill, None)],
        me_dict={}, hm_dict={"channel_a": hm_prior}, sl_dict={},
        prefix="latent/unadstocked/",
    )
    assert hill.half_max_prior == "default_hm"
