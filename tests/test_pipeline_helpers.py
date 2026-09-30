import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from pipeline import align_to, wmon_norm


def test_align_to_reindex():
    src_idx = pd.date_range("2023-01-09", periods=5, freq="W-MON")
    tgt_idx = pd.date_range("2023-01-02", periods=7, freq="W-MON")
    src = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=src_idx)
    result = align_to(src, tgt_idx)
    assert len(result) == 7
    assert result.index.equals(tgt_idx)
    assert result.iloc[0] == 0.0   # fill_value=0 for missing prefix


def test_wmon_norm_datetime():
    idx = pd.date_range("2023-01-04", periods=3, freq="W-WED")
    result = wmon_norm(idx)
    assert all(ts.weekday() == 0 for ts in result)


def test_wmon_norm_period():
    idx = pd.period_range("2023-01", periods=3, freq="W")
    result = wmon_norm(idx)
    assert len(result) == 3


def _fake_raven(posterior):
    class _Engine:
        posterior_samples_ = posterior

    class _Inner:
        inference_engine_ = _Engine()

    class _Raven:
        model_ = _Inner()

    return _Raven()


def test_hill_params_do_not_leak_between_overlapping_slugs():
    """Key format taken from a real MAP fit: latent/unadstocked/<slug>/<param>,
    with panel-0/* and max_effect_scaled siblings alongside. Matching by
    substring picked up a slug contained in another ("video" inside
    "videoview"), and next() over a set then resolved in iteration order.
    """
    from urllib.parse import quote
    from pipeline import extract_hill_params

    P = "$metric:m$category:cat:"
    short, long_ = P + "video", P + "videoview"
    assert quote(short, safe="") in quote(long_, safe="")   # o overlap que causa o bug

    vals = {short: (1.0, 2.0, 3.0), long_: (10.0, 20.0, 30.0)}
    posterior = {}
    for var, (me, hm, sl) in vals.items():
        site = f"latent/unadstocked/{quote(var, safe='')}"
        posterior[f"{site}/max_effect"] = np.array([me])
        posterior[f"{site}/half_max"] = np.array([hm])
        posterior[f"{site}/slope"] = np.array([sl])
        # irmas que o fit real tambem loga
        posterior[f"{site}/max_effect_scaled"] = np.array([me * 100])
        posterior[f"{site}/panel-0/half_max"] = np.array([hm * 100])
        posterior[f"{site}/panel-0/slope"] = np.array([sl * 100])

    out = extract_hill_params(_fake_raven(posterior), [short, long_])

    for var, (me, hm, sl) in vals.items():
        row = out.loc[var]
        assert (row["max_effect"], row["half_max_norm"], row["slope"]) == (me, hm, sl)


def test_hill_params_are_none_when_the_key_is_ambiguous():
    """Two sites ending the same way means the slug cannot be resolved; the
    extractor returns None instead of picking one."""
    from urllib.parse import quote
    from pipeline import extract_hill_params

    var = "$metric:m$category:cat:a"
    q = quote(var, safe="")
    posterior = {
        f"latent/unadstocked/{q}/max_effect": np.array([1.0]),
        f"other/branch/{q}/max_effect": np.array([9.0]),
    }

    row = extract_hill_params(_fake_raven(posterior), [var]).loc[var]
    assert row["max_effect"] is None
