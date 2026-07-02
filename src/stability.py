from __future__ import annotations

import contextlib
import io
import warnings
from unittest.mock import patch

import jax
import jax.random as jr
import pandas as pd

from config import DeepDiveConfig
from extraction import UpgradeResult
from pipeline import DDResult, run_deep_dive


def run_stability_test(
    config: DeepDiveConfig,
    upgrade: UpgradeResult,
    seeds: list[int] | None = None,
    instability_threshold: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, DDResult]]:
    """Run MAP stability test across multiple JAX seeds.

    Returns (df_stab, stats_stab, runs).
    df_stab: long-form contrib_share per (seed, dim, item).
    stats_stab: mean/std/range/cv per (dim, item).
    runs: raw DDResult per seed (used for downstream plots).
    """
    if seeds is None:
        seeds = [0, 42, 123, 456, 789]

    _orig_prng = jr.PRNGKey
    n = len(seeds)

    print(f"Stability test  seeds={seeds}  dims={list(config.dims)}")

    runs: dict[int, DDResult] = {}
    for i, seed in enumerate(seeds, 1):
        print(f"  [{i}/{n}] seed={seed} ...", end=" ", flush=True)
        jax.clear_caches()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                with patch("jax.random.PRNGKey", new=lambda s, _s=seed, _o=_orig_prng: _o(_s)):
                    runs[seed] = run_deep_dive(config, upgrade, verbose=False)
        pr_str = "  ".join(f"{d}={v:.3f}" for d, v in runs[seed].proxy_ratios.items())
        print(f"OK  proxy_ratios: {pr_str}")

    records = [
        {"seed": seed, "dim": dim, "item": item, "contrib_share": float(share)}
        for seed, res in runs.items()
        for dim, model_shares in res.shares_model.items()
        for item, share in model_shares.items()
    ]
    df_stab = pd.DataFrame(records)

    stats_stab = (
        df_stab.groupby(["dim", "item"])["contrib_share"]
        .agg(mean="mean", std="std", min="min", max="max")
        .assign(range=lambda x: x["max"] - x["min"])
        .assign(cv=lambda x: x["std"] / x["mean"].clip(lower=1e-9))
        .reset_index()
        .sort_values(["dim", "std"], ascending=[True, False])
    )

    summary = (
        stats_stab.groupby("dim")
        .agg(max_std=("std", "max"), mean_std=("std", "mean"), max_range=("range", "max"))
        .reset_index()
        .sort_values("max_std", ascending=False)
    )
    unstable = stats_stab[stats_stab["std"] > instability_threshold]

    print(f"\nConcluído: {n} runs.")
    print("\nEstabilidade por dimensão:")
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    if len(unstable):
        print(f"\n⚠  Sub-canais instáveis (std > {instability_threshold}):")
        print(unstable[["dim", "item", "mean", "std"]].to_string(index=False, float_format="{:.4f}".format))
    else:
        print(f"\n[ok] Nenhum sub-canal instável (std > {instability_threshold}) — MAP estável.")

    return df_stab, stats_stab, runs
