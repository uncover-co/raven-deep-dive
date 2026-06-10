"""Benchmark: share recovery accuracy vs. measurement prior quality.

Compares 5 scenarios for one synthetic dimension:
  baseline     — spend-share CSL (current method, no auxiliary data)
  σ_meas=0.00  — perfect measurement prior (true shares exactly)
  σ_meas=0.05  — tight prior (~5% std on log-share space)
  σ_meas=0.10  — moderate noise
  σ_meas=0.20  — loose prior

Metric: MAE and RMSE on shares (predicted vs. true), averaged over seeds.

Usage:
  cd /home/juliacanedo/projetos/raven_deep_dive
  .venv/bin/python deepdive/benchmarks/share_recovery_benchmark.py
  .venv/bin/python deepdive/benchmarks/share_recovery_benchmark.py --K 3 --T 26 --steps 500 --seeds 42
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from synthetic_data import generate_synthetic_dim, simulate_measurement_prior
from pipeline import _run_raven2_eletro


# ── core ─────────────────────────────────────────────────────────────────────

def _run_scenario(
    syn,
    auxiliary_metric_df,
    num_steps: int,
    share_prior_scale: float = 0.05,
) -> dict:
    r = _run_raven2_eletro(
        dim_name=syn.dim_name,
        features_df=syn.spend_df,
        eletro_contrib=syn.eletro_contrib,
        share_prior_scale=share_prior_scale,
        proxy_ct_tolerance=0.15,
        num_steps=num_steps,
        verbose=False,
        auxiliary_metric_df=auxiliary_metric_df,
    )
    diff = r["shares_model"] - syn.true_shares
    return {
        "mae":         float(diff.abs().mean()),
        "rmse":        float(np.sqrt((diff ** 2).mean())),
        "max_err":     float(diff.abs().max()),
        "proxy_ratio": float(r["proxy_ratio"]),
        "shares_hat":  r["shares_model"],
    }


def run_benchmark(
    K: int = 4,
    T: int = 52,
    num_steps: int = 1000,
    seeds: list | None = None,
    scales: list | None = None,
) -> pd.DataFrame:
    """Run full benchmark; return long-form DataFrame of metrics per scenario × seed × scale."""
    if seeds is None:
        seeds = [42, 99]
    if scales is None:
        scales = [0.05]

    sigmas = [None, 0.0, 0.05, 0.10, 0.20]
    rows = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2022-01-03", periods=T, freq="W-MON")
        cols = [f"praca_{i+1}" for i in range(K)]
        spend_df = pd.DataFrame(rng.uniform(100, 5000, (T, K)), index=idx, columns=cols)
        syn = generate_synthetic_dim("Praca", spend_df, rng_seed=seed)

        for scale in scales:
            print(f"\n▶ Seed {seed}  scale={scale}  (K={K}, T={T}, steps={num_steps})")
            print(f"  True shares: { {k: round(v,3) for k,v in syn.true_shares.items()} }")

            for sigma in sigmas:
                if sigma is None:
                    scenario = "baseline"
                    aux = None
                else:
                    scenario = f"s={sigma:.2f}"
                    aux = simulate_measurement_prior(
                        syn.true_shares, n_obs=T, sigma=sigma, rng_seed=seed,
                        index=idx,
                    )

                print(f"  {scenario:<12}", end=" ", flush=True)
                try:
                    m = _run_scenario(syn, aux, num_steps, share_prior_scale=scale)
                    print(f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  proxy={m['proxy_ratio']:.3f}")
                    rows.append({
                        "scale": scale, "scenario": scenario, "seed": seed,
                        "mae": m["mae"], "rmse": m["rmse"],
                        "max_err": m["max_err"], "proxy_ratio": m["proxy_ratio"],
                    })
                except Exception as e:
                    print(f"FAILED: {e}")
                    rows.append({
                        "scale": scale, "scenario": scenario, "seed": seed,
                        "mae": float("nan"), "rmse": float("nan"),
                        "max_err": float("nan"), "proxy_ratio": float("nan"),
                    })

    return pd.DataFrame(rows)


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    order = ["baseline", "s=0.00", "s=0.05", "s=0.10", "s=0.20"]
    scales = sorted(df["scale"].unique()) if "scale" in df.columns else [None]
    n_seeds = df["seed"].nunique()
    w = 84

    for scale in scales:
        sub = df[df["scale"] == scale] if "scale" in df.columns else df
        summary = (
            sub.groupby("scenario")[["mae", "rmse", "max_err", "proxy_ratio"]]
            .agg(["mean", "std"])
            .round(4)
        )
        summary.columns = ["_".join(c) for c in summary.columns]
        summary = summary.reindex([s for s in order if s in summary.index])

        print("\n" + "═" * w)
        print(f"  BENCHMARK  scale={scale}  ({n_seeds} seeds)")
        print("═" * w)
        print(f"  {'Cenário':<14}  {'MAE (mean)':>10}  {'±std':>6}  {'RMSE':>8}  {'MaxErr':>8}  {'Δ vs baseline':>16}")
        print("  " + "─" * (w - 2))

        baseline_mae = summary.loc["baseline", "mae_mean"] if "baseline" in summary.index else float("nan")

        for scenario, row in summary.iterrows():
            delta = ""
            if scenario != "baseline":
                d = baseline_mae - row["mae_mean"]
                pct = d / (baseline_mae + 1e-12) * 100
                delta = f"{d:+.4f} ({pct:+.1f}%)"
            print(
                f"  {scenario:<14}  {row['mae_mean']:>10.4f}  {row['mae_std']:>6.4f}"
                f"  {row['rmse_mean']:>8.4f}  {row['max_err_mean']:>8.4f}  {delta:>16}"
            )
        print("═" * w)
    print(f"\n  Δ positivo = prior auxiliar melhora recovery.")


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Share recovery benchmark")
    parser.add_argument("--K",      type=int,          default=4,             help="sub-channels (default 4)")
    parser.add_argument("--T",      type=int,          default=52,            help="weeks (default 52)")
    parser.add_argument("--steps",  type=int,          default=1000,          help="MAP steps (default 1000)")
    parser.add_argument("--seeds",  type=int, nargs="+", default=[42, 99],    help="random seeds")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.05],    help="share_prior_scale values (default 0.05)")
    args = parser.parse_args()

    df = run_benchmark(K=args.K, T=args.T, num_steps=args.steps, seeds=args.seeds, scales=args.scales)
    print_summary(df)

    out = os.path.join(os.path.dirname(__file__), "../outputs/benchmark_share_recovery.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  Salvo em: {out}")
