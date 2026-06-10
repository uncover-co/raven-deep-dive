from __future__ import annotations
import os
import numpy as np
import pandas as pd

from plots import plot_contributions, plot_roas_index


def generate_report(
    result,
    output_dir: str = "outputs/",
    client_name: str = "",
) -> dict[str, str]:
    """Export all artefacts: CSVs + interactive Plotly HTML.

    Returns dict of {key: absolute_file_path}.
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{client_name}_" if client_name else ""
    paths: dict[str, str] = {}

    # ── shares_e1.csv ─────────────────────────────────────────────────────────
    shares_rows = []
    for dim in result.config.dims:
        sh_m = result.shares_model.get(dim, pd.Series(dtype=float))
        sh_s = result.shares_spend.get(dim, pd.Series(dtype=float))
        proxy_r = result.proxy_ratios.get(dim, float("nan"))
        csl_d = result.csl_devs.get(dim, float("nan"))
        for item in sh_m.index:
            shares_rows.append({
                "dim": dim,
                "item": item,
                "contrib_share": float(sh_m[item]),
                "spend_share": float(sh_s.get(item, 0.0)),
                "proxy_ratio": proxy_r,
                "csl_max_dev": csl_d,
            })
    shares_df = pd.DataFrame(shares_rows)
    csv_shares = os.path.join(output_dir, f"{prefix}shares_e1.csv")
    shares_df.to_csv(csv_shares, index=False)
    paths["csv_shares"] = csv_shares

    # ── roas_index.csv ────────────────────────────────────────────────────────
    roas_rows = []
    for _, row in shares_df.iterrows():
        sv = row["spend_share"]
        roas_rows.append({
            "dim": row["dim"],
            "item": row["item"],
            "roas_index": row["contrib_share"] / sv if sv > 0 else float("nan"),
        })
    csv_roas = os.path.join(output_dir, f"{prefix}roas_index.csv")
    pd.DataFrame(roas_rows).to_csv(csv_roas, index=False)
    paths["csv_roas"] = csv_roas

    # ── contributions.html ────────────────────────────────────────────────────
    html_c = os.path.join(output_dir, f"{prefix}contributions.html")
    plot_contributions(result).write_html(html_c)
    paths["html_contributions"] = html_c

    # ── roas_index.html ───────────────────────────────────────────────────────
    html_r = os.path.join(output_dir, f"{prefix}roas_index.html")
    plot_roas_index(result).write_html(html_r)
    paths["html_roas"] = html_r

    print(f"Report saved → {output_dir}")
    for k, v in paths.items():
        print(f"  {k}: {os.path.basename(v)}")

    return paths
