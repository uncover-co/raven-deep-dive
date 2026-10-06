"""Gera o relatório HTML (Fractal Z) a partir de pastas já exportadas por `generate_report`.

    python src/build_html_report.py outputs/Cliente_VeiculoA outputs/Cliente_VeiculoB \\
        -o outputs/deepdive.html --client "Cliente"

Não refaz o fit: só lê os CSVs das pastas.
"""
import argparse

from report_gen import generate_html_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="pastas outputs/{client}_{vehicle}/")
    ap.add_argument("-o", "--out", help="HTML de saída (padrão: deepdive.html ao lado da primeira pasta)")
    ap.add_argument("--client", default="", help="overline do sidebar")
    ap.add_argument("--plotly", choices=["cdn", "inline"], default="cdn",
                    help="inline embute o plotly.js (~3 MB) e abre offline")
    a = ap.parse_args()
    print(generate_html_report(a.run_dirs, a.out, a.client, a.plotly))


if __name__ == "__main__":
    main()
