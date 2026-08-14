#!/usr/bin/env python3
"""
Issue #23: Analyze imputation results. Reads AR + DM JSON, produces
comparison table + verdict, writes results/imputation_report.md.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def load_result(path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt_cell(cell):
    if cell is None:
        return "—"
    return (f"{cell['field_f1_imputed']:.3f} | "
            f"{cell['latency'].get('p50_ms', 0):.0f}ms")


def main():
    results = {}
    for model_label, model_type in [("AR", "AR"), ("DM", "DM")]:
        for strat in ["random", "structural", "semantic"]:
            path = REPO / f"results/imputation_{model_label}_{strat}.json"
            data = load_result(path)
            if data:
                results[(model_type, strat)] = data

    if not results:
        print("No results found. Run the experiments first.")
        sys.exit(1)

    lines = ["# Issue #23: Missing-field imputation — results\n"]
    lines.append("**Date:** 2026-08-14\n")

    for strat in ["random", "structural", "semantic"]:
        ar = results.get(("AR", strat))
        dm = results.get(("DM", strat))
        if not ar and not dm:
            continue

        lines.append(f"\n## Stratification: {strat}\n")

        ar_cells = {c["k"]: c for c in ar["cells"]} if ar else {}
        dm_cells = {c["k"]: c for c in dm["cells"]} if dm else {}
        all_k = sorted(set(list(ar_cells.keys()) + list(dm_cells.keys())))

        lines.append(f"\n| K | AR F1(imp) | AR p50 | DM F1(imp) | DM p50 | "
                      "F1 Δ (DM-AR) | Speed ratio (AR/DM) |")
        lines.append("|---|---|---|---|---|---|---|")
        for k in all_k:
            ac = ar_cells.get(k)
            dc = dm_cells.get(k)
            ar_f1 = f"{ac['field_f1_imputed']:.3f}" if ac else "—"
            ar_p50 = f"{ac['latency'].get('p50_ms', 0):.0f}" if ac else "—"
            dm_f1 = f"{dc['field_f1_imputed']:.3f}" if dc else "—"
            dm_p50 = f"{dc['latency'].get('p50_ms', 0):.0f}" if dc else "—"
            if ac and dc:
                f1_delta = dc['field_f1_imputed'] - ac['field_f1_imputed']
                speed_ratio = ac['latency'].get('p50_ms', 1) / max(dc['latency'].get('p50_ms', 1), 0.1)
                delta_s = f"{f1_delta:+.3f}"
                ratio_s = f"{speed_ratio:.2f}x"
            else:
                delta_s = "—"
                ratio_s = "—"
            lines.append(f"| {k} | {ar_f1} | {ar_p50} | {dm_f1} | {dm_p50} | "
                         f"{delta_s} | {ratio_s} |")

    lines.append("\n## Verdict (P-F predictions)\n")
    lines.append("| ID | Prediction | Result |")
    lines.append("|---|---|---|")

    random_ar = results.get(("AR", "random"))
    random_dm = results.get(("DM", "random"))
    if random_ar and random_dm:
        ar_cells = {c["k"]: c for c in random_ar["cells"]}
        dm_cells = {c["k"]: c for c in random_dm["cells"]}
        all_k = sorted(set(list(ar_cells.keys()) + list(dm_cells.keys())))

        p_f1 = all(dm_cells[k]["field_f1_imputed"] >= ar_cells[k]["field_f1_imputed"]
                    for k in all_k if k in ar_cells and k in dm_cells)
        lines.append(f"| P-F1 | DM F1 ≥ AR F1 at all K | "
                      f"{'**PASS**' if p_f1 else '**REFUTED**'} |")

        if 8 in dm_cells and 1 in dm_cells:
            dm_lat_ratio = dm_cells[8]["latency"].get("p50_ms", 1) / dm_cells[1]["latency"].get("p50_ms", 1)
            p_f2 = dm_lat_ratio < 1.20
            lines.append(f"| P-F2 | DM latency flat in K (≤20% growth K1→K8) | "
                          f"{'**PASS**' if p_f2 else '**REFUTED**'} "
                          f"(ratio={dm_lat_ratio:.2f}) |")

        if 8 in ar_cells and 1 in ar_cells:
            ar_lat_ratio = ar_cells[8]["latency"].get("p50_ms", 1) / ar_cells[1]["latency"].get("p50_ms", 1)
            p_f3 = ar_lat_ratio > 1.20
            lines.append(f"| P-F3 | AR latency grows with K | "
                          f"{'**PASS**' if p_f3 else '**REFUTED**'} "
                          f"(ratio={ar_lat_ratio:.2f}) |")

        if 8 in ar_cells and 8 in dm_cells:
            ratio_8 = ar_cells[8]["latency"].get("p50_ms", 1) / max(dm_cells[8]["latency"].get("p50_ms", 1), 0.1)
            if 1 in ar_cells and 1 in dm_cells:
                ratio_1 = ar_cells[1]["latency"].get("p50_ms", 1) / max(dm_cells[1]["latency"].get("p50_ms", 1), 0.1)
                p_f4 = ratio_8 > ratio_1
                lines.append(f"| P-F4 | Cost ratio AR/DM grows with K | "
                              f"{'**PASS**' if p_f4 else '**REFUTED**'} "
                              f"(K=1: {ratio_1:.2f}x, K=8: {ratio_8:.2f}x) |")

    out_path = REPO / "results" / "imputation_report.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
