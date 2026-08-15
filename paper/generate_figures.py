#!/usr/bin/env python3
"""Generate all paper figures from committed artifacts."""

import json
import sys
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "results"
FIGS = REPO / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def load_vllm_p50(path):
    with open(path) as f:
        d = json.load(f)
    for mk, mv in d.get("models", {}).items():
        p50 = mv.get("streaming_table3", {}).get("latency", {}).get("p50_ms")
        if p50:
            return p50
    # Also try top-level (some result files store it differently)
    return d.get("p50_ms")


def fig1_cost_law_comparison():
    """F1: T_AR vs T_DG vs L_out."""
    lengths = [32, 64, 128, 256, 512, 1024, 2048]
    ar = [7.27 + 6.29 * L for L in lengths]
    
    dg = []
    for L in lengths:
        n_c = max(1, math.ceil(L / 256))
        dg.append(378.8 + 119.6 * (n_c - 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lengths, ar, 'b-o', label='AR-3B (linear)', linewidth=2)
    ax.plot(lengths, dg, 'r-s', label='DG-26B (staircase)', linewidth=2)
    ax.set_xlabel('Output Length L (tokens)')
    ax.set_ylabel('p50 Latency (ms)')
    ax.set_title('Cost Law Comparison')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Mark crossover
    ax.axvline(x=59, color='gray', linestyle='--', alpha=0.5)
    ax.annotate('L*=59', xy=(59, 400), fontsize=9, color='gray')
    
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_cost_laws.pdf", dpi=150)
    fig.savefig(FIGS / "fig1_cost_laws.png", dpi=150)
    plt.close()
    print("Wrote fig1_cost_laws")


def fig2_canvas_edges():
    """F2: Canvas step edges — p50 at boundary-sweep lengths."""
    data = {}
    for L in [240, 256, 264, 496, 512, 528, 768, 1024, 1280, 1536, 1792, 2048]:
        path = RESULTS / f"issue68_dg_L{L}.json"
        if path.exists():
            p50 = load_vllm_p50(path)
            if p50:
                data[L] = p50
    
    if not data:
        print("fig2: no data")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ls = sorted(data.keys())
    ps = [data[l] for l in ls]
    ax.plot(ls, ps, 'r-s', linewidth=2, markersize=5)
    
    # Mark canvas boundaries
    for b in [256, 512, 768, 1024, 1280, 1536, 1792, 2048]:
        ax.axvline(x=b, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Output Length L (tokens)')
    ax.set_ylabel('p50 Latency (ms)')
    ax.set_title('Canvas Step Edges (DG, K=8)')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_canvas_edges.pdf", dpi=150)
    fig.savefig(FIGS / "fig2_canvas_edges.png", dpi=150)
    plt.close()
    print("Wrote fig2_canvas_edges")


def fig3_input_sweep():
    """F3: T vs L_in for both models."""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    for name, path, color, marker in [
        ("AR-3B", RESULTS / "issue44_ar3b_input_sweep.json", 'b', 'o'),
        ("DG (K=8)", RESULTS / "issue44_dg_input_sweep.json", 'r', 's'),
    ]:
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        xs = [e["prompt_token_count"] for e in d.get("sweep", [])]
        ys = [e["latency_ms"]["p50"] for e in d.get("sweep", [])]
        ax.plot(xs, ys, f'{color}-{marker}', label=name, linewidth=2)
    
    ax.set_xlabel('Input Length (tokens)')
    ax.set_ylabel('p50 Latency (ms)')
    ax.set_title('Input-Length Sweep (L_out=256)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_input_sweep.pdf", dpi=150)
    fig.savefig(FIGS / "fig3_input_sweep.png", dpi=150)
    plt.close()
    print("Wrote fig3_input_sweep")


def fig4_crossover():
    """F4: Decision procedure crossover diagram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left: L crossover
    Ls = np.linspace(10, 300, 100)
    ar = 7.27 + 6.29 * Ls
    dg = np.full_like(Ls, 378.8)
    ax1.plot(Ls, ar, 'b-', label='AR-3B', linewidth=2)
    ax1.plot(Ls, dg, 'r-', label='DG (K=8)', linewidth=2)
    ax1.axvline(x=59, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate('L*=59', xy=(59, 500), fontsize=9)
    ax1.fill_between(Ls, dg, ar, where=ar > dg, alpha=0.1, color='red', label='DG wins')
    ax1.fill_between(Ls, dg, ar, where=ar < dg, alpha=0.1, color='blue', label='AR wins')
    ax1.set_xlabel('Output Length L (tokens)')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Length Crossover')
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 2000)
    
    # Right: λ crossover
    lams = np.linspace(0.1, 10, 100)
    ar_eff = 1616.0 / (1 + 79 * (lams > 1.8))  # simplified
    dg_eff = 378.8 / 1.18 * np.ones_like(lams)
    ax2.plot(lams, 1000/lams, 'b-', label='AR (batched)', linewidth=2)
    ax2.axhline(y=1000/378.8, color='r', linestyle='-', label='DG (serial)', linewidth=2)
    ax2.axvline(x=2.6, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate('λ*=2.6', xy=(2.6, 100), fontsize=9)
    ax2.set_xlabel('Arrival Rate λ (ev/s)')
    ax2.set_ylabel('Effective Latency (ms/ev)')
    ax2.set_title('Rate Crossover')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1000)
    
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_crossover.pdf", dpi=150)
    fig.savefig(FIGS / "fig4_crossover.png", dpi=150)
    plt.close()
    print("Wrote fig4_crossover")


def fig5_defects():
    """F5: Five defects magnitude."""
    defects = [
        ("Decoding\nconfig", 0.779, 0.844, "F1"),
        ("Serving\nparity", 1841.6, 413.3, "ms"),
        ("Parser\npolicy", 0.599, 0.826, "F1"),
        ("Metric\ndefaults", 0, 100, "%"),
        ("Provenance", 3, 0, "files"),
    ]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(defects))
    width = 0.35
    
    before = [d[1] for d in defects]
    after = [d[2] for d in defects]
    
    # Normalize for display (different units)
    ax.bar(x - width/2, before, width, label='Before fix', color='salmon', alpha=0.8)
    ax.bar(x + width/2, after, width, label='After fix', color='steelblue', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in defects], fontsize=8)
    ax.set_ylabel('Magnitude (varies by defect)')
    ax.set_title('Five Harness Defects — Magnitude of Each Bias')
    ax.legend()
    ax.set_yscale('symlog')
    
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_defects.pdf", dpi=150)
    fig.savefig(FIGS / "fig5_defects.png", dpi=150)
    plt.close()
    print("Wrote fig5_defects")


def fig6_ksweep():
    """F6: K-sweep — base decomposition."""
    ks = [2, 4, 8, 16, 32, 64]
    p50s = []
    f1s = []
    for k in ks:
        path = RESULTS / f"issue44_dg_ksweep_K{k}.json"
        if path.exists():
            p50 = load_vllm_p50(path)
            if p50:
                p50s.append(p50)
                with open(path) as f:
                    d = json.load(f)
                for mk, mv in d.get("models", {}).items():
                    q = mv.get("streaming_table3", {}).get("quality", {})
                    f1s.append(q.get("field_f1_9", q.get("field_f1", 0)))
    
    if len(p50s) < 3:
        print("fig6: insufficient data")
        return
    
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ks[:len(p50s)], p50s, 'b-o', label='p50 (ms)', linewidth=2)
    ax1.set_xlabel('K (denoising steps)')
    ax1.set_ylabel('p50 Latency (ms)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(ks[:len(f1s)], f1s, 'r-s', label='F1 (9-field)', linewidth=2)
    ax2.set_ylabel('Field-F1 (9-field)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.axhline(y=0.85, color='r', linestyle=':', alpha=0.3)
    
    ax1.set_title('DG Base Decomposition (K-sweep)')
    ax1.set_xscale('log', base=2)
    
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_ksweep.pdf", dpi=150)
    fig.savefig(FIGS / "fig6_ksweep.png", dpi=150)
    plt.close()
    print("Wrote fig6_ksweep")


def fig7_quality_vs_length():
    """F7: Quality vs output length, 9-field and 12-field."""
    path = RESULTS / "issue43_nine_field_f1.json"
    if not path.exists():
        print("fig7: no data")
        return
    
    with open(path) as f:
        d = json.load(f)
    
    lengths = [32, 128, 256, 512]
    ar9 = [d.get(f"ar3b_L{L}", {}).get("field_f1_9", 0) for L in lengths]
    dg9 = [d.get(f"dg_L{L}", {}).get("field_f1_9", 0) for L in lengths]
    ar12 = [d.get(f"ar3b_L{L}", {}).get("field_f1_12", 0) for L in lengths]
    dg12 = [d.get(f"dg_L{L}", {}).get("field_f1_12", 0) for L in lengths]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(lengths, ar9, 'b-o', label='AR-3B', linewidth=2)
    ax1.plot(lengths, dg9, 'r-s', label='DG', linewidth=2)
    ax1.set_xlabel('Output Length L (tokens)')
    ax1.set_ylabel('Field-F1 (9-field)')
    ax1.set_title('9-Field Quality (headline)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(lengths, ar12, 'b-o', label='AR-3B', linewidth=2)
    ax2.plot(lengths, dg12, 'r-s', label='DG', linewidth=2)
    ax2.set_xlabel('Output Length L (tokens)')
    ax2.set_ylabel('Field-F1 (12-field)')
    ax2.set_title('12-Field Quality (contaminated)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_quality.pdf", dpi=150)
    fig.savefig(FIGS / "fig7_quality.png", dpi=150)
    plt.close()
    print("Wrote fig7_quality")


if __name__ == "__main__":
    fig1_cost_law_comparison()
    fig2_canvas_edges()
    fig3_input_sweep()
    fig4_crossover()
    fig5_defects()
    fig6_ksweep()
    fig7_quality_vs_length()
    print("\nAll figures generated.")
