#!/usr/bin/env python3
"""Issue #15: VIP (latency-critical) lag vs background offered rate.

Reads results/vip_dg.json + results/vip_ar3b.json; writes
results/vip_lag.png + results/vip_lag.html. Runs in .venv-dg.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _REPO_ROOT / "results"

MODELS = {
    "vip_dg.json": dict(label="DiffusionGemma (dLLM)", color="#E8734A"),
    "vip_ar3b.json": dict(label="AR-3B (Qwen2.5)", color="#4A90D9"),
}
SLA_MS = 1000.0


def load(fname):
    rows = {}
    for r in json.load(open(RESULTS / fname))["rates"]:
        v = r.get("vip_lag_ms")
        if v and r.get("vip_n"):
            rows[r["offered_rate_ev_s"]] = (
                v.get("p50_ms"), v.get("p99_ms"), v.get("p99_9_ms"),
                r.get("vip_n"))
    return rows


def series(fname):
    d = load(fname)
    rates = sorted(d)
    return rates, [d[x] for x in rates]


def plotly_fig():
    fig = go.Figure()
    for fname, m in MODELS.items():
        rates, rows = series(fname)
        p50 = [r[0] for r in rows]
        p999 = [r[2] for r in rows]
        fig.add_trace(go.Scatter(
            x=rates, y=p50, mode="lines+markers", name=f"{m['label']} VIP p50",
            line=dict(color=m["color"])))
        fig.add_trace(go.Scatter(
            x=rates, y=p999, mode="lines+markers",
            name=f"{m['label']} VIP p99.9", line=dict(color=m["color"],
                                                      dash="dash"),
            opacity=0.6))
    fig.add_hline(y=SLA_MS, line_dash="dot", line_color="gray",
                  annotation_text="SLA 1 s")
    fig.update_xaxes(type="log", title="background offered rate (events/s)")
    fig.update_yaxes(type="log", title="VIP answer lag (ms, log)")
    fig.update_layout(
        template="plotly_white",
        title=("Issue #15 — latency-critical (VIP) request lag vs background "
               "load<br><sup>A100 · vLLM 0.25.1 + priority scheduling · BGL · "
               "DG served --max-num-seqs 1 (its honest mode: mixed-phase "
               "concurrency crashes it)</sup>"),
        legend=dict(x=0.02, y=0.98))
    return fig


def mpl_fig():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for fname, m in MODELS.items():
        rates, rows = series(fname)
        ax.plot(rates, [r[0] for r in rows], "o-", color=m["color"],
                label=f"{m['label']} VIP p50")
        ax.plot(rates, [r[2] for r in rows], "s--", color=m["color"],
                alpha=0.55, label=f"{m['label']} VIP p99.9")
    ax.axhline(SLA_MS, color="gray", ls=":", lw=1)
    ax.annotate("1 s SLA", xy=(0.55, 1050), fontsize=8, color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("background offered rate (events/s)")
    ax.set_ylabel("VIP answer lag (ms, log)")
    ax.set_title("Latency-critical request lag vs background load\n"
                 "A100 · vLLM 0.25.1 · priority scheduling · BGL")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    plotly_fig().write_html(RESULTS / "vip_lag.html", include_plotlyjs="cdn")
    mpl_fig().savefig(RESULTS / "vip_lag.png", dpi=150)
    for fname, m in MODELS.items():
        d = load(fname)
        for lam in sorted(d):
            p50, p99, p999, n = d[lam]
            print("%-24s λ=%5.1f  VIP n=%d p50=%6.0fms p99=%6.0fms p99.9=%6.0fms"
                  % (m["label"], lam, n, p50, p99, p999))
