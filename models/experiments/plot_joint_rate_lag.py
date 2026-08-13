#!/usr/bin/env python3
"""
Issue #14 (joint): ONE chart — x=offered rate λ, y=p50 lag, hue=model, dash=L.

Reads results/rate_lag_{dg,ar3b}_L{64,128,256,512}.json (uniform λ grid
{1,2,4,8,16}) and writes:
    results/joint_rate_lag.png   (matplotlib, for issue embedding)
    results/joint_rate_lag.html  (plotly, interactive)

Runs in .venv-dg.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _REPO_ROOT / "results"
LS = (64, 128, 256, 512)

MODELS = {
    "dg": dict(label="DiffusionGemma-26B (dLLM)", color="#E8734A"),
    "ar3b": dict(label="AR-3B (Qwen2.5)", color="#4A90D9"),
}
DASHES = {64: "solid", 128: "dash", 256: "dot", 512: "dashdot"}
MARKERS = {64: "o", 128: "s", 256: "^", 512: "D"}


def load(tag, L):
    d = json.load(open(RESULTS / f"rate_lag_{tag}_L{L}.json"))
    rows = sorted(d["rates"], key=lambda r: r["offered_rate_ev_s"])
    return ([r["offered_rate_ev_s"] for r in rows],
            [r["lag_ms"].get("p50_ms") for r in rows],
            [(r["completed"], r["sent"], r["abandoned_at_drain"]) for r in rows])


def plotly_fig():
    fig = go.Figure()
    for tag, m in MODELS.items():
        for L in LS:
            rate, p50, meta = load(tag, L)
            fig.add_trace(go.Scatter(
                x=rate, y=p50, mode="lines+markers",
                name=f"{m['label']} · L={L}",
                line=dict(color=m["color"], dash=DASHES[L]),
                legendgroup=m["label"],
                hovertemplate=("%{fullData.name}<br>λ=%{x} ev/s"
                               "<br>p50 lag=%{y:.0f} ms"
                               "<br>done=%{customdata[0]}/%{customdata[1]}"
                               " aband=%{customdata[2]}"),
                customdata=meta))
    fig.update_xaxes(type="log", title="offered stream rate λ (events/s)")
    fig.update_yaxes(type="log", title="lag: arrival → finish p50 (ms, log)")
    fig.update_layout(
        template="plotly_white", hovermode="closest",
        title=("Issue #14 joint — lag vs stream rate, by output length L"
               "<br><sup>A100 · vLLM 0.25.1 · BGL · DG serialized "
               "(--max-num-seqs 1, engine-crash workaround); "
               "AR default scheduler</sup>"),
        legend=dict(x=0.01, y=0.99, font=dict(size=10)))
    return fig


def matplotlib_fig():
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, m in MODELS.items():
        for L in LS:
            rate, p50, _ = load(tag, L)
            ax.plot(rate, p50, marker=MARKERS[L], color=m["color"],
                    linestyle=(0, (4, 2)) if DASHES[L] != "solid" else "solid",
                    label=f"{m['label']} · L={L}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("offered stream rate λ (events/s)")
    ax.set_ylabel("lag: arrival → finish p50 (ms, log)")
    ax.set_title("Issue #14: lag vs stream rate by output length L\n"
                 "A100 · vLLM 0.25.1 · BGL · DG serialized "
                 "(engine-crash workaround)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    plotly_fig().write_html(RESULTS / "joint_rate_lag.html",
                            include_plotlyjs="cdn")
    matplotlib_fig().savefig(RESULTS / "joint_rate_lag.png", dpi=150)
    print("wrote results/joint_rate_lag.{html,png}")
