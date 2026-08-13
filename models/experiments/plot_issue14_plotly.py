#!/usr/bin/env python3
"""
Issue #14: interactive plotly visualizations.

Same data as plot_issue14.py (matplotlib/PNG) but interactive HTML:
    results/issue14_rate_lag.html      — P1 lag vs offered rate, both models
    results/issue14_length_scaling.html — P2 latency vs output length, both models

Runs in .venv-dg (plotly installed there — experiment venv only).

Usage:
    .venv-dg/bin/python models/experiments/plot_issue14_plotly.py
"""

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _REPO_ROOT / "results"

DG = dict(name="DiffusionGemma-26B (dLLM)", color="#E8734A")
AR = dict(name="AR-3B (Qwen2.5)", color="#4A90D9")


def load_rate_lag(path):
    d = json.load(open(path))
    out = {"rate": [], "p50": [], "p99": [], "sent": [], "done": [],
           "aband": []}
    for r in sorted(d["rates"], key=lambda r: r["offered_rate_ev_s"]):
        out["rate"].append(r["offered_rate_ev_s"])
        out["p50"].append(r["lag_ms"].get("p50_ms"))
        out["p99"].append(r["lag_ms"].get("p99_ms"))
        out["sent"].append(r["sent"])
        out["done"].append(r["completed"])
        out["aband"].append(r["abandoned_at_drain"])
    return out


def load_p2():
    rows = {}
    for L in (32, 64, 128, 256, 512):
        for tag, pat in (("dg", f"p2_dg_L{L}.json"), ("ar", f"p2_ar3b_L{L}.json")):
            d = json.load(open(RESULTS / pat))
            m = list(d["models"].values())[0]["streaming_table3"]
            rows[(tag, L)] = (m["latency"]["p50_ms"], m["latency"]["p99_ms"],
                              m["mean_output_tokens"])
    return rows


def rate_lag_figure():
    dg = load_rate_lag(RESULTS / "rate_lag_dg_seq1.json")
    ar = load_rate_lag(RESULTS / "rate_lag_ar3b.json")
    fig = make_subplots(rows=1, cols=1)
    for data, model in ((dg, DG), (ar, AR)):
        hover = ("λ=%{x:.2f} ev/s<br>lag p50=%{y:.0f} ms"
                 "<br>done=%{customdata[0]}/%{customdata[1]}"
                 " abandoned=%{customdata[2]}")
        customdata = list(zip(data["done"], data["sent"], data["aband"]))
        fig.add_trace(go.Scatter(
            x=data["rate"], y=data["p50"], mode="lines+markers",
            name=f"{model['name']} p50", line=dict(color=model["color"]),
            customdata=customdata, hovertemplate=hover))
        fig.add_trace(go.Scatter(
            x=data["rate"], y=data["p99"], mode="lines+markers",
            name=f"{model['name']} p99", line=dict(color=model["color"],
                                                   dash="dash"),
            opacity=0.6, customdata=customdata, hovertemplate=hover))
    fig.add_vline(x=1.81, line_dash="dot", line_color="gray",
                  annotation_text="p50 crossover ≈ 1.8 ev/s")
    fig.add_annotation(x=2.2, y=25000, showarrow=True, arrowhead=2,
                       text="DG knee ≈ 2.0–2.5 ev/s", ax=-80, ay=-40)
    fig.add_annotation(x=70, y=5000, showarrow=True, arrowhead=2,
                       text="AR-3B knee ≈ 64–96 ev/s", ax=-60, ay=-60)
    fig.update_xaxes(type="log", title_text="offered stream rate λ (events/s, Poisson)")
    fig.update_yaxes(type="log", title_text="lag: arrival → finish (ms, log)")
    fig.update_layout(
        title=("Issue #14 P1 — lag vs offered rate"
               "<br><sup>A100 80GB · vLLM 0.25.1 · BGL · DG served "
               "--max-num-seqs 1 (engine crash fix), AR default scheduler</sup>"),
        template="plotly_white", hovermode="x unified",
        legend=dict(x=0.02, y=0.98))
    return fig


def length_scaling_figure():
    rows = load_p2()
    Ls = [32, 64, 128, 256, 512]
    fig = go.Figure()
    for tag, model in (("dg", DG), ("ar", AR)):
        p50 = [rows[(tag, L)][0] for L in Ls]
        p99 = [rows[(tag, L)][1] for L in Ls]
        fig.add_trace(go.Scatter(
            x=Ls, y=p50, mode="lines+markers", name=f"{model['name']} p50",
            line=dict(color=model["color"]),
            hovertemplate="L=%{x} tok<br>p50=%{y:.0f} ms"))
        fig.add_trace(go.Scatter(
            x=Ls, y=p99, mode="lines+markers", name=f"{model['name']} p99",
            line=dict(color=model["color"], dash="dash"), opacity=0.6,
            hovertemplate="L=%{x} tok<br>p99=%{y:.0f} ms"))
    fig.add_vline(x=65, line_dash="dot", line_color="gray",
                  annotation_text="crossover ≈ 65 tokens")
    fig.add_vrect(x0=256, x1=512, fillcolor="gray", opacity=0.08,
                  annotation_text="DG pays +1 canvas block",
                  annotation_position="top left")
    fig.add_annotation(x=380, y=2500, showarrow=True, arrowhead=2,
                       text="AR: 6.28 ms/token (R²≈1.0)", ax=60, ay=-50)
    fig.add_annotation(x=160, y=418, showarrow=True, arrowhead=2,
                       text="DG: flat 418 ms within the 256-token canvas",
                       ax=-30, ay=-55)
    fig.update_xaxes(title_text="forced output length L (tokens)")
    fig.update_yaxes(title_text="m=1 latency (ms)")
    fig.update_layout(
        title=("Issue #14 P2 — latency vs output length"
               "<br><sup>A100 80GB · vLLM 0.25.1 · BGL · n=300 per cell, "
               "temperature 0</sup>"),
        template="plotly_white", hovermode="x unified",
        legend=dict(x=0.02, y=0.98))
    return fig


if __name__ == "__main__":
    f1 = rate_lag_figure()
    f1.write_html(RESULTS / "issue14_rate_lag.html", include_plotlyjs="cdn")
    print(f"wrote {RESULTS/'issue14_rate_lag.html'}")
    f2 = length_scaling_figure()
    f2.write_html(RESULTS / "issue14_length_scaling.html", include_plotlyjs="cdn")
    print(f"wrote {RESULTS/'issue14_length_scaling.html'}")
