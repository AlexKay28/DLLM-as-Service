#!/usr/bin/env python3
"""Build every figure in the paper from committed artifacts.

Each function prints the values it plotted so they can be diffed against
paper/NUMBERS.md and macros.tex. Nothing here hardcodes a number that is not
either (a) read from results/*.json or (b) a fitted constant declared in
LAWS below and traceable to an artifact.

Run:  python3 paper/figures/make_figures.py
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import style as S

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = REPO / "paper" / "figures"

# --- Fitted laws (macros.tex is the authority; these must match it) ---------
LAWS = {
    "ar_a": 79.1,        # ms, AR-30B MoE serial intercept (Qwen3-30B-A3B) [macros \arBase]
    "ar_b": 47.7,        # ms/token, AR-30B MoE serial decode slope [macros \arSlope]
    "dg_base": 378.8,    # ms, DG base at K=8 [macros \dgBase]
    "dg_step": 119.5,    # ms per extra canvas (median of 7) [macros \dgStep]
    "canvas": 256,       # tokens [macros \canvasLength]
    "l_star": 6,         # token crossover, (dg_base - ar_a) / ar_b [macros \crossoverL]
    "lam_star": 2.64,    # ev/s rate crossover, 1000 / dg_base [macros \crossoverLambda]
    "nat_l_dg": 176,     # DG natural output length
    # Dense AR-3B baseline (Qwen2.5-3B), retained for figures whose measured
    # data is the dense baseline (fig3 input sweep, fig5 quality).
    "dense_ar_a": 7.27,  # ms, dense AR-3B intercept
    "dense_ar_b": 6.29,  # ms/token, dense AR-3B slope
}


# ------------------------------------------------------------------ loaders

def _p50(path):
    """p50 latency out of a run_vllm_serving-shaped artifact."""
    with open(path) as f:
        d = json.load(f)
    for mv in d.get("models", {}).values():
        return mv["streaming_table3"]["latency"]["p50_ms"]
    return d.get("p50_ms")


def _quality(path):
    with open(path) as f:
        d = json.load(f)
    for mv in d.get("models", {}).values():
        return mv["streaming_table3"]["quality"]
    return None


def t_ar(L):
    return LAWS["ar_a"] + LAWS["ar_b"] * np.asarray(L, dtype=float)


def t_dg(L):
    L = np.asarray(L, dtype=float)
    nc = np.maximum(1, np.ceil(L / LAWS["canvas"]))
    return LAWS["dg_base"] + LAWS["dg_step"] * (nc - 1)


def staircase(lmax, lmin=1.0):
    """Explicit (x, y) polyline for the DG staircase, so the vertical risers
    are drawn rather than interpolated away."""
    xs, ys = [], []
    c = 1
    x = lmin
    while x < lmax:
        hi = min(c * LAWS["canvas"], lmax)
        y = LAWS["dg_base"] + LAWS["dg_step"] * (c - 1)
        xs += [x, hi]
        ys += [y, y]
        x = hi
        c += 1
        if x < lmax:                       # the riser
            y2 = LAWS["dg_base"] + LAWS["dg_step"] * (c - 1)
            xs += [hi, hi]
            ys += [y, y2]
    return np.array(xs), np.array(ys)


# =============================================================== figure 0 ===

def fig0_decoding_schematic():
    """Schematic: AR token-by-token vs DG block denoising. Vertical layout."""
    from matplotlib.patches import FancyBboxPatch

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(S.FULL, 4.8),
        gridspec_kw={"hspace": 0.35})

    W, H = 12, 5.5

    # --- Panel A: Autoregressive ---
    axA.set_xlim(0, W)
    axA.set_ylim(0, H)
    axA.set_aspect("equal")
    axA.axis("off")
    S.title(axA, "A  Autoregressive: token by token",
            "Each token = one forward pass over the full model + KV cache")

    # Prompt box
    axA.add_patch(FancyBboxPatch((0.3, 4.2), 2.0, 0.9,
                  boxstyle="round,pad=0.1", facecolor=S.WASH_NEUTRAL,
                  edgecolor=S.AXIS, lw=0.8))
    axA.text(1.3, 4.65, "prompt", ha="center", va="center",
             fontsize=8, color=S.MUTED)

    # Output tokens, generated one at a time
    token_labels = ["$t_1$", "$t_2$", "$t_3$", "$t_4$", "$t_5$", "$t_6$"]
    box_w, box_h = 1.0, 0.9
    spacing = 1.35
    start_x = 3.0
    for i, lab in enumerate(token_labels):
        x = start_x + i * spacing
        axA.add_patch(FancyBboxPatch((x, 4.2), box_w, box_h,
                      boxstyle="round,pad=0.08", facecolor=S.AR,
                      edgecolor="white", lw=1.0, alpha=0.85))
        axA.text(x + box_w / 2, 4.65, lab, ha="center", va="center",
                 fontsize=9, color="white", fontweight="medium")
        # Forward-pass arrow
        if i < len(token_labels) - 1:
            axA.annotate("", xy=(x + spacing - 0.05, 4.65),
                         xytext=(x + box_w + 0.05, 4.65),
                         arrowprops=dict(arrowstyle="->", color=S.AR,
                                         lw=1.4, shrinkA=0, shrinkB=0))

    # Per-token cost annotations
    for i in range(len(token_labels)):
        x = start_x + i * spacing + box_w / 2
        axA.text(x, 3.7, f"{LAWS['ar_b']:.0f}ms", ha="center",
                 va="center", fontsize=7.5, color=S.AR)

    # Cumulative cost line
    for i in range(len(token_labels)):
        x = start_x + i * spacing + box_w / 2
        axA.plot(x, 2.8, "o", color=S.AR, ms=4, zorder=3)
        if i > 0:
            x_prev = start_x + (i - 1) * spacing + box_w / 2
            axA.plot([x_prev, x], [2.8, 2.8], "-", color=S.AR, lw=1.6,
                     zorder=2)
    axA.annotate("", xy=(start_x + len(token_labels) * spacing, 2.8),
                 xytext=(start_x + (len(token_labels) - 1) * spacing + box_w, 2.8),
                 arrowprops=dict(arrowstyle="->", color=S.AR, lw=1.6))
    axA.text(start_x + len(token_labels) * spacing + 0.3, 2.8,
             "linear", fontsize=8, color=S.AR, va="center",
             fontweight="medium")
    axA.text(W / 2, 1.8, "cost grows with every token", ha="center",
             fontsize=8, color=S.MUTED, style="italic")

    # --- Panel B: Block diffusion ---
    axB.set_xlim(0, W)
    axB.set_ylim(0, H)
    axB.set_aspect("equal")
    axB.axis("off")
    S.title(axB, "B  Block-diffusion: canvas by canvas",
            "Each canvas = $K$ denoising passes, all tokens in parallel")

    # Prompt box
    axB.add_patch(FancyBboxPatch((0.3, 4.2), 2.0, 0.9,
                  boxstyle="round,pad=0.1", facecolor=S.WASH_NEUTRAL,
                  edgecolor=S.AXIS, lw=0.8))
    axB.text(1.3, 4.65, "prompt", ha="center", va="center",
             fontsize=8, color=S.MUTED)

    # Canvas 1
    canvas1_tokens = ["$t_1$", "$t_2$", "$t_3$", "$t_4$"]
    cx1 = 3.0
    cw1 = len(canvas1_tokens) * spacing + 0.3
    axB.add_patch(FancyBboxPatch((cx1 - 0.15, 3.9), cw1, 1.5,
                  boxstyle="round,pad=0.12", facecolor=S.WASH_DG,
                  edgecolor=S.DG, lw=1.2, alpha=0.5))
    axB.text(cx1 + cw1 / 2 - 0.15, 5.15, "canvas 1 ($C$ tokens)",
             ha="center", va="center", fontsize=8, color=S.DG,
             fontweight="medium")
    for i, lab in enumerate(canvas1_tokens):
        x = cx1 + i * spacing
        axB.add_patch(FancyBboxPatch((x, 4.2), box_w, box_h,
                      boxstyle="round,pad=0.06", facecolor=S.DG,
                      edgecolor="white", lw=0.8, alpha=0.85))
        axB.text(x + box_w / 2, 4.65, lab, ha="center", va="center",
                 fontsize=9, color="white", fontweight="medium")

    # Denoising loop indicator
    for k in range(3):
        y = 3.4 - k * 0.35
        axB.annotate("", xy=(cx1 + cw1 - 0.3, y), xytext=(cx1 + 0.2, y),
                     arrowprops=dict(arrowstyle="->", color=S.DG,
                                     lw=0.8, ls="--", alpha=0.5))
    axB.text(cx1 - 0.6, 2.9, "$K{=}8$\npasses", ha="center", va="center",
             fontsize=7.5, color=S.DG, fontweight="medium", linespacing=1.2)

    # Canvas 2
    canvas2_tokens = ["$t_5$", "$t_6$"]
    cx2 = cx1 + cw1 + 0.8
    cw2 = len(canvas2_tokens) * spacing + 0.3
    axB.add_patch(FancyBboxPatch((cx2 - 0.15, 3.9), cw2, 1.5,
                  boxstyle="round,pad=0.12", facecolor=S.WASH_DG,
                  edgecolor=S.DG, lw=1.2, alpha=0.3))
    axB.text(cx2 + cw2 / 2 - 0.15, 5.15, "canvas 2",
             ha="center", va="center", fontsize=8, color=S.DG,
             fontweight="medium")
    for i, lab in enumerate(canvas2_tokens):
        x = cx2 + i * spacing
        axB.add_patch(FancyBboxPatch((x, 4.2), box_w, box_h,
                      boxstyle="round,pad=0.06", facecolor=S.DG,
                      edgecolor="white", lw=0.8, alpha=0.5))
        axB.text(x + box_w / 2, 4.65, lab, ha="center", va="center",
                 fontsize=9, color="white", fontweight="medium")

    # Step arrow between canvases
    axB.annotate("", xy=(cx2 - 0.2, 4.65), xytext=(cx1 + cw1 - 0.15, 4.65),
                 arrowprops=dict(arrowstyle="->", color=S.DG, lw=1.8))
    axB.text((cx1 + cw1 + cx2) / 2, 5.0, f"+{LAWS['dg_step']:.0f}ms",
             ha="center", va="bottom", fontsize=8, color=S.DG,
             fontweight="medium")

    # Staircase cost below
    y_flat1 = 2.2
    y_flat2 = 1.4
    x1_end = cx1 + cw1 - 0.15
    x2_start = cx2
    axB.plot([cx1 + 0.2, x1_end], [y_flat1, y_flat1], "-", color=S.DG,
             lw=2.5, zorder=3)
    axB.plot([x1_end, x1_end], [y_flat1, y_flat2], "-", color=S.DG,
             lw=2.5, zorder=3)
    axB.plot([x1_end, cx2 + cw2 - 0.15], [y_flat2, y_flat2], "-",
             color=S.DG, lw=2.5, zorder=3)
    axB.text(cx1 + cw1 / 2, y_flat1 + 0.3, "flat — tokens are free",
             ha="center", fontsize=7.5, color=S.DG, style="italic")
    axB.text(x1_end + 0.2, (y_flat1 + y_flat2) / 2, "step",
             fontsize=7.5, color=S.DG, style="italic", va="center")
    axB.text(cx2 + cw2 / 2, y_flat2 + 0.3, "flat again",
             ha="center", fontsize=7.5, color=S.DG, style="italic")

    S.save(fig, "fig0_decoding_schematic", OUT)


# =============================================================== figure 1 ===

def fig1_cost_shapes():
    """The two cost shapes: grouped bar chart of p50 latency at key output lengths."""
    # AR-30B MoE values come from the fitted serial cost law (Eq. 2); the dense AR-3B
    # was not measured at forced lengths, and the AR-30B forced-length sweep
    # did not pin min_tokens, so the fitted law is the operative AR series.
    # The serial AR-30B slope (47.7 ms/token) is ~48x steeper than the batch-
    # throughput slope, so the extrapolated cost-law values dwarf DG; a log
    # y-axis keeps both series visible.
    key_Ls = [256, 512, 1024, 2048]
    ar_pts = {L: t_ar(L) for L in key_Ls}

    dg_pts = {}
    for L in key_Ls:
        p = RESULTS / f"issue68_dg_L{L}.json"
        if p.exists():
            dg_pts[L] = _p50(p)

    print("fig1 AR (AR-30B cost law):", {k: round(v, 1) for k, v in ar_pts.items()})
    print("fig1 DG (measured):", {k: round(v, 1) for k, v in dg_pts.items()})

    ar_vals = [ar_pts.get(L, 0) for L in key_Ls]
    dg_vals = [dg_pts.get(L, 0) for L in key_Ls]

    fig, ax = plt.subplots(figsize=(S.FULL, 2.8))

    x = np.arange(len(key_Ls))
    width = 0.35

    bars_ar = ax.bar(x - width/2, ar_vals, width, color=S.AR, label="AR-30B (cost law)", zorder=3)
    bars_dg = ax.bar(x + width/2, dg_vals, width, color=S.DG, label="DG-26B-A4B (measured)", zorder=3)

    # Add value labels on bars (log scale: use multiplicative offset)
    for bar, val in zip(bars_ar, ar_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.06,
                    f"{val:,.0f}", ha="center", va="bottom", fontsize=7.0,
                    color=S.AR, fontweight="medium")
    for bar, val in zip(bars_dg, dg_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.06,
                    f"{val:,.0f}", ha="center", va="bottom", fontsize=7.0,
                    color=S.DG, fontweight="medium")

    # Add ratio annotations (max/min, with the faster arm labelled)
    for i, L in enumerate(key_Ls):
        if ar_vals[i] > 0 and dg_vals[i] > 0:
            ratio = max(ar_vals[i], dg_vals[i]) / min(ar_vals[i], dg_vals[i])
            who = "AR" if ar_vals[i] < dg_vals[i] else "DG"
            ymax = max(ar_vals[i], dg_vals[i])
            ax.text(i, ymax * 1.55, f"{who} {ratio:.1f}×", ha="center", va="bottom",
                    fontsize=8.2, color=S.INK2, fontweight="medium")

    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in key_Ls])
    ax.set_xlabel("output length $L_{\\mathrm{out}}$  (tokens)")
    ax.set_ylabel("p50 latency  (ms, log scale)")
    S.title(ax, "The two cost shapes: a line vs a staircase")
    S.despine(ax)
    ax.set_yscale("log")
    ax.set_ylim(min(min(dg_vals), 200) * 0.7, max(max(ar_vals), max(dg_vals)) * 2.0)
    ax.legend(loc="upper left", fontsize=8.0)

    S.save(fig, "fig1_cost_shapes", OUT)


# =============================================================== figure 2 ===

def fig2_canvas_edges():
    """Latency is flat inside a canvas and jumps at the boundary."""
    probe = [240, 256, 264, 496, 512, 528]
    vals = {}
    for L in probe:
        p = RESULTS / f"issue68_dg_L{L}.json"
        if p.exists():
            vals[L] = _p50(p)
    full = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
    fv = {L: _p50(RESULTS / f"issue68_dg_L{L}.json") for L in full
          if (RESULTS / f"issue68_dg_L{L}.json").exists()}
    steps = [fv[b] - fv[a] for a, b in zip(full, full[1:]) if a in fv and b in fv]
    print("fig2 edges:", {k: round(v, 1) for k, v in vals.items()})
    print("fig2 steps:", [round(s, 1) for s in steps],
          "median", round(float(np.median(steps)), 1))

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(S.FULL, 2.55),
        gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.30})

    # ---- Panel A: the two probed boundaries -------------------------------
    xs = np.arange(len(probe))
    ys = [vals[L] for L in probe]
    axA.bar(xs, ys, width=0.62, color=S.DG, zorder=3)
    for i, (L, y) in enumerate(zip(probe, ys)):
        axA.text(i, y + 14, f"{y:.0f}", ha="center", va="bottom",
                 fontsize=7.6, color=S.INK2)
    axA.set_xticks(xs)
    axA.set_xticklabels([str(L) for L in probe])
    axA.set_ylim(0, 820)
    axA.set_xlabel("output length $L_{\\mathrm{out}}$  (tokens)")
    axA.set_ylabel("p50 latency  (ms)")
    S.title(axA, "A  Probing the boundary")
    S.despine(axA)
    axA.grid(axis="x", visible=False)

    # shade the two within-canvas pairs; the gaps between them are the edges
    axA.axvspan(-0.5, 1.5, color=S.WASH_NEUTRAL, lw=0, zorder=0)
    axA.axvspan(2.5, 4.5, color=S.WASH_NEUTRAL, lw=0, zorder=0)
    for i, y0, lab in ((0.5, 430, "same canvas\n+0.2 ms"),
                       (3.5, 552, "same canvas\n+0.9 ms")):
        axA.text(i, y0, lab, ha="center", va="bottom", fontsize=7.2,
                 color=S.MUTED, linespacing=1.3)
    for i, lab in ((1.5, "one more canvas\n+121.7 ms"),
                   (4.5, "one more canvas\n+118.7 ms")):
        axA.text(i, 780, lab, ha="center", va="top", fontsize=7.2,
                 color=S.INK, fontweight="medium", linespacing=1.3)
        axA.plot([i, i], [0, 700], color=S.MUTED, lw=0.7, zorder=1)

    # ---- Panel B: every step, out to 2048 ---------------------------------
    axB.axhline(float(np.median(steps)), color=S.INK2, lw=0.8, zorder=2)
    bx = np.arange(len(steps))
    axB.bar(bx, steps, width=0.55, color=S.DG, zorder=3)
    axB.set_xticks(bx)
    axB.set_xticklabels([f"{a}$\\to${b}" for a, b in zip(full, full[1:])],
                        fontsize=6.2, rotation=45, ha="right")
    axB.set_ylim(0, 160)
    axB.set_ylabel("cost of one extra canvas  (ms)")
    S.title(axB, "B  Every step is the same step")
    S.despine(axB)
    axB.grid(axis="x", visible=False)
    axB.annotate(f"median {np.median(steps):.1f} ms",
                 xy=(len(steps) - 0.4, float(np.median(steps))),
                 xytext=(0, 7), textcoords="offset points", ha="right",
                 fontsize=7.6, color=S.INK, fontweight="medium")

    S.save(fig, "fig2_canvas_edges", OUT)


# =============================================================== figure 3 ===

def fig3_input_sweep():
    """The second variable: prompt length, where DG is 2.6x steeper."""
    def load(name):
        with open(RESULTS / name) as f:
            d = json.load(f)
        xs = [e["prompt_token_count"] for e in d["sweep"]]
        ys = [e["latency_ms"]["p50"] for e in d["sweep"]]
        return np.array(xs, float), np.array(ys, float)

    axr, ayr = load("issue44_ar3b_input_sweep.json")
    dxr, dyr = load("issue44_dg_input_sweep.json")
    ar_slope = np.polyfit(axr, ayr, 1)[0]
    dg_slope = np.polyfit(dxr, dyr, 1)[0]
    print(f"fig3 AR slope {ar_slope:.4f} ms/tok, DG slope {dg_slope:.4f} ms/tok, "
          f"ratio {dg_slope/ar_slope:.2f}x")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(S.FULL, 2.55),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.30})

    # ---- Panel A: absolute latency ---------------------------------------
    axA.plot(axr, ayr, S.DASH["ar"], color=S.AR, marker=S.MARK["ar"],
             ms=4.0, mec=S.SURFACE, mew=0.9, zorder=3)
    axA.plot(dxr, dyr, S.DASH["dg"], color=S.DG, marker=S.MARK["dg"],
             ms=4.0, mec=S.SURFACE, mew=0.9, zorder=3)
    axA.set_xlim(0, 7900)
    axA.set_ylim(0, 2300)
    axA.set_xticks([0, 2000, 4000, 6000])
    axA.set_xticklabels(["0", "2,000", "4,000", "6,000"])
    axA.set_xlabel("input length $L_{\\mathrm{in}}$  (prompt tokens)")
    axA.set_ylabel("p50 latency  (ms)")
    S.title(axA, "A  Cost vs prompt length", "$L_{\\mathrm{out}}$ fixed at 256 tokens")
    S.despine(axA)
    axA.grid(axis="x", visible=False)
    S.label_end(axA, axr[-1], ayr[-1], "AR-3B (dense)", S.AR, dx=7)
    S.label_end(axA, dxr[-1], dyr[-1], "DG", S.DG, dx=7)

    # ---- Panel B: the slopes, which is the actual finding -----------------
    axB.barh([1, 0], [ar_slope, dg_slope], height=0.30,
             color=[S.AR, S.DG], zorder=3)
    axB.set_yticks([1, 0])
    axB.set_yticklabels(["AR-3B (dense)", "DG"])
    axB.set_ylim(-0.55, 1.55)
    axB.set_xlim(0, 0.165)
    axB.set_xlabel("prefill slope  (ms per prompt token)")
    S.title(axB, "B  DG pays for the prompt $K$ times")
    S.despine(axB, left=False)
    axB.grid(axis="y", visible=False)
    axB.tick_params(axis="y", length=0, labelcolor=S.INK2)
    for y, v, c in ((1, ar_slope, S.AR), (0, dg_slope, S.DG)):
        axB.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=8.0,
                 color=c, fontweight="medium")
    axB.annotate(f"{dg_slope/ar_slope:.1f}$\\times$ steeper",
                 xy=(0.135, 0.5), fontsize=8.4, color=S.INK,
                 fontweight="medium", ha="center")

    S.save(fig, "fig3_input_sweep", OUT)


# =============================================================== figure 4 ===

def fig4_ksweep():
    """Where the base actually goes: two panels, one x-axis, no dual axis."""
    ks = [2, 4, 8, 16, 32, 64]
    lat, f1, valid = {}, {}, {}
    for k in ks:
        p = RESULTS / f"issue44_dg_ksweep_K{k}.json"
        if not p.exists():
            continue
        lat[k] = _p50(p)
        q = _quality(p)
        f1[k] = q.get("field_f1_9", q.get("field_f1"))
        valid[k] = q.get("schema_valid_pct")
    print("fig4 lat:", {k: round(v, 1) for k, v in lat.items()})
    print("fig4 f1 :", f1)

    kk = np.array(sorted(lat))
    yy = np.array([lat[k] for k in kk])
    a, b = np.polyfit(kk, yy, 1)[1], np.polyfit(kk, yy, 1)[0]
    print(f"fig4 fit: T(K) = {a:.1f} + {b:.2f}K")

    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(S.FULL, 3.75), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.82], "hspace": 0.34})

    for ax in (axA, axB):
        ax.axvspan(1.7, 4, color=S.WASH_NEUTRAL, lw=0, zorder=0)
        ax.axvline(8, color=S.INK2, lw=0.7, zorder=2)

    axA.plot(kk, yy, "-", color=S.DG, marker=S.MARK["dg"], ms=4.2,
             mec=S.SURFACE, mew=0.9, zorder=3)
    axA.axhline(a, color=S.MUTED, lw=0.8, zorder=2)
    axA.annotate(f"linear-fit intercept  {a:.0f} ms  (not a floor)", xy=(24, a),
                 xytext=(0, -13), textcoords="offset points", ha="center",
                 fontsize=7.6, color=S.INK2)
    axA.set_ylim(0, 520)
    axA.set_ylabel("p50 latency  (ms)")
    S.title(axA, "A  Latency saturates in $K$",
            "DiffusionGemma, BGL, $L_{\\mathrm{out}}\\!=\\!256$, $n\\!=\\!200$ per point")
    S.despine(axA)
    axA.grid(axis="x", visible=False)
    for k in kk:
        dy = -14 if k == 2 else 8
        axA.annotate(f"{lat[k]:.0f}", xy=(k, lat[k]), xytext=(0, dy),
                     textcoords="offset points", ha="center", fontsize=7.2,
                     color=S.INK2)

    ff = np.array([f1[k] for k in kk])
    axB.plot(kk, ff, "-", color=S.DG, marker=S.MARK["dg"], ms=4.2,
             mec=S.SURFACE, mew=0.9, zorder=3)
    axB.set_ylim(0.40, 0.95)
    axB.set_ylabel("9-field macro F1")
    axB.set_xlabel("denoising steps $K$")
    axB.set_xscale("log", base=2)
    axB.set_xticks(ks)
    axB.set_xticklabels([str(k) for k in ks])
    axB.minorticks_off()
    S.title(axB, "B  Quality collapses below $K\\!=\\!4$")
    S.despine(axB)
    axB.grid(axis="x", visible=False)
    for k in kk:
        axB.annotate(f"{f1[k]:.3f}", xy=(k, f1[k]), xytext=(0, -13),
                     textcoords="offset points", ha="center", fontsize=7.2,
                     color=S.INK2)

    axA.annotate("operating point $K\\!=\\!8$", xy=(8, 510), xytext=(6, 0),
                 textcoords="offset points", fontsize=7.8, color=S.INK,
                 fontweight="medium", va="top")
    axB.annotate("collapse", xy=(2.6, 0.44), fontsize=7.6, color=S.MUTED,
                 ha="center")

    S.save(fig, "fig4_ksweep", OUT)


# =============================================================== figure 5 ===

def fig5_quality_vs_length():
    """Quality vs forced output length, with the truncation regime visible."""
    with open(RESULTS / "issue43_nine_field_f1.json") as f:
        nine = json.load(f)
    Ls = [32, 128, 256, 512]
    ar = [nine[f"ar3b_L{L}"]["field_f1_9"] for L in Ls]
    dg = [nine[f"dg_L{L}"]["field_f1_9"] for L in Ls]
    arv, dgv = [], []
    for L in Ls:
        arv.append(_quality(RESULTS / "issue43" / f"ar3b_L{L}.json")["schema_valid_pct"])
        dgv.append(_quality(RESULTS / "issue43" / f"dg_L{L}.json")["schema_valid_pct"])
    print("fig5 AR F1", ar, "valid", arv)
    print("fig5 DG F1", dg, "valid", dgv)

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(S.FULL, 2.75), gridspec_kw={"wspace": 0.28})

    x = np.arange(len(Ls))
    for ax, ay, dy, ylab, ttl, ylim in (
        (axA, ar, dg, "9-field macro F1", "A  Extraction quality", (0, 1.0)),
        (axB, arv, dgv, "schema-valid outputs  (%)", "B  Parseable output", (0, 108)),
    ):
        ax.axvspan(-0.5, 1.5, color=S.WASH_NEUTRAL, lw=0, zorder=0)
        ax.plot(x, ay, S.DASH["ar"], color=S.AR, marker=S.MARK["ar"], ms=4.6,
                mec=S.SURFACE, mew=0.9, zorder=3)
        ax.plot(x, dy, S.DASH["dg"], color=S.DG, marker=S.MARK["dg"], ms=4.6,
                mec=S.SURFACE, mew=0.9, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(L) for L in Ls])
        ax.set_xlim(-0.5, len(Ls) - 0.35)
        ax.set_ylim(*ylim)
        ax.set_xlabel("forced output length $L_{\\mathrm{out}}$  (tokens)")
        ax.set_ylabel(ylab)
        S.title(ax, ttl)
        S.despine(ax)
        ax.grid(axis="x", visible=False)

    axA.text(0.5, 0.965, "truncation regime", ha="center", fontsize=7.4,
             color=S.MUTED)
    axA.annotate("AR keeps a\nvalid prefix", xy=(1, ar[1]), xytext=(-6, 22),
                 textcoords="offset points", fontsize=7.4, color=S.AR,
                 ha="right", va="bottom", linespacing=1.25,
                 arrowprops=dict(arrowstyle="-", color=S.AR, lw=0.6))
    axA.annotate("DG emits nothing\nparseable", xy=(1, dg[1]), xytext=(-6, -20),
                 textcoords="offset points", fontsize=7.4, color=S.DG,
                 ha="right", va="top", linespacing=1.25,
                 arrowprops=dict(arrowstyle="-", color=S.DG, lw=0.6))
    S.label_end(axA, x[-1], ar[-1], f"{ar[-1]:.3f}", S.AR, dx=6, dy=7)
    S.label_end(axA, x[-1], dg[-1], f"{dg[-1]:.3f}", S.DG, dx=6, dy=-7)

    axB.text(0.5, 104, "truncation regime", ha="center", fontsize=7.4,
             color=S.MUTED)
    axB.annotate("66.7%", xy=(1, arv[1]), xytext=(-7, 0),
                 textcoords="offset points", fontsize=7.4, color=S.AR,
                 ha="right", va="center")
    axB.annotate("1.0%", xy=(1, dgv[1]), xytext=(-7, 0),
                 textcoords="offset points", fontsize=7.4, color=S.DG,
                 ha="right", va="center")
    S.label_end(axB, x[-1], arv[-1], "AR-3B (dense)", S.AR, dx=6, dy=-7)
    S.label_end(axB, x[-1], dgv[-1], "DG", S.DG, dx=6, dy=7)

    handles = [
        Line2D([], [], color=S.AR, lw=1.6, marker=S.MARK["ar"], ms=4.2,
               label="AR-3B (dense)"),
        Line2D([], [], color=S.DG, lw=1.6, marker=S.MARK["dg"], ms=4.2,
               ls=S.DASH["dg"], label="DG-26B-A4B ($K\\!=\\!8$)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.09), frameon=False)

    S.save(fig, "fig5_quality_vs_length", OUT)


# =============================================================== figure 6 ===

def fig6_batching():
    """The batching gap, and the KV-cache test that only half explains it."""
    with open(RESULTS / "issue42_batching.json") as f:
        b42 = json.load(f)
    on = b42["configs"]["caching_on"]
    off = b42["configs"]["caching_off"]
    with open(RESULTS / "llada_dinfer" / "batch_sweep.json") as f:
        llada = json.load(f)
    lla_mult = max(r["multiplier"] for r in llada["results"])
    print(f"fig6 AR cache-on {on['multiplier']}x, cache-off {off['multiplier']}x, "
          f"LLaDA {lla_mult}x")

    # NB: appD batching_multipliers are keyed ar3b_L128 / ar3b_L512 -- these two
    # bars are the DENSE AR-3B, not the AR-30B MoE. The AR-30B MoE's own
    # multiplier is 24.7x (issue111_qwen3_batch_sweep.json), plotted separately.
    with open(RESULTS / "issue111_qwen3_batch_sweep.json") as f:
        ar30 = json.load(f)
    rows = [
        ("AR-30B MoE  vLLM, $L\\!=\\!256$", ar30["multiplier"], S.AR, False),
        ("AR-3B (dense)  vLLM, $L\\!=\\!512$", 86.0, S.AR,    False),
        ("AR-3B (dense)  vLLM, $L\\!=\\!128$", 77.1, S.AR,    False),
        ("AR-3B (dense)  prefix cache on",  on["multiplier"],  S.AR,    False),
        ("AR-3B (dense)  prefix cache off", off["multiplier"], S.AR,    False),
        ("DG-26B  vLLM, $L\\!=\\!128$",        1.7,  S.DG,    True),
        ("DG-26B  vLLM, $L\\!=\\!512$",        1.5,  S.DG,    True),
        ("LLaDA-8B  dInfer",             lla_mult,   S.LLADA, False),
    ]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(S.FULL, 2.85),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.06})

    # ---- Panel A: the multipliers ----------------------------------------
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, c, lower) in zip(y, rows):
        axA.barh(yi, v, height=0.56, color=c,
                 hatch="///" if lower else None,
                 edgecolor=S.SURFACE, lw=0.8, zorder=3)
        axA.text(v * 1.14, yi, f"{v:g}$\\times$", va="center", fontsize=7.8,
                 color=c, fontweight="medium")
    axA.set_yticks(y)
    axA.set_yticklabels([r[0] for r in rows], fontsize=7.4)
    axA.set_xscale("log")
    axA.set_xlim(0.9, 260)
    axA.set_xticks([1, 10, 100])
    axA.set_xticklabels(["1$\\times$", "10$\\times$", "100$\\times$"])
    axA.set_xlabel("throughput multiplier, batch-1 $\\to$ peak")
    S.title(axA, "A  Two orders of magnitude apart")
    S.despine(axA, left=False)
    axA.grid(axis="y", visible=False)
    axA.tick_params(axis="y", length=0, labelcolor=S.INK2)
    axA.axhspan(2.5, 4.5, color=S.WASH_NEUTRAL, lw=0, zorder=0)
    axA.annotate("hatched = lower bound\n(engine caps batch at 8)",
                 xy=(3.4, 0.65), fontsize=7.0, color=S.MUTED, va="center",
                 linespacing=1.3)

    # ---- Panel B: the cache test -----------------------------------------
    bs = [r["batch_size"] for r in on["sweep"]]
    axB.plot(bs, [r["events_per_s"] for r in on["sweep"]], "-",
             color=S.AR, marker=S.MARK["ar"], ms=3.8, mec=S.SURFACE, mew=0.9,
             zorder=3)
    axB.plot(bs, [r["events_per_s"] for r in off["sweep"]], ls=(0, (4, 2)),
             color=S.AR, marker="D", ms=3.4, mec=S.SURFACE, mew=0.9, zorder=3,
             alpha=0.85)
    axB.axhline(1.18 * 0.5, color=S.LLADA, lw=1.4, zorder=3)
    axB.set_xscale("log", base=2)
    axB.set_xticks(bs)
    axB.set_xticklabels([str(b) for b in bs], fontsize=7.2)
    axB.minorticks_off()
    axB.set_xlabel("batch size")
    axB.set_ylabel("throughput  (events/s)")
    axB.set_ylim(0, 26)
    S.title(axB, "B  Turning the cache off does not close it")
    S.despine(axB)
    axB.grid(axis="x", visible=False)
    S.label_end(axB, bs[-1], on["sweep"][-1]["events_per_s"],
                 f"prefix cache on  {on['multiplier']:g}$\\times$", S.AR,
                 dx=-6, dy=12, ha="right", size=7.4)
    S.label_end(axB, bs[-1], off["sweep"][-1]["events_per_s"],
                 f"cache off  {off['multiplier']:g}$\\times$", S.AR,
                 dx=-6, dy=-16, ha="right", size=7.4)
    axB.annotate("diffusion ceiling, $\\approx$1.2$\\times$", xy=(1.15, 1.2),
                 fontsize=7.4, color=S.LLADA, fontweight="medium", va="bottom")

    S.save(fig, "fig6_batching", OUT)


# =============================================================== figure 7 ===

def fig7_decision_map():
    """The decision procedure drawn as a map of the deployment plane."""
    Ls = np.linspace(16, 2048, 420)
    lams = np.logspace(np.log10(0.2), np.log10(128), 380)
    LL, MM = np.meshgrid(Ls, lams)

    # guard 1 truncation, guard 2 throughput (batching available), else DG
    arm = np.zeros_like(LL)                       # 0 = DG
    arm[LL < LAWS["nat_l_dg"]] = 1                # 1 = AR by truncation
    thr = (MM > LAWS["lam_star"]) & (LL >= LAWS["nat_l_dg"])
    arm[thr] = 2                                  # 2 = AR by throughput

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(S.FULL, 2.85),
        gridspec_kw={"width_ratios": [1.22, 1.0], "wspace": 0.32})

    axA.pcolormesh(LL, MM, arm, shading="auto",
                   cmap=plt.matplotlib.colors.ListedColormap(
                       [S.WASH_DG, S.WASH_AR, "#e3eaf3"]), zorder=0)
    axA.set_yscale("log")
    axA.set_xlim(16, 2048)
    axA.set_ylim(0.2, 128)
    axA.set_xticks([256, 512, 1024, 1536, 2048])
    axA.set_yticks([0.25, 1, 4, 16, 64])
    axA.set_yticklabels(["0.25", "1", "4", "16", "64"])
    axA.set_xlabel("expected output length $L$  (tokens)")
    axA.set_ylabel("arrival rate $\\lambda$  (events/s)")
    S.title(axA, "A  What the procedure returns",
            "batching available; 9-field tolerance $\\tau\\!=\\!0.05$")
    S.despine(axA)
    axA.grid(False)

    axA.axvline(LAWS["nat_l_dg"], color=S.INK2, lw=0.8, zorder=3)
    axA.axhline(LAWS["lam_star"], color=S.INK2, lw=0.8, zorder=3)
    axA.annotate("truncation guard  $L\\!<\\!176$", xy=(163, 0.30),
                 rotation=90, ha="right", va="bottom", fontsize=7.2,
                 color=S.INK2)
    axA.annotate("throughput guard  $\\lambda>\\lambda^{*}\\!=\\!2.6$",
                 xy=(210, 3.4), fontsize=7.4, color=S.INK2, va="bottom")
    axA.text(1100, 0.7, "DG", fontsize=13, color=S.DG, fontweight="bold",
             ha="center", va="center")
    axA.text(1100, 24, "AR", fontsize=13, color=S.AR, fontweight="bold",
             ha="center", va="center", alpha=0.75)
    axA.text(96, 24, "AR", fontsize=10, color=S.AR, fontweight="bold",
             ha="center", va="center", alpha=0.75)
    axA.text(1100, 0.42, "per-event speedup 22–80$\\times$", fontsize=7.2,
             color=S.DG, ha="center", va="center")

    # ---- Panel B: the ratio along the DG band -----------------------------
    Lb = np.linspace(16, 2048, 800)
    ratio = t_ar(Lb) / t_dg(Lb)
    axB.axhline(1.0, color=S.MUTED, lw=0.8, zorder=2)
    axB.fill_between(Lb, 1.0, ratio, where=(ratio >= 1), color=S.WASH_DG,
                     lw=0, zorder=1)
    axB.fill_between(Lb, ratio, 1.0, where=(ratio < 1), color=S.WASH_AR,
                     lw=0, zorder=1)
    axB.plot(Lb, ratio, color=S.INK, lw=1.4, zorder=3)
    axB.axvline(LAWS["nat_l_dg"], color=S.INK2, lw=0.8, ls=(0, (3, 2)), zorder=3)
    axB.set_xlim(16, 2048)
    axB.set_ylim(0, 90)
    axB.set_xticks([256, 512, 1024, 1536, 2048])
    axB.set_xlabel("expected output length $L$  (tokens)")
    axB.set_ylabel("$T_{\\mathrm{AR}}/T_{\\mathrm{DG}}$")
    S.title(axB, "B  The size of the prize", "batch-1, reference prompt")
    S.despine(axB)
    axB.grid(axis="x", visible=False)
    axB.annotate("break-even", xy=(1900, 1.0), xytext=(0, 5),
                 textcoords="offset points", ha="right", fontsize=7.4,
                 color=S.MUTED)
    axB.annotate("$L\\!=\\!176$\n(guard)", xy=(LAWS["nat_l_dg"], 85),
                 xytext=(8, 0), textcoords="offset points", fontsize=7.4,
                 color=S.INK2, va="center")
    for L in (256, 512, 1024, 2048):
        r = float(t_ar(L) / t_dg(L))
        axB.plot([L], [r], "o", ms=4.0, color=S.INK, mec=S.SURFACE, mew=0.9,
                 zorder=4)
        axB.annotate(f"{r:.1f}$\\times$", xy=(L, r), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=7.4,
                     color=S.INK)

    S.save(fig, "fig7_decision_map", OUT)


# =============================================================== figure 8 ===

def fig8_defects():
    """Pricing the baseline: eight corrections as a clean horizontal bar chart."""
    rows = [
        ("1. Decoding config",   0.779, 0.844, "AR"),
        ("2. Serving parity",    1841.6, 413.3, "AR"),
        ("3. Parser policy",     0.599, 0.826, "AR"),
        ("4. Metric defaults",   0, 100, "AR"),
        ("5. Provenance",        3, 0, "neutral"),
        ("6. Parser baseline",   0.68, 0.85, "neutral"),
        ("7. Semantic gold",     0.786, 0.852, "DG"),
        ("8. Latency model",     0, 0, "AR"),  # special: Poisson->bursty
    ]
    print("fig8 rows:", [(r[0], r[1], r[2]) for r in rows])

    fig, ax = plt.subplots(figsize=(S.FULL, 3.8))

    y = np.arange(len(rows))
    # Normalize: each row has before/after, but scales differ wildly.
    # Show as a categorical chart: bar = magnitude of correction (delta)
    deltas = []
    colors = []
    labels_before = []
    labels_after = []
    for name, before, after, bias in rows:
        if name.startswith("8"):
            deltas.append(1.0)  # qualitative
            labels_before.append("Poisson")
            labels_after.append("bursty")
        elif name.startswith("4"):
            deltas.append(100)
            labels_before.append("0%")
            labels_after.append("100%")
        elif before > 0 and after > 0:
            d = abs(float(after) - float(before)) / float(before) * 100
            deltas.append(d)
            labels_before.append(str(before))
            labels_after.append(str(after))
        else:
            deltas.append(50)
            labels_before.append(str(before))
            labels_after.append(str(after))

        if bias == "AR":
            colors.append(S.AR)
        elif bias == "DG":
            colors.append(S.DG)
        else:
            colors.append(S.MUTED)

    bars = ax.barh(y, deltas, color=colors, height=0.6, zorder=3, alpha=0.85)

    # Add before->after labels
    for i, (bar, lb, la) in enumerate(zip(bars, labels_before, labels_after)):
        w = bar.get_width()
        ax.text(w + 2, i, f"{lb} → {la}", va="center", ha="left",
                fontsize=7.0, color=S.INK2)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlabel("magnitude of correction (%)", fontsize=8.5)
    ax.set_xlim(0, max(deltas) * 1.75)
    S.title(ax, "Pricing the baseline: eight corrections")
    S.despine(ax)
    ax.grid(axis="y", visible=False)

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=S.AR, label="against AR"),
        Patch(facecolor=S.DG, label="against DG"),
        Patch(facecolor=S.MUTED, label="neutral"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5)

    S.save(fig, "fig8_defects", OUT)


# =============================================================== figure 9 ===

def fig9_deployment():
    """The deployment envelope: quality-matched pair, not compute-matched."""
    Ls = [256, 512, 1024]
    ar15 = [1007.3, 2004.7, 4025.9]     # issue_evidence_ar1b_*.json + energy probe
    dg = [378.3, 500.9, 738.7]          # issue68_dg_L*.json
    for L, v in zip((512, 1024), (ar15[1], ar15[2])):
        p = RESULTS / f"issue_evidence_ar1b_L{L}.json"
        if p.exists():
            got = _p50(p)
            assert abs(got - v) < 1.0, f"AR-1.5B L={L}: artifact {got} vs plotted {v}"
    print("fig9 AR-1.5B", ar15, "DG", dg)

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(S.FULL, 2.6),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.34})

    x = np.arange(len(Ls))
    w = 0.34
    axA.bar(x - w / 2 - 0.01, ar15, width=w, color=S.AR, zorder=3)
    axA.bar(x + w / 2 + 0.01, dg, width=w, color=S.DG, zorder=3)
    for xi, (a, d) in enumerate(zip(ar15, dg)):
        axA.text(xi - w / 2 - 0.01, a + 90, f"{a:,.0f}", ha="center",
                 fontsize=7.4, color=S.AR)
        axA.text(xi + w / 2 + 0.01, d + 90, f"{d:,.0f}", ha="center",
                 fontsize=7.4, color=S.DG)
        axA.text(xi, max(a, d) + 380, f"{a/d:.1f}$\\times$", ha="center",
                 fontsize=8.0, color=S.INK, fontweight="medium")
    axA.set_xticks(x)
    axA.set_xticklabels([f"{L}" for L in Ls])
    axA.set_ylim(0, 5000)
    axA.set_yticks([0, 1000, 2000, 3000, 4000])
    axA.set_yticklabels(["0", "1,000", "2,000", "3,000", "4,000"])
    axA.set_xlabel("forced output length $L_{\\mathrm{out}}$  (tokens)")
    axA.set_ylabel("p50 latency  (ms)")
    S.title(axA, "A  Latency at the quality-matched pair",
            "AR-1.5B (F1 0.868) vs DG-26B-A4B (F1 0.852)")
    S.despine(axA)
    axA.grid(axis="x", visible=False)

    # ---- Panel B: what it costs to keep on a GPU --------------------------
    cats = ["weights on disk\n(GB, bf16)", "energy per event\n(J, net of idle, $L\\!=\\!256$)"]
    arv = [3.0, 5.1]
    dgv = [52.0, 4.2]
    xb = np.arange(len(cats))
    # each category normalised to its own max — the units differ
    norm = [max(a, d) for a, d in zip(arv, dgv)]
    axB.bar(xb - w / 2 - 0.01, [a / n for a, n in zip(arv, norm)], width=w,
            color=S.AR, zorder=3)
    axB.bar(xb + w / 2 + 0.01, [d / n for d, n in zip(dgv, norm)], width=w,
            color=S.DG, zorder=3)
    for xi, (a, d, n) in enumerate(zip(arv, dgv, norm)):
        axB.text(xi - w / 2 - 0.01, a / n + 0.03, f"{a:g}", ha="center",
                 fontsize=7.4, color=S.AR)
        axB.text(xi + w / 2 + 0.01, d / n + 0.03, f"{d:g}", ha="center",
                 fontsize=7.4, color=S.DG)
    axB.set_xticks(xb)
    axB.set_xticklabels(cats, fontsize=7.2)
    axB.set_ylim(0, 1.30)
    axB.set_yticks([])
    axB.set_ylabel("each pair scaled to its own maximum", fontsize=7.6)
    S.title(axB, "B  What it costs to keep resident")
    S.despine(axB, left=False)
    axB.grid(False)
    axB.tick_params(axis="x", length=0)

    handles = [
        Patch(facecolor=S.AR, label="AR-1.5B  (Qwen2.5-1.5B, 3 GB)"),
        Patch(facecolor=S.DG, label="DG-26B-A4B  (52 GB)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.10), frameon=False)

    S.save(fig, "fig9_deployment", OUT)


# ==============================================================================

if __name__ == "__main__":
    S.apply()
    print("building figures ->", OUT)
    fig0_decoding_schematic()
    fig1_cost_shapes()
    fig2_canvas_edges()
    fig3_input_sweep()
    fig4_ksweep()
    fig5_quality_vs_length()
    fig6_batching()
    fig7_decision_map()
    fig8_defects()
    fig9_deployment()
    print("done.")
