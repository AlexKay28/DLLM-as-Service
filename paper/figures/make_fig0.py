"""Generate the AR vs DG decoding schematic — compact, no collisions.

Style: paper-figure (clean, NeurIPS-like)
Layout: vertical, two panels (A: AR, B: DG)
Size: matches other figures (6.3 x 2.8 in per panel pair)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

AR_COLOR = "#2a78d6"
DG_COLOR = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#6b6b6b"
SURFACE = "#ffffff"
WASH_DG = "#fdeee7"
WASH_NEUTRAL = "#f4f3f0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Fira Sans", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

TW, TH = 0.85, 0.65   # token box dimensions
SP = 1.15              # spacing between tokens

def token(ax, x, y, label, color, alpha=0.9):
    box = FancyBboxPatch((x, y), TW, TH, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor="white", lw=0.8, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + TW/2, y + TH/2, label, ha="center", va="center",
            fontsize=8, color="white", fontweight="medium")

def arrow(ax, x1, y1, x2, y2, color, lw=1.2, ls="-", alpha=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, ls=ls,
                                alpha=alpha, shrinkA=1, shrinkB=1))

def make_schematic():
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(6.3, 3.6),
                                    gridspec_kw={"hspace": 0.55})

    W, H = 10, 4.5

    for ax in (axA, axB):
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.axis("off")

    # =================== Panel A: AR ===================
    axA.text(0, H - 0.15, "A  Autoregressive: one forward pass per token",
             fontsize=9, fontweight="medium", color=INK, va="top")

    # Prompt
    token(axA, 0.2, 3.2, "prompt", MUTED, alpha=0.25)

    # Tokens
    labels = ["$t_1$", "$t_2$", "$t_3$", "$t_4$", "$t_5$"]
    sx = 2.0
    for i, lab in enumerate(labels):
        x = sx + i * SP
        token(axA, x, 3.2, lab, AR_COLOR)
        if i < len(labels) - 1:
            arrow(axA, x + TW + 0.03, 3.55, x + SP - 0.03, 3.55, AR_COLOR)

    # "sequential" label
    axA.text(sx + 2 * SP, 2.7, "sequential: must finish $t_i$ before $t_{i+1}$",
             ha="center", fontsize=7, color=INK2, style="italic")

    # Cost line (linear) with mini-axes
    axA.annotate("", xy=(W - 0.3, 0.3), xytext=(1.5, 0.3),
                 arrowprops=dict(arrowstyle="->", color=AXIS_C, lw=0.6))
    axA.annotate("", xy=(1.5, 1.8), xytext=(1.5, 0.3),
                 arrowprops=dict(arrowstyle="->", color=AXIS_C, lw=0.6))
    axA.text(1.3, 0.3, "0", fontsize=6, color=MUTED, ha="right", va="center")
    axA.text(1.5, 1.95, "latency", fontsize=6, color=MUTED, ha="center")
    axA.text(W - 0.3, 0.12, "tokens", fontsize=6, color=MUTED, ha="center")

    xs = np.linspace(1.8, W - 0.5, 50)
    ys = 0.4 + (xs - 1.8) * 0.25
    axA.plot(xs, ys, "-", color=AR_COLOR, lw=2)
    axA.text(W - 1.8, 1.5, "linear cost", fontsize=7, color=AR_COLOR,
             fontweight="medium")

    # =================== Panel B: DG ===================
    axB.text(0, H - 0.15,
             "B  Block-diffusion: $K$ passes per canvas, tokens in parallel",
             fontsize=9, fontweight="medium", color=INK, va="top")

    # Prompt
    token(axB, 0.2, 3.2, "prompt", MUTED, alpha=0.25)

    # Canvas 1
    c1_tokens = ["$t_1$", "$t_2$", "$t_3$", "$t_4$"]
    cx1 = 2.0
    cw1 = len(c1_tokens) * SP + 0.15

    cb1 = FancyBboxPatch((cx1 - 0.1, 2.95), cw1, 1.15,
                          boxstyle="round,pad=0.1",
                          facecolor=WASH_DG, edgecolor=DG_COLOR,
                          lw=1.0, alpha=0.5)
    axB.add_patch(cb1)
    axB.text(cx1 + cw1/2 - 0.05, 3.9, "canvas 1", ha="center",
             fontsize=7.5, color=DG_COLOR, fontweight="medium")

    for i, lab in enumerate(c1_tokens):
        token(axB, cx1 + i * SP, 3.2, lab, DG_COLOR)

    # Ellipsis to show C=256 is more than 4 tokens
    axB.text(cx1 + len(c1_tokens) * SP + 0.1, 3.55, "...",
             fontsize=8, color=DG_COLOR, va="center")

    # "parallel" annotation
    axB.text(cx1 + cw1/2 - 0.05, 2.65, "all tokens at once",
             ha="center", fontsize=7, color=INK2, style="italic")

    # Canvas 2
    c2_tokens = ["$t_5$", "$t_6$"]
    cx2 = cx1 + cw1 + 0.7
    cw2 = len(c2_tokens) * SP + 0.15

    cb2 = FancyBboxPatch((cx2 - 0.1, 2.95), cw2, 1.15,
                          boxstyle="round,pad=0.1",
                          facecolor=WASH_DG, edgecolor=DG_COLOR,
                          lw=1.0, alpha=0.3)
    axB.add_patch(cb2)
    axB.text(cx2 + cw2/2 - 0.05, 3.9, "canvas 2", ha="center",
             fontsize=7.5, color=DG_COLOR, fontweight="medium")

    for i, lab in enumerate(c2_tokens):
        token(axB, cx2 + i * SP, 3.2, lab, DG_COLOR, alpha=0.5)

    # Step arrow
    arrow(axB, cx1 + cw1 - 0.05, 3.55, cx2 - 0.05, 3.55, DG_COLOR, lw=1.5)
    axB.text((cx1 + cw1 + cx2) / 2, 3.75, "+120ms",
             ha="center", fontsize=7, color=DG_COLOR, fontweight="medium")

    # Staircase with mini-axes
    axB.annotate("", xy=(W - 0.3, 0.3), xytext=(1.5, 0.3),
                 arrowprops=dict(arrowstyle="->", color=AXIS_C, lw=0.6))
    axB.annotate("", xy=(1.5, 1.8), xytext=(1.5, 0.3),
                 arrowprops=dict(arrowstyle="->", color=AXIS_C, lw=0.6))
    axB.text(1.3, 0.3, "0", fontsize=6, color=MUTED, ha="right", va="center")
    axB.text(1.5, 1.95, "latency", fontsize=6, color=MUTED, ha="center")
    axB.text(W - 0.3, 0.12, "tokens", fontsize=6, color=MUTED, ha="center")

    # Staircase steps
    mid1 = cx1 + cw1 / 2
    mid2 = cx2 + cw2 / 2
    step_x = cx1 + cw1 + 0.25

    axB.plot([1.8, step_x], [0.5, 0.5], "-", color=DG_COLOR, lw=2)
    axB.plot([step_x, step_x], [0.5, 1.1], "-", color=DG_COLOR, lw=2)
    axB.plot([step_x, W - 0.5], [1.1, 1.1], "-", color=DG_COLOR, lw=2)

    axB.text(mid1, 0.7, "flat", ha="center", fontsize=6.5, color=DG_COLOR,
             style="italic")
    axB.text(step_x + 0.1, 0.8, "step", fontsize=6.5, color=DG_COLOR,
             style="italic")
    axB.text(mid2 + 0.3, 1.3, "flat", ha="center", fontsize=6.5,
             color=DG_COLOR, style="italic")

    # Save
    outdir = "paper/figures"
    os.makedirs(outdir, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(f"{outdir}/fig0_decoding_schematic.{ext}",
                    dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote fig0_decoding_schematic")

AXIS_C = "#c3c2b7"

if __name__ == "__main__":
    make_schematic()
