"""Minimal AR vs DG decoding schematic — just sequential vs block nature."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

AR = "#2a78d6"
DG = "#eb6834"
INK = "#0b0b0b"
MUTED = "#6b6b6b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Fira Sans", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

TW, TH = 0.7, 0.55

def tok(ax, x, y, lab, color, alpha=0.9):
    ax.add_patch(FancyBboxPatch((x, y), TW, TH, boxstyle="round,pad=0.04",
                 facecolor=color, edgecolor="white", lw=0.7, alpha=alpha))
    ax.text(x + TW/2, y + TH/2, lab, ha="center", va="center",
            fontsize=8, color="white", fontweight="medium")

def arr(ax, x1, y1, x2, y2, color, lw=1.0, ls="-", alpha=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                ls=ls, alpha=alpha, shrinkA=1, shrinkB=1))

def make():
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(6.3, 2.4),
                                    gridspec_kw={"hspace": 0.5})

    for ax in (axA, axB):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")

    # --- A: AR ---
    axA.text(0.1, 2.15, "A  AR: one token at a time", fontsize=8.5,
             color=INK, fontweight="medium", va="top")

    sx = 1.5
    labs = ["$t_1$", "$t_2$", "$t_3$", "$t_4$", "$t_5$"]
    for i, l in enumerate(labs):
        x = sx + i * 1.3
        tok(axA, x, 1.2, l, AR)
        if i < len(labs) - 1:
            arr(axA, x + TW + 0.02, 1.48, x + 1.3 - 0.02, 1.48, AR)

    axA.text(5, 0.6, "cost: $a + b \\cdot L$  (linear)", ha="center",
             fontsize=7.5, color=AR, style="italic")

    # --- B: DG ---
    axB.text(0.1, 2.15, "B  DG: one canvas at a time", fontsize=8.5,
             color=INK, fontweight="medium", va="top")

    # Canvas 1
    c1 = ["$t_1$", "$t_2$", "$t_3$", "$t_4$"]
    cx1 = 1.5
    cw1 = len(c1) * 1.3 + 0.1
    axB.add_patch(FancyBboxPatch((cx1 - 0.1, 0.95), cw1, 1.05,
                 boxstyle="round,pad=0.08", facecolor="#fdeee7",
                 edgecolor=DG, lw=0.8, alpha=0.5))
    for i, l in enumerate(c1):
        tok(axB, cx1 + i * 1.3, 1.2, l, DG)
    axB.text(cx1 + cw1/2 - 0.05, 1.75, "canvas 1", ha="center",
             fontsize=7, color=DG, fontweight="medium")
    axB.text(cx1 + cw1 + 0.05, 1.48, "...", fontsize=7, color=DG)

    # Canvas 2
    cx2 = cx1 + cw1 + 0.6
    c2 = ["$t_5$", "$t_6$"]
    cw2 = len(c2) * 1.3 + 0.1
    axB.add_patch(FancyBboxPatch((cx2 - 0.1, 0.95), cw2, 1.05,
                 boxstyle="round,pad=0.08", facecolor="#fdeee7",
                 edgecolor=DG, lw=0.8, alpha=0.3))
    for i, l in enumerate(c2):
        tok(axB, cx2 + i * 1.3, 1.2, l, DG, alpha=0.5)
    axB.text(cx2 + cw2/2 - 0.05, 1.75, "canvas 2", ha="center",
             fontsize=7, color=DG, fontweight="medium")

    # Step arrow
    arr(axB, cx1 + cw1 - 0.05, 1.48, cx2 - 0.05, 1.48, DG, lw=1.2)

    axB.text(5, 0.6, "cost: flat per canvas, steps at boundary", ha="center",
             fontsize=7.5, color=DG, style="italic")

    outdir = "paper/figures"
    for ext in ["pdf", "png"]:
        fig.savefig(f"{outdir}/fig0_decoding_schematic.{ext}",
                    dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote fig0_decoding_schematic")

if __name__ == "__main__":
    make()
