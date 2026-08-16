"""AR vs DG decoding schematic — clean, no collisions, high quality."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

AR = "#2a78d6"
DG = "#eb6834"
INK = "#1a1a1a"
MUTED = "#555555"
WASH_DG = "#fdf0ea"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Fira Sans", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
})

TW, TH = 0.8, 0.6
GAP = 0.25
STEP = TW + GAP  # 1.05

def tok(ax, x, y, lab, color, alpha=0.9):
    ax.add_patch(FancyBboxPatch((x, y), TW, TH, boxstyle="round,pad=0.05",
                 facecolor=color, edgecolor="white", lw=1.0, alpha=alpha))
    ax.text(x + TW/2, y + TH/2, lab, ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")

def arr(ax, x1, y1, x2, y2, color, lw=1.2, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, ls=ls,
                                shrinkA=2, shrinkB=2))

def make():
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(6.3, 3.2),
                                    gridspec_kw={"hspace": 0.6})

    W, H = 10, 4.0

    for ax in (axA, axB):
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.axis("off")

    # y-bands with generous gaps:
    #   3.6 — title
    #   2.8 — canvas label
    #   1.8 — tokens (1.8 to 2.4)
    #   0.6 — cost annotation

    TY = 1.8  # token y (bottom of token box)

    # =================== Panel A: AR ===================
    axA.text(0.2, 3.6, "A  Autoregressive: one token at a time",
             fontsize=10, color=INK, fontweight="medium", va="top")

    sx = 1.2
    labs = ["$t_1$", "$t_2$", "$t_3$", "$t_4$", "$t_5$"]
    for i, l in enumerate(labs):
        x = sx + i * STEP
        tok(axA, x, TY, l, AR)
        if i < len(labs) - 1:
            arr(axA, x + TW + 0.02, TY + TH/2, x + STEP - 0.02, TY + TH/2, AR)

    axA.text(W/2, 0.6, "cost: $a + b \\cdot L$  (linear)",
             ha="center", fontsize=8.5, color=AR, style="italic")

    # =================== Panel B: DG ===================
    axB.text(0.2, 3.6, "B  Block-diffusion: one canvas at a time",
             fontsize=10, color=INK, fontweight="medium", va="top")

    c1 = ["$t_1$", "$t_2$", "$t_3$", "$t_4$"]
    cx1 = 1.2
    cw1 = len(c1) * STEP - GAP + 0.2
    cy = 1.5   # canvas box bottom
    ch = 1.2   # canvas box height -> top at 2.7

    axB.add_patch(FancyBboxPatch((cx1 - 0.1, cy), cw1, ch,
                 boxstyle="round,pad=0.1", facecolor=WASH_DG,
                 edgecolor=DG, lw=1.0, alpha=0.6))
    # Canvas label ABOVE the box, not inside
    axB.text(cx1 + cw1/2 - 0.1, cy + ch + 0.15, "canvas 1",
             ha="center", va="bottom", fontsize=8, color=DG, fontweight="medium")

    for i, l in enumerate(c1):
        tok(axB, cx1 + i * STEP, TY, l, DG)

    # Ellipsis — placed after canvas 1 box, at token height
    ellipsis_x = cx1 + cw1 + 0.15
    axB.text(ellipsis_x, TY + TH/2, "$\\cdots$",
             fontsize=10, color=DG, va="center")

    # Canvas 2
    c2 = ["$t_5$", "$t_6$"]
    cx2 = ellipsis_x + 0.7
    cw2 = len(c2) * STEP - GAP + 0.2

    axB.add_patch(FancyBboxPatch((cx2 - 0.1, cy), cw2, ch,
                 boxstyle="round,pad=0.1", facecolor=WASH_DG,
                 edgecolor=DG, lw=1.0, alpha=0.3))
    axB.text(cx2 + cw2/2 - 0.1, cy + ch + 0.15, "canvas 2",
             ha="center", va="bottom", fontsize=8, color=DG, fontweight="medium")

    for i, l in enumerate(c2):
        tok(axB, cx2 + i * STEP, TY, l, DG, alpha=0.5)

    # Step arrow between canvases (below tokens, not through ellipsis)
    arr_y = TY - 0.15
    arr(axB, cx1 + cw1 - 0.05, arr_y, cx2 - 0.05, arr_y, DG, lw=1.5)

    axB.text(W/2, 0.6, "cost: flat per canvas, steps at boundary",
             ha="center", fontsize=8.5, color=DG, style="italic")

    # Save
    outdir = "paper/figures"
    for ext in ["pdf", "png"]:
        fig.savefig(f"{outdir}/fig0_decoding_schematic.{ext}",
                    dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote fig0_decoding_schematic")

if __name__ == "__main__":
    make()
