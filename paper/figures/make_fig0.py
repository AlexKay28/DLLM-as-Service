"""Generate the AR vs DG decoding schematic using paper-craft principles.

Style: paper-figure (clean, NeurIPS-like)
Layout: vertical, two panels (A: AR, B: DG)
Principle: one figure = one mechanism, understood without reading the paper.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Arc
import matplotlib.patheffects as pe
import numpy as np

# Palette (matches paper style)
AR_COLOR = "#2a78d6"
DG_COLOR = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
SURFACE = "#ffffff"
WASH_AR = "#eaf1fb"
WASH_DG = "#fdeee7"
WASH_NEUTRAL = "#f4f3f0"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Fira Sans", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

def draw_token(ax, x, y, w, h, label, color, alpha=0.9, fontsize=9, text_color="white"):
    """Draw a rounded token box with label."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                         facecolor=color, edgecolor="white", lw=1.0, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="medium")

def draw_arrow(ax, x1, y1, x2, y2, color, lw=1.4, style="->", ls="-"):
    """Draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls,
                                shrinkA=2, shrinkB=2))

def draw_cost_line(ax, xs, ys, color, label=None, label_x=None, label_y=None):
    """Draw a cost trajectory line with optional label."""
    ax.plot(xs, ys, "-", color=color, lw=2.5, zorder=3, solid_capstyle="round")
    if label:
        ax.text(label_x, label_y, label, fontsize=8, color=color,
                fontweight="medium", va="center")

def make_schematic():
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(6.3, 5.2),
                                    gridspec_kw={"hspace": 0.45})

    W, H = 12, 6

    # =================== Panel A: Autoregressive ===================
    axA.set_xlim(0, W)
    axA.set_ylim(0, H)
    axA.set_aspect("equal")
    axA.axis("off")

    # Title
    axA.text(0, H - 0.2, "A  Autoregressive decoding", fontsize=10,
             fontweight="medium", color=INK, va="top")
    axA.text(0, H - 0.7, "One forward pass per token. Cost grows linearly.",
             fontsize=8, color=MUTED, va="top")

    # Prompt
    draw_token(axA, 0.3, 4.0, 2.0, 0.8, "prompt", MUTED, alpha=0.3,
               fontsize=8, text_color=INK2)

    # Tokens with sequential arrows
    token_labels = ["$t_1$", "$t_2$", "$t_3$", "$t_4$", "$t_5$", "$t_6$"]
    tw, th = 1.0, 0.8
    spacing = 1.4
    start_x = 3.0

    for i, lab in enumerate(token_labels):
        x = start_x + i * spacing
        # Token box
        draw_token(axA, x, 4.0, tw, th, lab, AR_COLOR, fontsize=9)
        # Sequential arrow (can't start next until current finishes)
        if i < len(token_labels) - 1:
            draw_arrow(axA, x + tw + 0.02, 4.4, x + spacing - 0.02, 4.4,
                       AR_COLOR, lw=1.4)
        # Per-token cost label
        axA.text(x + tw/2, 3.5, "47ms", ha="center", va="center",
                 fontsize=7, color=AR_COLOR, fontweight="medium")

    # "wait" annotation between tokens
    axA.text(start_x + 0.5, 3.0, "must finish $t_i$\nbefore $t_{i+1}$",
             ha="center", va="center", fontsize=7, color=MUTED, style="italic")

    # Cost trajectory (linear)
    cost_y = 2.0
    for i in range(len(token_labels)):
        x = start_x + i * spacing + tw/2
        axA.plot(x, cost_y, "o", color=AR_COLOR, ms=4, zorder=3)
        if i > 0:
            x_prev = start_x + (i-1) * spacing + tw/2
            axA.plot([x_prev, x], [cost_y, cost_y], "-", color=AR_COLOR, lw=2)
    # Arrow showing growth direction
    draw_arrow(axA, start_x + 4*spacing + tw/2, cost_y,
               start_x + 6*spacing, cost_y, AR_COLOR, lw=2)
    axA.text(start_x + 6*spacing + 0.2, cost_y, "linear\n$T = a + b \\cdot L$",
             fontsize=7.5, color=AR_COLOR, va="center", fontweight="medium",
             linespacing=1.2)

    # Cost annotation
    axA.text(W/2, 0.8, "Every token costs exactly $b$ ms. No parallelism.",
             ha="center", fontsize=8, color=MUTED, style="italic")

    # =================== Panel B: Block-diffusion ===================
    axB.set_xlim(0, W)
    axB.set_ylim(0, H)
    axB.set_aspect("equal")
    axB.axis("off")

    # Title
    axB.text(0, H - 0.2, "B  Block-diffusion decoding", fontsize=10,
             fontweight="medium", color=INK, va="top")
    axB.text(0, H - 0.7, "$K$ denoising passes per canvas. Tokens within a canvas are free.",
             fontsize=8, color=MUTED, va="top")

    # Prompt
    draw_token(axB, 0.3, 4.0, 2.0, 0.8, "prompt", MUTED, alpha=0.3,
               fontsize=8, text_color=INK2)

    # Canvas 1: all tokens in parallel
    canvas1 = ["$t_1$", "$t_2$", "$t_3$", "$t_4$"]
    cx1 = 3.0
    cw1 = len(canvas1) * spacing + 0.2

    # Canvas background
    canvas_box1 = FancyBboxPatch((cx1 - 0.15, 3.7), cw1, 1.4,
                                  boxstyle="round,pad=0.12",
                                  facecolor=WASH_DG, edgecolor=DG_COLOR,
                                  lw=1.2, alpha=0.6)
    axB.add_patch(canvas_box1)
    axB.text(cx1 + cw1/2 - 0.1, 4.85, "canvas 1 ($C = 256$ tokens)",
             ha="center", va="center", fontsize=8, color=DG_COLOR,
             fontweight="medium")

    for i, lab in enumerate(canvas1):
        x = cx1 + i * spacing
        draw_token(axB, x, 4.0, tw, th, lab, DG_COLOR, alpha=0.9, fontsize=9)

    # "all at once" annotation
    axB.annotate("all tokens\nin parallel", xy=(cx1 + cw1/2, 3.75),
                 xytext=(cx1 + cw1/2, 2.8), fontsize=7, color=DG_COLOR,
                 ha="center", va="center", style="italic", linespacing=1.2,
                 arrowprops=dict(arrowstyle="->", color=DG_COLOR, lw=0.8))

    # Denoising loop indicator
    for k in range(3):
        y = 2.2 - k * 0.3
        axB.annotate("", xy=(cx1 + cw1 - 0.3, y), xytext=(cx1 + 0.3, y),
                     arrowprops=dict(arrowstyle="->", color=DG_COLOR,
                                     lw=0.7, ls="--", alpha=0.4))
    axB.text(cx1 - 0.5, 1.7, "$K{=}8$\nrefine\npasses", ha="center",
             va="center", fontsize=7, color=DG_COLOR, fontweight="medium",
             linespacing=1.2)

    # Canvas 2
    canvas2 = ["$t_5$", "$t_6$"]
    cx2 = cx1 + cw1 + 0.9
    cw2 = len(canvas2) * spacing + 0.2

    canvas_box2 = FancyBboxPatch((cx2 - 0.15, 3.7), cw2, 1.4,
                                  boxstyle="round,pad=0.12",
                                  facecolor=WASH_DG, edgecolor=DG_COLOR,
                                  lw=1.2, alpha=0.3)
    axB.add_patch(canvas_box2)
    axB.text(cx2 + cw2/2 - 0.1, 4.85, "canvas 2",
             ha="center", va="center", fontsize=8, color=DG_COLOR,
             fontweight="medium")

    for i, lab in enumerate(canvas2):
        x = cx2 + i * spacing
        draw_token(axB, x, 4.0, tw, th, lab, DG_COLOR, alpha=0.5, fontsize=9)

    # Step between canvases
    draw_arrow(axB, cx1 + cw1 - 0.1, 4.4, cx2 - 0.05, 4.4, DG_COLOR, lw=1.8)
    step_x = (cx1 + cw1 + cx2) / 2
    axB.text(step_x, 4.8, "+120ms", ha="center", va="bottom",
             fontsize=8, color=DG_COLOR, fontweight="medium")

    # Staircase cost
    y1, y2 = 1.2, 0.5
    x1_end = cx1 + cw1 - 0.15
    x2_end = cx2 + cw2 - 0.15

    # Flat segment 1
    axB.plot([cx1 + 0.2, x1_end], [y1, y1], "-", color=DG_COLOR, lw=2.5,
             zorder=3, solid_capstyle="round")
    # Step down
    axB.plot([x1_end, x1_end], [y1, y2], "-", color=DG_COLOR, lw=2.5, zorder=3)
    # Flat segment 2
    axB.plot([x1_end, x2_end], [y2, y2], "-", color=DG_COLOR, lw=2.5,
             zorder=3, solid_capstyle="round")

    axB.text(cx1 + cw1/2, y1 + 0.25, "flat: tokens are free",
             ha="center", fontsize=7, color=DG_COLOR, style="italic")
    axB.text(x1_end + 0.15, (y1 + y2)/2, "step",
             fontsize=7, color=DG_COLOR, style="italic", va="center")

    # Save
    import os
    outdir = "paper/figures"
    os.makedirs(outdir, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(f"{outdir}/fig0_decoding_schematic.{ext}",
                    dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote fig0_decoding_schematic")

if __name__ == "__main__":
    make_schematic()
