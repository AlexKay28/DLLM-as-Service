"""Shared visual system for every figure in the paper.

One palette, one type scale, one set of mark specs — so the figure set reads
as a single system rather than seven unrelated plots.

Palette provenance: categorical slots 1/2/3 of the validated default
data-viz palette (blue / orange / aqua). Validated on a white surface with
`--pairs all`: worst CVD dE 9.2 (deutan), worst normal-vision dE 24.0, all
checks PASS. Aqua sits below 3:1 contrast on white, so it is only ever used
with a direct label (the "relief rule"), never as the sole carrier of
identity.

Greyscale safety: every series also differs in marker shape and line style,
so the figures survive a black-and-white print.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------------------------------------------------- palette ---

AR = "#2a78d6"       # slot 1 — autoregressive (Qwen2.5)
DG = "#eb6834"       # slot 2 — block diffusion (DiffusionGemma)
LLADA = "#1baf7a"    # slot 3 — LLaDA-8B (second diffusion family)

INK = "#0b0b0b"      # primary ink
INK2 = "#52514e"     # secondary ink
MUTED = "#898781"    # axis labels, annotations
GRID = "#e8e7e1"     # hairline gridline
AXIS = "#c3c2b7"     # baseline / axis rule
SURFACE = "#ffffff"  # the page

# Washes for regime bands — neutral, never a series hue.
WASH_AR = "#eaf1fb"
WASH_DG = "#fdeee7"
WASH_NEUTRAL = "#f4f3f0"

STATUS_GOOD = "#0ca30c"
STATUS_BAD = "#d03b3b"

# Marker + linestyle per series, so identity is never colour-alone.
MARK = {"ar": "o", "dg": "s", "llada": "^"}
DASH = {"ar": "-", "dg": "-", "llada": (0, (4, 2))}

# ------------------------------------------------------------------ sizes ---

FULL = 6.30   # inches — \linewidth at 11pt / 1.05in margins on letter
HALF = 3.05

# ------------------------------------------------------------------- type ---

SANS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def apply():
    """Install the house style. Call once before drawing anything."""
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # recessive chrome
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 7,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",       # never dashed
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.handlelength": 1.9,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.4,
        "legend.labelspacing": 0.35,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.2,
        "lines.markeredgewidth": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def despine(ax, left=True, bottom=True):
    """Drop the top/right rules; keep the two that carry scale."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def title(ax, text, sub=None):
    """Left-aligned title with an optional muted sub-line."""
    if sub:
        ax.set_title(f"{text}\n", loc="left", color=INK, fontweight="medium")
        ax.text(0, 1.012, sub, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.0, color=MUTED)
    else:
        ax.set_title(text, loc="left", color=INK, fontweight="medium")


def label_end(ax, x, y, text, color, dx=6, dy=0, ha="left", va="center",
              weight="medium", size=8.2):
    """Direct-label a series at its endpoint instead of relying on a legend."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, color=color, fontsize=size, fontweight=weight)


def note(ax, x, y, text, color=MUTED, size=7.6, **kw):
    """A quiet annotation in axes-fraction coordinates."""
    ax.text(x, y, text, transform=ax.transAxes, color=color, fontsize=size, **kw)


def save(fig, name, outdir):
    """Emit PDF (for print) and PNG (for review) under one name."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}")
