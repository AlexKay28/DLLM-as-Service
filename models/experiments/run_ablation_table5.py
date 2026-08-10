"""
Validate ablation claims from FlowFill paper Table 5 & K=8 knee (Issue #6).

Two ablations:
  A) Semi-AR blocks: B=1/2/4 (trades latency for cross-field consistency)
  B) K-sweep: K in {2,4,6,8,10,12,16,24} (finds the quality-vs-steps knee)

BLOCKED on FlowFill-MDM weights. These drafts use an open MDM substitute
(LLaDA-8B) if available, or document the blocker.

Usage (GPU + MDM weights required):
  # Claim A: Semi-AR block ablation
  python models/experiments/run_ablation_table5.py --model LLaDA-8B --ablation blocks

  # Claim B: K-sweep
  python models/experiments/run_ablation_table5.py --model LLaDA-8B --ablation k-sweep

  # Both
  python models/experiments/run_ablation_table5.py --model LLaDA-8B --ablation all

Outputs:
  results/ablation_table5.json
  results/ablation_table5.md
  results/ablation_knee.json
  results/ablation_knee.png
"""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES, FieldCategory, FIELD_BY_NAME
from metrics.quality_metrics import field_f1, schema_validity, consistency
from metrics.latency_metrics import latency_percentiles

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"
RESULTS_DIR = _REPO_ROOT / "results"

PAPER_TABLE5 = {
    "B=1 (pure parallel)": {"field_f1": 0.907, "consistency_pct": 97.8, "p50_ms": 23,  "speedup": 7.3},
    "B=2":                 {"field_f1": 0.912, "consistency_pct": 98.6, "p50_ms": 29,  "speedup": 5.8},
    "B=4":                 {"field_f1": 0.914, "consistency_pct": 99.0, "p50_ms": 41,  "speedup": 4.1},
}

PAPER_KNEE = {
    "K=4":  {"field_f1": None, "p50_ms": 12},
    "K=6":  {"field_f1": None, "p50_ms": 18},
    "K=8":  {"field_f1": 0.907, "p50_ms": 23},
    "K=12": {"field_f1": None, "p50_ms": 35},
    "K=16": {"field_f1": None, "p50_ms": 46},
}

K_SWEEP_VALUES = [2, 4, 6, 8, 10, 12, 16, 24]
BLOCK_VALUES = [1, 2, 4]

# 12 fields split into B blocks (even split, remainder in last block)
FIELD_BLOCK_SPLITS = {
    1: [[f for f in FIELD_NAMES]],
    2: [FIELD_NAMES[:6], FIELD_NAMES[6:]],
    4: [FIELD_NAMES[:3], FIELD_NAMES[3:6], FIELD_NAMES[6:9], FIELD_NAMES[9:]],
}


def load_log_lines(path, n=None):
    lines = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def load_records(path, n=None):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                records.append(Record(fields=json.loads(line)))
    return records


# ---------------------------------------------------------------------------
# Claim A: Semi-AR block ablation (Table 5)
# ---------------------------------------------------------------------------

def run_block_ablation(model, log_lines, gold_records, source, model_name, n=5000):
    """Run B=1/2/4 block ablation.

    BLOCKED: requires MDM model that supports block-sequential denoising.
    The draft interface assumes the model wrapper exposes a `denoise` method
    with a `blocks` parameter. Real implementation TBD when weights available.
    """
    results = []
    for B in BLOCK_VALUES:
        print(f"\n  B={B} (blocks={FIELD_BLOCK_SPLITS[B]})...", end=" ", flush=True)

        # DRAFT: this calls model.extract with a blocks parameter.
        # The real MDM wrapper needs to implement block-sequential denoising.
        try:
            t0 = time.perf_counter()
            if hasattr(model, 'extract_with_blocks'):
                result = model.extract_with_blocks(
                    log_lines[:n], source=source, blocks=B)
            else:
                # Fallback: use regular extract (ignores blocks)
                result = model.extract(log_lines[:n], source=source,
                                       show_progress=False)
            elapsed = time.perf_counter() - t0

            f1_res = field_f1(result.records, gold_records[:len(result.records)])
            cons = consistency(result.records)
            lat = latency_percentiles(result.latencies_ms)

            entry = {
                "B": B,
                "field_f1": round(f1_res["macro_field_f1"], 4),
                "consistency_pct": round(cons, 2),
                "p50_ms": round(lat.get("p50_ms", 0), 2),
                "n_events": len(result.records),
            }
            ar_p50 = 168  # paper's AR-1.5B p50
            entry["speedup_vs_ar"] = round(ar_p50 / entry["p50_ms"], 1) if entry["p50_ms"] > 0 else 0
            results.append(entry)
            print(f"F1={entry['field_f1']:.3f} Cons={entry['consistency_pct']:.1f}% "
                  f"p50={entry['p50_ms']:.1f}ms ({entry['speedup_vs_ar']}x)")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"B": B, "error": str(e)})

    return results


# ---------------------------------------------------------------------------
# Claim B: K-sweep (K=8 knee)
# ---------------------------------------------------------------------------

def run_k_sweep(model, log_lines, gold_records, source, model_name, n=5000):
    """Sweep K (denoising steps) and measure quality + latency.

    BLOCKED: requires MDM model that exposes a `k_steps` parameter.
    """
    results = []
    for K in K_SWEEP_VALUES:
        print(f"\n  K={K}...", end=" ", flush=True)

        try:
            t0 = time.perf_counter()
            if hasattr(model, 'extract_with_k'):
                result = model.extract_with_k(
                    log_lines[:n], source=source, k_steps=K)
            else:
                result = model.extract(log_lines[:n], source=source,
                                       show_progress=False)
            elapsed = time.perf_counter() - t0

            f1_res = field_f1(result.records, gold_records[:len(result.records)])
            lat = latency_percentiles(result.latencies_ms)

            entry = {
                "K": K,
                "field_f1": round(f1_res["macro_field_f1"], 4),
                "p50_ms": round(lat.get("p50_ms", 0), 2),
                "n_events": len(result.records),
            }
            results.append(entry)
            print(f"F1={entry['field_f1']:.3f} p50={entry['p50_ms']:.1f}ms")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"K": K, "error": str(e)})

    return results


def plot_knee(k_results, path):
    """Plot Field-F1 vs K and p50 vs K to visualize the knee."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in k_results if "error" not in r]
    if not valid:
        print("  No valid K-sweep results to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ks = [r["K"] for r in valid]
    f1s = [r["field_f1"] for r in valid]
    p50s = [r["p50_ms"] for r in valid]

    ax1.plot(ks, f1s, "o-", color="#DD8452", linewidth=2, markersize=8)
    ax1.axvline(x=8, color="gray", linestyle="--", alpha=0.5, label="Paper's K=8")
    ax1.set_xlabel("K (denoising steps)")
    ax1.set_ylabel("Field-F1")
    ax1.set_title("Quality vs K")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ks, p50s, "s-", color="#4C72B0", linewidth=2, markersize=8)
    ax2.axvline(x=8, color="gray", linestyle="--", alpha=0.5, label="Paper's K=8")
    ax2.set_xlabel("K (denoising steps)")
    ax2.set_ylabel("p50 latency (ms)")
    ax2.set_title("Latency vs K (should be linear)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("K-Sweep: Finding the Quality-vs-Steps Knee", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Wrote {path}")


def write_report(block_results, k_results, path):
    lines = [
        "# Ablation Claims — Issue #6: Table 5 & K=8 Knee",
        "",
        f"**Date:** 2026-08-11",
        f"**STATUS:** DRAFT — requires MDM model weights (FlowFill or LLaDA substitute)",
        "",
        "---",
        "",
        "## Claim A: Semi-AR Block Ablation (Table 5)",
        "",
        "| B | Field-F1 | Consistency | p50 (ms) | Speedup vs AR |",
        "|---|---------|-------------|---------|--------------|",
    ]
    for r in block_results:
        if "error" in r:
            lines.append(f"| {r['B']} | ERROR | — | — | — |")
        else:
            lines.append(f"| {r['B']} | {r['field_f1']:.3f} | "
                         f"{r['consistency_pct']:.1f}% | {r['p50_ms']:.0f} | "
                         f"{r['speedup_vs_ar']}x |")
    lines.append("| | | | | |")
    for name, p in PAPER_TABLE5.items():
        lines.append(f"| {name} (paper) | {p['field_f1']:.3f} | "
                     f"{p['consistency_pct']:.1f}% | {p['p50_ms']} | "
                     f"{p['speedup']}x |")

    lines.extend([
        "",
        "## Claim B: K=8 Knee",
        "",
        "| K | Field-F1 | p50 (ms) |",
        "|---|---------|---------|",
    ])
    for r in k_results:
        if "error" in r:
            lines.append(f"| {r['K']} | ERROR | — |")
        else:
            lines.append(f"| {r['K']} | {r['field_f1']:.3f} | {r['p50_ms']:.0f} |")

    lines.extend([
        "",
        "## Allegations to verify",
        "",
        "### Claim A (Semi-AR blocks)",
        "1. B=2 recovers 0.8pp consistency (97.8 -> 98.6) at +6ms",
        "2. B=4 recovers 1.2pp consistency (97.8 -> 99.0) at +18ms",
        "3. Field-F1 improves with more blocks (0.907 -> 0.912 -> 0.914)",
        "4. Even B=4 stays 4.1x faster than AR",
        "5. Speedup scales linearly with B",
        "",
        "### Claim B (K=8 knee)",
        "6. Quality saturates by K=8",
        "7. Latency scales linearly in K (~3ms/step)",
        "8. K is content-independent (same K for short/long messages)",
        "",
        "**BLOCKER:** Both ablations require MDM weights. Draft code committed;",
        "needs FlowFill-MDM or LLaDA-8B substitute to run.",
        "",
        "---",
        "*Generated by `models/experiments/run_ablation_table5.py`*",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="LLaDA-8B",
                        help="MDM model (FlowFill or LLaDA substitute)")
    parser.add_argument("--source", default="bgl")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--ablation", default="all",
                        choices=["all", "blocks", "k-sweep"])
    parser.add_argument("--mock", action="store_true",
                        help="Use MockARModel (will not produce real MDM results)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.source} test data...")
    log_lines = load_log_lines(str(SPLIT_DIR / f"{args.source}.raw"), n=args.n)
    gold_records = load_records(str(SPLIT_DIR / f"{args.source}.jsonl"), n=args.n)
    print(f"  {len(log_lines):,} log lines, {len(gold_records):,} gold records")

    # Load model
    # NOTE: MDM wrapper not yet implemented. This is the blocker.
    if args.mock:
        from models.baselines.ar_wrapper import MockARModel
        model = MockARModel(model_name=args.model)
        print("WARNING: Using MockARModel — results are NOT valid MDM ablation data.")
    else:
        print("ERROR: MDM model wrapper not yet implemented.")
        print("To run ablations, implement an MDM wrapper that exposes:")
        print("  - extract_with_blocks(log_lines, source, blocks=B)")
        print("  - extract_with_k(log_lines, source, k_steps=K)")
        print("Using --mock to produce placeholder output...")
        from models.baselines.ar_wrapper import MockARModel
        model = MockARModel(model_name=args.model)

    block_results = []
    k_results = []

    if args.ablation in ("all", "blocks"):
        print(f"\n{'='*60}")
        print("Claim A: Semi-AR Block Ablation (Table 5)")
        print(f"{'='*60}")
        block_results = run_block_ablation(
            model, log_lines, gold_records, args.source, args.model, n=args.n)

    if args.ablation in ("all", "k-sweep"):
        print(f"\n{'='*60}")
        print("Claim B: K-Sweep (K=8 Knee)")
        print(f"{'='*60}")
        k_results = run_k_sweep(
            model, log_lines, gold_records, args.source, args.model, n=args.n)
        if k_results:
            plot_knee(k_results, RESULTS_DIR / "ablation_knee.png")

    output = {
        "source": args.source,
        "n_events": args.n,
        "model": args.model,
        "block_ablation": block_results,
        "k_sweep": k_results,
        "paper_table5": PAPER_TABLE5,
        "paper_knee": PAPER_KNEE,
    }
    json_path = RESULTS_DIR / "ablation_table5.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    write_report(block_results, k_results, RESULTS_DIR / "ablation_table5.md")
    print("Done!")


if __name__ == "__main__":
    main()
