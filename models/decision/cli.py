#!/usr/bin/env python3
"""CLI for the decision procedure.

Usage:
    python -m models.decision --L 512 --lambda 1.0 --no-batching --f1-tolerance 0.05

    python -m models.decision --L 256 --lambda 64 --batching --f1-tolerance 0.05

    # Use the two-variable DG law (after #44 lands)
    python -m models.decision --L 512 --lambda 1.0 --no-batching --f1-tolerance 0.05 --two-variable --l-in 2048

    # Report under min/max step
    python -m models.decision --L 512 --lambda 1.0 --no-batching --f1-tolerance 0.05 --step-range
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.decision.procedure import (
    decide, CONSTANTS, VALIDITY_DOMAIN,
    compute_l_cross, compute_lambda_cross,
    t_dg_one_variable, t_dg_two_variable,
)


def main():
    ap = argparse.ArgumentParser(
        description="Decision procedure: AR vs diffusion LLM for structured extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Validity domain:
  Hardware: one A100 80GB PCIe, Campaign B
  Runtime: vLLM 0.25.1, BF16, serial diffusion (--max-num-seqs 1)
  Models: DiffusionGemma-26B-A4B-it vs Qwen2.5-3B-Instruct (NOT size-matched)
  Task: 12-field structured extraction over BGL, few-shot ~1047 tokens

  MUST NOT be applied to: input lengths >> 1047 tokens (pending #44),
  batch-served diffusion, non-extractive tasks, canvas != 256,
  other GPUs/runtimes, or 12-field F1 tolerances (contaminated, use 9-field).
""")
    ap.add_argument("--L", type=float, required=True,
                    help="expected output length (tokens)")
    ap.add_argument("--lambda", dest="lam", type=float, required=True,
                    help="arrival rate (events/s)")
    ap.add_argument("--batching", action="store_true", default=False,
                    help="cross-request batching is available")
    ap.add_argument("--no-batching", action="store_false", dest="batching",
                    help="cross-request batching is NOT available (default)")
    ap.add_argument("--f1-tolerance", type=float, required=True,
                    help="Field-F1 the caller will give up to gain latency")
    ap.add_argument("--f1-metric", choices=["9field", "12field"],
                    default="9field",
                    help="which F1 metric the tolerance is against (default: 9field)")
    ap.add_argument("--l-in", type=float, default=1047,
                    help="expected input length (tokens, default 1047)")
    ap.add_argument("--two-variable", action="store_true",
                    help="use the two-variable DG law T_DG(L_in, L_out)")
    ap.add_argument("--step-range", action="store_true",
                    help="report recommendation under min and max published step")
    ap.add_argument("--json", action="store_true",
                    help="output as JSON")
    args = ap.parse_args()

    t_dg_fn = t_dg_two_variable if args.two_variable else t_dg_one_variable

    result = decide(
        L=args.L, lam=args.lam, batching=args.batching,
        f1_tolerance=args.f1_tolerance,
        l_in=args.l_in, t_dg_fn=t_dg_fn,
        f1_metric=args.f1_metric,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(result)
    print()
    if result.warnings:
        for w in result.warnings:
            print(f"  WARNING: {w}")
        print()

    if args.step_range:
        step_min = CONSTANTS["dg_step_min_ms"]["value"]
        step_max = CONSTANTS["dg_step_max_ms"]["value"]
        r_min = decide(L=args.L, lam=args.lam, batching=args.batching,
                       f1_tolerance=args.f1_tolerance, l_in=args.l_in,
                       t_dg_fn=t_dg_fn, step=step_min,
                       f1_metric=args.f1_metric)
        r_max = decide(L=args.L, lam=args.lam, batching=args.batching,
                       f1_tolerance=args.f1_tolerance, l_in=args.l_in,
                       t_dg_fn=t_dg_fn, step=step_max,
                       f1_metric=args.f1_metric)
        print(f"  Under step={step_min} ms: arm={r_min.arm}  "
              f"ratio={r_min.ratio:.2f}x")
        print(f"  Under step={step_max} ms: arm={r_max.arm}  "
              f"ratio={r_max.ratio:.2f}x")
        if r_min.arm == r_max.arm:
            print("  → Recommendation is ROBUST to step range.")
        else:
            print("  → Recommendation is SENSITIVE to step range.")

    # Show crossovers
    l_cross = compute_l_cross()
    lambda_cross = compute_lambda_cross()
    print(f"\n  Crossovers: L*={l_cross:.1f} tokens  λ*={lambda_cross:.1f} ev/s")


if __name__ == "__main__":
    main()
