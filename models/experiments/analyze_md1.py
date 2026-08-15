#!/usr/bin/env python3
"""Issue #48 — M/D/1 queueing analysis: compare measured lag to theoretical
M/D/1 predictions for Poisson and MMPP arrivals.

M/D/1 (Poisson arrivals, deterministic service time):
  E[W] = (ρ/(1-ρ)) * (1 - ρ/2) * S       (mean wait)
  p99(W) ≈ S * ln(100/(1-ρ))             (tail approximation)
  where ρ = λ * S (utilization), S = service time

MMPP/M/1 is not analytically solvable, but we can compare the bursty
arrival lag to the Poisson lag at the same average rate to see if the
M/D/1 knee prediction holds under bursty traffic.

Usage:
    python models/experiments/analyze_md1.py results/rate_lag_ar3b.json
    python models/experiments/analyze_md1.py results/rate_lag_ar3b_mmpp.json --mode mmpp
"""

import argparse
import json
import math
import sys
from pathlib import Path


def md1_wait_mean(rate, service_s):
    """M/D/1 mean waiting time (Kingman/Erlang)."""
    rho = rate * service_s
    if rho >= 1:
        return float("inf")
    return (rho / (1 - rho)) * (1 - rho / 2) * service_s


def md1_wait_p(rho, S, p):
    """M/D/1 p-th percentile wait approximation.

    For M/D/1, the waiting time CDF is complex, but a reasonable
    approximation for the tail is:
      W_p ≈ S * -ln(1-p) / (1-ρ)    for ρ not too close to 1
    """
    if rho >= 1:
        return float("inf")
    return S * (-math.log(1 - p)) / (1 - rho)


def md1_lag_p(rate, service_s, p):
    """Total lag = wait + service. p-th percentile."""
    rho = rate * service_s
    if rho >= 1:
        return float("inf")
    return (md1_wait_p(rho, service_s, p) + service_s) * 1000  # ms


def find_service_time(records):
    """Estimate deterministic service time S from the low-rate baseline.

    At very low utilization (ρ→0), lag ≈ S (no queueing).
    Use the p50 at the lowest rate as the service time estimate.
    """
    rates = sorted(records, key=lambda r: r["offered_rate_ev_s"])
    if not rates:
        return None
    baseline = rates[0]
    p50 = baseline["lag_ms"].get("p50_ms")
    if p50 and p50 > 0:
        return p50 / 1000.0  # seconds
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="path to rate_lag JSON file")
    ap.add_argument("--mode", default="poisson", choices=["poisson", "mmpp"])
    ap.add_argument("--service-ms", type=float, default=None,
                    help="override service time (ms); auto-detected if omitted")
    args = ap.parse_args()

    with open(args.path) as f:
        data = json.load(f)

    rates = data.get("rates", [])
    if not rates:
        print("No rate data found")
        sys.exit(1)

    if args.service_ms:
        S = args.service_ms / 1000.0
    else:
        S = find_service_time(rates)
        if S is None:
            print("Could not estimate service time from data")
            sys.exit(1)

    print(f"Service time S = {S*1000:.1f} ms (auto-detected from lowest rate)")
    print(f"Mode: {args.mode}")
    print()
    print(f"{'rate':>6s}  {'ρ':>5s}  {'meas p50':>9s}  {'M/D/1 p50':>10s}  "
          f"{'ratio':>6s}  {'meas p99':>9s}  {'M/D/1 p99':>10s}  {'ratio':>6s}  "
          f"{'knee?':>5s}")
    print("-" * 85)

    knee_predicted = 1.0 / S  # rate where ρ=1
    for r in rates:
        rate = r["offered_rate_ev_s"]
        rho = rate * S
        meas_p50 = r["lag_ms"].get("p50_ms", 0)
        meas_p99 = r["lag_ms"].get("p99_ms", 0)

        if args.mode == "poisson" and rho < 1:
            pred_p50 = md1_lag_p(rate, S, 0.5)
            pred_p99 = md1_lag_p(rate, S, 0.99)
            r50 = meas_p50 / pred_p50 if pred_p50 > 0 else float("nan")
            r99 = meas_p99 / pred_p99 if pred_p99 > 0 else float("nan")
            at_knee = "KNEE" if abs(rho - 0.8) < 0.1 else ""
        elif rho >= 1:
            pred_p50 = float("inf")
            pred_p99 = float("inf")
            r50 = float("inf")
            r99 = float("inf")
            at_knee = "SAT"
        else:
            pred_p50 = md1_lag_p(rate, S, 0.5)
            pred_p99 = md1_lag_p(rate, S, 0.99)
            r50 = meas_p50 / pred_p50 if pred_p50 > 0 else float("nan")
            r99 = meas_p99 / pred_p99 if pred_p99 > 0 else float("nan")
            at_knee = ""

        print(f"{rate:6.1f}  {rho:5.2f}  {meas_p50:9.0f}  "
              f"{pred_p50:10.0f}  {r50:6.2f}  {meas_p99:9.0f}  "
              f"{pred_p99:10.0f}  {r99:6.2f}  {at_knee:>5s}")

    print()
    print(f"M/D/1 knee (ρ=1): {knee_predicted:.1f} ev/s")
    print(f"M/D/1 knee (ρ=0.8, heavy traffic): {0.8*knee_predicted:.1f} ev/s")

    # Check if measured saturation matches predicted
    sat_found = None
    for r in rates:
        if r["lag_ms"].get("p99_ms", 0) > r["lag_ms"].get("p50_ms", 0) * 3:
            sat_found = r["offered_rate_ev_s"]
            break

    if sat_found:
        print(f"Measured saturation (p99 > 3*p50): {sat_found:.1f} ev/s")
        print(f"M/D/1 prediction accuracy: "
              f"{sat_found/knee_predicted:.1%} of knee")

    result = {
        "service_time_ms": round(S * 1000, 1),
        "mode": args.mode,
        "md1_knee_ev_s": round(knee_predicted, 1),
        "md1_knee_0_8_ev_s": round(0.8 * knee_predicted, 1),
        "measured_saturation_ev_s": sat_found,
        "saturation_vs_knee_pct": round(sat_found / knee_predicted * 100, 1)
        if sat_found else None,
    }

    out_path = str(Path(args.path).with_suffix(".md1_analysis.json"))
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
