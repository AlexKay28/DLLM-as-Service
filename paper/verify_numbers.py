#!/usr/bin/env python3
"""
verify_numbers.py -- re-verify every number in paper/NUMBERS.md against the
results/*.json artifacts it cites.

Usage (from the repository root, or from anywhere -- paths are resolved
relative to this file):

    python3 paper/verify_numbers.py
    python3 paper/verify_numbers.py -v      # also print the artifact value

Contract
--------
* stdlib only (json, os, sys, math, argparse).
* One PASS/FAIL line per checked quantity, naming the artifact it came from.
* Exit code 0 iff every check passed. Any FAIL -> exit 1.
* Tolerance: 0.5% relative, OR half a unit in the last decimal place of the
  value as written in NUMBERS.md, whichever is looser. (NUMBERS.md rounds for
  display -- e.g. it writes 0.844 for an artifact value of 0.8435 -- and a
  rounding difference is not a mismatch.)
* Known documented discrepancies are reported as WARNING and do NOT fail the
  run. They are listed in the summary so they cannot be forgotten; each one is
  a submission blocker tracked in NUMBERS.md.

Nothing in this script is authoritative. NUMBERS.md is authoritative for what
the paper may say; this script only asserts that NUMBERS.md still matches the
artifacts. If a check fails, fix NUMBERS.md and macros.tex together -- never
edit a number into a .tex file directly.
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

REL_TOL = 0.005  # 0.5%

# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------

STATE = {"pass": 0, "fail": 0, "skip": 0, "warn": 0}
FAILURES = []
WARNINGS = []
VERBOSE = False
_CACHE = {}


def section(title):
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


def load(relpath):
    """Load results/<relpath>. Returns None (and records a FAIL) if missing."""
    if relpath in _CACHE:
        return _CACHE[relpath]
    path = os.path.join(RESULTS, relpath)
    try:
        with open(path, "r") as fh:
            obj = json.load(fh)
    except Exception as exc:  # missing file, bad JSON
        obj = None
        fail("artifact %s" % relpath, "could not be read: %s" % exc)
    _CACHE[relpath] = obj
    return obj


def fail(label, reason):
    STATE["fail"] += 1
    FAILURES.append("%s -- %s" % (label, reason))
    print("FAIL  %-58s %s" % (label, reason))


def warn(label, reason):
    STATE["warn"] += 1
    WARNINGS.append("%s -- %s" % (label, reason))
    print("WARN  %-58s %s" % (label, reason))


def ok(label, detail=""):
    STATE["pass"] += 1
    if VERBOSE and detail:
        print("PASS  %-58s %s" % (label, detail))
    else:
        print("PASS  %s" % label)


def _decimals(value):
    s = repr(float(value))
    if "e" in s or "E" in s:
        return 0
    if "." in s:
        return len(s.split(".", 1)[1])
    return 0


def tolerance(expected, atol=None):
    display = 0.5 * (10.0 ** -_decimals(expected))
    tol = max(REL_TOL * abs(float(expected)), display)
    if atol is not None:
        tol = max(tol, float(atol))
    return tol + 1e-12


def close(expected, actual, atol=None):
    return abs(float(actual) - float(expected)) <= tolerance(expected, atol)


def dig(obj, path, relpath):
    """Walk a list of keys/indices. Returns (found, value)."""
    cur = obj
    for key in path:
        try:
            if isinstance(cur, list):
                cur = cur[key]
            else:
                cur = cur[key]
        except Exception:
            return False, None
    return True, cur


def check(label, expected, actual, source, atol=None):
    """Compare a NUMBERS.md value against an artifact value.

    atol widens the tolerance for quantities NUMBERS.md states as approximate
    (e.g. "~45x"); it is never used to paper over a real mismatch.
    """
    if actual is None:
        fail(label, "value not found in %s" % source)
        return False
    try:
        actual_f = float(actual)
    except (TypeError, ValueError):
        fail(label, "non-numeric value %r in %s" % (actual, source))
        return False
    if close(expected, actual_f, atol):
        ok(label, "expected %s, artifact %s (%s)" % (expected, actual_f, source))
        return True
    fail(
        label,
        "NUMBERS.md says %s, %s says %s (tol %.4g)"
        % (expected, source, actual_f, tolerance(expected, atol)),
    )
    return False


def check_path(label, expected, relpath, *path):
    obj = load(relpath)
    if obj is None:
        STATE["skip"] += 1
        return False
    found, value = dig(obj, list(path), relpath)
    if not found:
        fail(label, "path %s missing from %s" % ("/".join(map(str, path)), relpath))
        return False
    return check(label, expected, value, relpath)


# --------------------------------------------------------------------------
# artifact-shape helpers
# --------------------------------------------------------------------------


def single_model(relpath):
    """results/*.json written by the serving harness hold exactly one model."""
    obj = load(relpath)
    if obj is None:
        return None
    models = obj.get("models")
    if isinstance(models, dict) and len(models) >= 1:
        return list(models.values())[0]
    return None


def streaming(relpath):
    m = single_model(relpath)
    return None if m is None else m.get("streaming_table3")


def lat(relpath, key):
    s = streaming(relpath)
    if s is None:
        return None
    return s.get("latency", {}).get(key)


def qual(relpath, key):
    s = streaming(relpath)
    if s is None:
        return None
    return s.get("quality", {}).get(key)


def perfield(relpath, field):
    s = streaming(relpath)
    if s is None:
        return None
    return s.get("quality", {}).get("per_field_f1", {}).get(field)


def model_row(relpath, name_fragment):
    """quality_table2_*.json hold a list of model dicts."""
    obj = load(relpath)
    if obj is None:
        return None
    for row in obj.get("models", []):
        if name_fragment.lower() in str(row.get("model", "")).lower():
            return row
    return None


# ==========================================================================
# 1. The AR baseline correction (NUMBERS.md sect. 1) -- Campaigns A and D
# ==========================================================================


def check_ar_baseline():
    section("NUMBERS.md sect.1 -- AR baseline correction (Campaigns A, D)")

    # --- 1.3 corrected streaming latency, vLLM (Campaign A, n=2000) --------
    served = [
        ("AR-1.5B", "vllm_serving_1.5b.json", 413.3, 685.5, 805.8, 1.95, 0.844),
        ("AR-3B", "vllm_serving_3b.json", 662.0, 1062.1, 1336.3, 2.02, 0.826),
        ("AR-7B", "vllm_serving_7b.json", 1161.1, 1521.1, 2310.9, 1.99, 0.842),
    ]
    for name, f, p50, p99, p999, tail, f1 in served:
        check("sect1.3 %s p50 (ms)" % name, p50, lat(f, "p50_ms"), f)
        check("sect1.3 %s p99 (ms)" % name, p99, lat(f, "p99_ms"), f)
        check("sect1.3 %s p99.9 (ms)" % name, p999, lat(f, "p99_9_ms"), f)
        s = streaming(f)
        check(
            "sect1.3 %s tail ratio p99.9/p50" % name,
            tail,
            None if s is None else s.get("tail_ratio_p999_over_p50"),
            f,
        )
        check("sect1.3 %s Field-F1" % name, f1, qual(f, "field_f1"), f)

    # --- 1.4 throughput under continuous batching --------------------------
    for name, f, peak in [
        ("AR-1.5B", "vllm_serving_1.5b.json", 204.9),
        ("AR-3B", "vllm_serving_3b.json", 145.2),
    ]:
        m = single_model(f)
        t = None if m is None else m.get("throughput_table4", {})
        check(
            "sect1.4 %s peak events/s" % name,
            peak,
            None if t is None else t.get("peak_events_per_s"),
            f,
        )
    # spot-check two rows of the AR-1.5B burst sweep quoted in sect.1.4
    m = single_model("vllm_serving_1.5b.json")
    sweep = {} if m is None else {r["batch_size"]: r for r in m["throughput_table4"]["sweep"]}
    for b, p99, evs in [(1, 418.6, 2.4), (512, 2498.5, 204.9)]:
        row = sweep.get(b)
        check(
            "sect1.4 AR-1.5B B=%d p99 (ms)" % b,
            p99,
            None if row is None else row.get("p99_ms"),
            "vllm_serving_1.5b.json",
        )
        check(
            "sect1.4 AR-1.5B B=%d events/s" % b,
            evs,
            None if row is None else row.get("events_per_s"),
            "vllm_serving_1.5b.json",
        )

    # --- 1.3 superseded HF rows (Campaign D) -------------------------------
    hf = [
        ("AR-1.5B (v1)", "quality_table2_ar_n2000.json", "Qwen2.5-1.5B", 1841.6, 3053.4, 4162.5, 0.779),
        ("AR-7B (v1)", "quality_table2_ar_n2000.json", "Qwen2.5-7B", 2122.0, 5479.7, 7808.0, 0.839),
    ]
    for name, f, frag, p50, p99, p999, f1 in hf:
        row = model_row(f, frag)
        if row is None:
            fail("sect1.3 %s row" % name, "model %r not found in %s" % (frag, f))
            continue
        check("sect1.3 %s p50 (ms)" % name, p50, row.get("p50_ms"), f)
        check("sect1.3 %s p99 (ms)" % name, p99, row.get("p99_ms"), f)
        check("sect1.3 %s p99.9 (ms)" % name, p999, row.get("p999_ms"), f)
        check("sect1.3 %s Field-F1" % name, f1, row.get("field_f1"), f)

    # --- 1.3 diffusion reference row (Campaign D) --------------------------
    f = "quality_table2_mdm_llada_n2000.json"
    row = model_row(f, "LLaDA-8B")
    if row is None:
        fail("sect1.3 LLaDA-8B row", "model not found in %s" % f)
        return
    check("sect1.3 LLaDA-8B K=8 p50 (ms)", 1389.0, row.get("p50_ms"), f)
    check("sect1.3 LLaDA-8B K=8 p99 (ms)", 1555.5, row.get("p99_ms"), f)
    check("sect1.3 LLaDA-8B K=8 p99.9 (ms)", 1648.2, row.get("p999_ms"), f)
    check("sect1.3 LLaDA-8B K=8 Field-F1", 0.779, row.get("field_f1"), f)
    check("sect7.5 LLaDA-8B consistency (%)", 76.6, row.get("consistency_pct"), f)

    # --- derived quantities the paper states as headline numbers -----------
    ar15 = qual("vllm_serving_1.5b.json", "field_f1")
    ar3 = qual("vllm_serving_3b.json", "field_f1")
    ar7 = qual("vllm_serving_7b.json", "field_f1")
    mdm_f1 = row.get("field_f1")
    mdm_p50 = row.get("p50_ms")
    mdm_p999 = row.get("p999_ms")
    ar15_p50 = lat("vllm_serving_1.5b.json", "p50_ms")

    if None not in (ar15, mdm_f1):
        check(
            "sect1.3 derived parity gap (AR-1.5B - LLaDA-8B)",
            0.065,
            ar15 - mdm_f1,
            "vllm_serving_1.5b.json + " + f,
        )
    if None not in (ar15, ar3, ar7):
        check(
            "sect1.3 derived AR scale spread (max-min over 1.5B/3B/7B)",
            0.018,
            max(ar15, ar3, ar7) - min(ar15, ar3, ar7),
            "vllm_serving_{1.5b,3b,7b}.json",
        )
    if None not in (mdm_p50, ar15_p50):
        check(
            "sect1.3 derived AR-1.5B speedup over LLaDA-8B",
            3.36,
            mdm_p50 / ar15_p50,
            "vllm_serving_1.5b.json + " + f,
        )
    if None not in (mdm_p50, mdm_p999):
        check("sect1.3 derived LLaDA-8B tail ratio", 1.19, mdm_p999 / mdm_p50, f)


# ==========================================================================
# 2. Cost laws (NUMBERS.md sect. 2) -- Campaign B
# ==========================================================================

AR_L = {
    32: "p2_ar3b_L32.json",
    64: "p2_ar3b_L64.json",
    128: "p2_ar3b_L128.json",
    256: "p2_ar3b_L256.json",
    512: "p2_ar3b_L512.json",
    1024: "appD_ar3b_L1024.json",
    2048: "appD_ar3b_L2048.json",
}
DG_L = {
    32: "p2_dg_L32.json",
    64: "p2_dg_L64.json",
    128: "p2_dg_L128.json",
    256: "p2_dg_L256.json",
    512: "p2_dg_L512.json",
    1024: "appD_dg_L1024.json",
    2048: "appD_dg_L2048.json",
}


def check_cost_laws():
    section("NUMBERS.md sect.2 -- fitted cost laws and the L sweep (Campaign B)")

    A = "appD_analysis.json"
    check_path("sect2.1 T_AR intercept (ms)", 7.27, A, "ar_fit", "intercept_ms")
    check_path("sect2.1 T_AR slope (ms/token)", 6.29, A, "ar_fit", "slope_ms_per_token")
    check_path("sect2.1 T_AR residual sigma (ms)", 2.53, A, "ar_fit", "resid_sigma_ms")
    check_path("sect2.1 T_DG base (ms)", 418.24, A, "dg_fit", "base_ms")
    check_path("sect2.1 T_DG step per canvas (ms)", 113.43, A, "dg_fit", "step_per_canvas_ms")

    obj = load(A)
    if obj is not None:
        step = obj["dg_fit"]["step_per_canvas_ms"]
        slope = obj["ar_fit"]["slope_ms_per_token"]
        check("sect2.2 DG marginal cost (ms/token) = step/256", 0.443, step / 256.0, A)
        check("sect2.2 asymptotic ratio = AR slope / DG marginal", 14.2, slope / (step / 256.0), A)

    # --- 2.3 measured p50 per L -------------------------------------------
    ar_p50 = {32: 210.6, 64: 411.5, 128: 812.6, 256: 1617.6, 512: 3226.4,
              1024: 6443.3, 2048: 12892.7}
    dg_p50 = {32: 416.9, 64: 418.5, 128: 418.1, 256: 419.4, 512: 519.3,
              1024: 779.1, 2048: 1250.7}
    for L in sorted(ar_p50):
        check("sect2.3 T_AR p50 at L=%d (ms)" % L, ar_p50[L], lat(AR_L[L], "p50_ms"), AR_L[L])
    for L in sorted(dg_p50):
        check("sect2.3 T_DG p50 at L=%d (ms)" % L, dg_p50[L], lat(DG_L[L], "p50_ms"), DG_L[L])

    # --- 2.3 ratio column --------------------------------------------------
    ratios = {32: 0.50, 64: 0.98, 128: 1.94, 256: 3.87, 512: 6.07, 1024: 8.50, 2048: 10.63}
    for L in sorted(ratios):
        check_path("sect2.3 AR/DG ratio at L=%d" % L, ratios[L], A, "ratio_T_AR_over_T_DG", str(L))

    # The stored ratio column is the FITTED ratio, not the quotient of the two
    # measured p50 columns printed beside it in NUMBERS.md sect.2.3.
    if obj is not None:
        worst_L, worst_delta = None, 0.0
        for L in sorted(ratios):
            a, g = lat(AR_L[L], "p50_ms"), lat(DG_L[L], "p50_ms")
            if a is None or g is None:
                continue
            measured = a / g
            stored = obj["ratio_T_AR_over_T_DG"][str(L)]
            delta = abs(measured - stored) / stored
            if delta > worst_delta:
                worst_L, worst_delta = L, delta
        if worst_delta > REL_TOL:
            warn(
                "sect2.3 ratio column provenance",
                "stored ratios are the fitted-law ratios, not measured p50 quotients "
                "(worst: L=%s, measured %.2f vs stored %.2f, %.1f%% apart). "
                "NUMBERS.md sect.2.3 heads the table 'Measured latency and ratio'."
                % (
                    worst_L,
                    lat(AR_L[worst_L], "p50_ms") / lat(DG_L[worst_L], "p50_ms"),
                    obj["ratio_T_AR_over_T_DG"][str(worst_L)],
                    100 * worst_delta,
                ),
            )

    # --- 2.3 extrapolation and linearity checks ----------------------------
    check_path("sect2.3 P-D2 predicted DG at L=1024 (ms)", 719.0, A, "P_D2_check", "1024", "predicted_ms")
    check_path("sect2.3 P-D2 measured DG at L=1024 (ms)", 779.1, A, "P_D2_check", "1024", "measured_ms")
    check_path("sect2.3 P-D2 predicted DG at L=2048 (ms)", 1120.0, A, "P_D2_check", "2048", "predicted_ms")
    check_path("sect2.3 P-D2 measured DG at L=2048 (ms)", 1250.7, A, "P_D2_check", "2048", "measured_ms")
    check_path("sect2.3 P-D3 fitted AR slope", 6.29, A, "P_D3_check", "slope")
    check_path("sect2.3 P-D3 expected AR slope", 6.28, A, "P_D3_check", "expected")

    if obj is not None:
        for L, err in [(1024, 8.4), (2048, 11.7)]:
            cell = obj["P_D2_check"][str(L)]
            rel = 100.0 * (cell["measured_ms"] - cell["predicted_ms"]) / cell["predicted_ms"]
            check("sect2.3 P-D2 error at L=%d (%%)" % L, err, rel, A)

        # P_D2's predicted values are NOT generated by this file's own dg_fit.
        base = obj["dg_fit"]["base_ms"]
        step = obj["dg_fit"]["step_per_canvas_ms"]
        for L in (1024, 2048):
            from_fit = base + step * (math.ceil(L / 256.0) - 1)
            stored = obj["P_D2_check"][str(L)]["predicted_ms"]
            if not close(stored, from_fit):
                warn(
                    "sect2.3 P-D2 predicted at L=%d vs dg_fit" % L,
                    "P_D2 stores %.1f ms but this file's own dg_fit (base %.2f, step %.2f) "
                    "gives %.1f ms; P_D2 appears to use the superseded ~100 ms step. "
                    "The extrapolation check therefore does not validate the fit "
                    "NUMBERS.md sect.2.1 publishes." % (stored, base, step, from_fit),
                )


def check_fit_discrepancy():
    """NUMBERS.md sect.2.1 [DISCREPANCY]: appD_report.md disagrees with the JSON."""
    section("NUMBERS.md sect.2.1 -- known fit discrepancy (WARNING, not a failure)")
    obj = load("appD_analysis.json")
    path = os.path.join(RESULTS, "appD_report.md")
    try:
        with open(path, "r") as fh:
            text = fh.read()
    except Exception as exc:
        warn("sect2.1 appD_report.md", "could not be read: %s" % exc)
        return
    if obj is None:
        return
    json_slope = obj["ar_fit"]["slope_ms_per_token"]
    json_int = obj["ar_fit"]["intercept_ms"]
    json_sig = obj["ar_fit"]["resid_sigma_ms"]
    if "9.1 + 6.283" in text:
        warn(
            "sect2.1 T_AR fit: report vs JSON",
            "appD_report.md states T_AR = 9.1 + 6.283*L (sigma 0.39 ms); "
            "appD_analysis.json states intercept %.2f, slope %.2f, sigma %.2f. "
            "The JSON is authoritative per NUMBERS.md; re-run the fit from raw "
            "per-cell data and reconcile before submission (paper blocker)."
            % (json_int, json_slope, json_sig),
        )
    else:
        ok("sect2.1 appD_report.md no longer states the superseded T_AR fit")


# ==========================================================================
# 3. Quality cost-invariance (NUMBERS.md sect. 3)
# ==========================================================================


def check_quality_invariance():
    section("NUMBERS.md sect.3 -- quality cost-invariance across the L sweep")

    dg_f1 = {32: 0.3167, 64: 0.3167, 128: 0.3287, 256: 0.7871, 512: 0.7945,
             1024: 0.7887, 2048: 0.7902}
    ar_f1 = {32: 0.3167, 64: 0.3167, 128: 0.6544, 256: 0.8263, 512: 0.5991,
             1024: 0.4947, 2048: 0.4727}
    dg_valid = {32: 0.0, 64: 0.0, 128: 3.3, 256: 99.3, 512: 100.0,
                1024: 100.0, 2048: 100.0}
    ar_valid = {32: 0.0, 64: 0.0, 128: 66.7, 256: 99.3, 512: 99.3,
                1024: 99.7, 2048: 99.3}

    for L in sorted(dg_f1):
        check("sect3 DG Field-F1 at L=%d" % L, dg_f1[L], qual(DG_L[L], "field_f1"), DG_L[L])
        check("sect3 DG schema-valid %% at L=%d" % L, dg_valid[L],
              qual(DG_L[L], "schema_valid_pct"), DG_L[L])
    for L in sorted(ar_f1):
        check("sect3 AR Field-F1 at L=%d" % L, ar_f1[L], qual(AR_L[L], "field_f1"), AR_L[L])
        check("sect3 AR schema-valid %% at L=%d" % L, ar_valid[L],
              qual(AR_L[L], "schema_valid_pct"), AR_L[L])

    # monotonicity claims the paper makes in prose
    dg_over = [qual(DG_L[L], "field_f1") for L in (512, 1024, 2048)]
    ar_over = [qual(AR_L[L], "field_f1") for L in (512, 1024, 2048)]
    if all(v is not None for v in dg_over):
        spread = max(dg_over) - min(dg_over)
        if spread <= 0.01:
            ok("sect3 DG Field-F1 flat over L=512..2048 (spread %.4f <= 0.01)" % spread)
        else:
            fail("sect3 DG Field-F1 flat over L=512..2048", "spread %.4f > 0.01" % spread)
    if all(v is not None for v in ar_over):
        if ar_over[0] > ar_over[1] > ar_over[2]:
            ok("sect3 AR Field-F1 monotonically decreasing over L=512..2048")
        else:
            fail("sect3 AR Field-F1 monotonically decreasing over L=512..2048",
                 "sequence %s is not monotone" % ar_over)

    # per-field corruption at L=512
    fields = {
        "f1_host": (0.6733, 1.0000),
        "f1_component": (0.7633, 0.9900),
        "f1_severity": (0.6167, 0.9933),
        "f1_timestamp_norm": (0.6100, 0.8367),
        "f1_is_anomaly": (0.6733, 0.9933),
    }
    for field, (a, g) in fields.items():
        check("sect3 AR %s at L=512" % field, a, perfield(AR_L[512], field), AR_L[512])
        check("sect3 DG %s at L=512" % field, g, perfield(DG_L[512], field), DG_L[512])
    check("sect3 AR anomaly_f1 at L=512", 0.3553, qual(AR_L[512], "anomaly_f1"), AR_L[512])
    check("sect3 DG anomaly_f1 at L=512", 0.9846, qual(DG_L[512], "anomaly_f1"), DG_L[512])


# ==========================================================================
# 5. Batching multipliers (NUMBERS.md sect. 5) -- Campaigns B + C
# ==========================================================================


def check_batching():
    section("NUMBERS.md sect.5 -- batching multipliers and the dInfer control")

    A = "appD_analysis.json"
    rows = [
        ("dg_L128", 2.451, 4.110, 1.7),
        ("dg_L512", 1.872, 2.820, 1.5),
        ("ar3b_L128", 1.230, 94.90, 77.1),
        ("ar3b_L512", 0.310, 26.68, 86.0),
    ]
    for key, b1, peak, mult in rows:
        check_path("sect5 %s B=1 events/s" % key, b1, A, "batching_multipliers", key, "batch1_ev_s")
        check_path("sect5 %s peak events/s" % key, peak, A, "batching_multipliers", key, "peak_ev_s")
        check_path("sect5 %s multiplier" % key, mult, A, "batching_multipliers", key, "multiplier")

    obj = load(A)
    if obj is not None:
        bm = obj["batching_multipliers"]
        check(
            "sect5 derived batching-capability gap at L=128 (~45x)",
            45.0,
            bm["ar3b_L128"]["multiplier"] / bm["dg_L128"]["multiplier"],
            A,
            atol=1.0,  # NUMBERS.md sect.5 states this as "~45x"
        )

    # dInfer control -- the load-bearing evidence that the ~1x cap is architectural
    D = os.path.join("llada_dinfer", "batch_sweep.json")
    check_path("sect5 dInfer LLaDA-8B batch multiplier", 1.18, D, "batch_multiplier")
    check_path("sect5 dInfer LLaDA-8B peak batch", 4, D, "peak_batch")
    sweep = load(D)
    if sweep is not None:
        mults = [r["multiplier"] for r in sweep["results"]]
        batches = [r["batch"] for r in sweep["results"]]
        if max(batches) >= 64 and max(mults) <= 1.2:
            ok("sect5 dInfer multiplier flat to B=%d (max %.3f)" % (max(batches), max(mults)))
        else:
            fail("sect5 dInfer multiplier flat to B=64",
                 "max batch %s, max multiplier %.3f" % (max(batches), max(mults)))

    # "Also at B=1, DG leads: 2.12 vs 1.18 ev/s"
    P = "diffusiongemma_perf.json"
    m = single_model(P)
    if m is None:
        fail("sect5 DG B=1 events/s", "model block missing from %s" % P)
    else:
        sweep = {r["batch_size"] if "batch_size" in r else r.get("batch"): r
                 for r in m["throughput_table4"]["sweep"]}
        row = sweep.get(1)
        check("sect5 DG B=1 events/s", 2.12, None if row is None else row.get("events_per_s"), P)
        warn(
            "sect5 B=1 comparison provenance",
            "NUMBERS.md sect.5 cites appD_analysis.json for the '2.12 vs 1.18 ev/s' "
            "B=1 comparison, but those values live in %s (natural output length), "
            "not in appD_analysis.json (whose B=1 cells are 2.451 / 1.230 at L=128). "
            "Fix the citation before submission." % P,
        )

    # sect.8 dInfer quality/latency control
    C = os.path.join("llada_dinfer", "correctness_latency.json")
    check_path("sect8 dInfer LLaDA-8B Field-F1", 0.872, C, "field_f1")
    check_path("sect8 dInfer LLaDA-8B consistency (%)", 88.3, C, "consistency_pct")
    check_path("sect8 dInfer LLaDA-8B p50 (ms)", 10886, C, "p50_ms")


# ==========================================================================
# 6. Applications (NUMBERS.md sect. 6)
# ==========================================================================


def check_chains():
    section("NUMBERS.md sect.6.1 -- serial investigation chains")

    C = "chain_analysis.json"
    obj = load(C)
    if obj is None:
        return

    # slope_cv is stored as a fraction; NUMBERS.md quotes it in percent
    dg_cv = obj["P_E1_linearity"]["dg"]["slope_cv"] * 100.0
    ar_cv = obj["P_E1_linearity"]["ar"]["slope_cv"] * 100.0
    check("sect6.1 P-E1 DG slope CV (%)", 1.1, dg_cv, C)
    check("sect6.1 P-E1 AR slope CV (%)", 1.2, ar_cv, C)

    depths = [1, 3, 5, 10, 20]
    dg_hop = [418.6, 407.2, 410.0, 409.0, 416.9]
    ar_hop = [807.8, 823.7, 825.2, 833.7, 837.0]
    p999_ratio = [1.041, 1.872, 2.070, 1.894, 1.769]
    p50_ratio = [1.930, 2.023, 2.013, 2.038, 2.008]
    dg_f1 = [0.783, 0.4565, 0.4554, 0.4654, 0.4598]
    ar_f1 = [0.8258, 0.4953, 0.4962, 0.5054, 0.5009]

    got_dg = obj["P_E1_linearity"]["dg"]["cumul_p50_per_depth"]
    got_ar = obj["P_E1_linearity"]["ar"]["cumul_p50_per_depth"]
    for i, D in enumerate(depths):
        check("sect6.1 DG per-hop p50 at D=%d (ms)" % D, dg_hop[i], got_dg[i], C)
        check("sect6.1 AR per-hop p50 at D=%d (ms)" % D, ar_hop[i], got_ar[i], C)
        check("sect6.1 p99.9 ratio AR/DG at D=%d" % D, p999_ratio[i],
              obj["P_E2_tail_concentration"]["p999_ratio_per_depth"][i], C)
        check("sect6.1 p50 ratio AR/DG at D=%d" % D, p50_ratio[i],
              obj["P_E2_tail_concentration"]["p50_ratio_per_depth"][i], C)
        check("sect6.1 DG Field-F1 at D=%d" % D, dg_f1[i],
              obj["P_E4_quality"]["dg"]["f1_per_depth"][i], C)
        check("sect6.1 AR Field-F1 at D=%d" % D, ar_f1[i],
              obj["P_E4_quality"]["ar"]["f1_per_depth"][i], C)

    check_path("sect6.1 P-E2 single-hop p99.9 ratio", 1.041, C,
               "P_E2_tail_concentration", "single_hop_p999_ratio")
    check_path("sect6.1 P-E2 mean p50 ratio", 1.93, C,
               "P_E2_tail_concentration", "mean_ratio_p50")
    check_path("sect6.1 CLT DG sigma (ms)", 268.2, C, "analytical_CLT", "dg", "sigma_ms")
    check_path("sect6.1 CLT AR sigma (ms)", 158.7, C, "analytical_CLT", "ar", "sigma_ms")
    check("sect6.1 CLT DG cumulative p99.9 at D=20 (ms)", 12072,
          obj["analytical_CLT"]["dg"]["cumul_p999_predicted"][-1], C)
    check("sect6.1 CLT AR cumulative p99.9 at D=20 (ms)", 18935,
          obj["analytical_CLT"]["ar"]["cumul_p999_predicted"][-1], C)

    # P-E4 correction: quality is NOT stable across depths
    if obj["P_E4_quality"]["dg"]["unchanged"] or obj["P_E4_quality"]["ar"]["unchanged"]:
        fail("sect6.1 P-E4 quality NOT stable across depths",
             "artifact now reports unchanged=True, contradicting NUMBERS.md sect.6.1")
    else:
        ok("sect6.1 P-E4 quality NOT stable across depths (unchanged=False both models)")


def check_wide_schema():
    section("NUMBERS.md sect.6.2 -- wide-schema normalization")

    W = "wide_schema_analysis.json"
    obj = load(W)
    if obj is None:
        return

    levels = [12, 25, 50, 100]
    dg_p50 = [417.0, 589.3, 1036.6, 2142.2]
    ar_p50 = [737.6, 1440.2, 3121.8, 6239.1]
    ratio = [1.77, 2.44, 3.01, 2.91]
    dg_cons = [91.0, 92.5, 87.0, 87.5]
    ar_cons = [97.0, 94.0, 96.5, 96.5]
    dg_valid = [97.0, 98.0, 97.5, 99.0]

    for i, F in enumerate(levels):
        check("sect6.2 DG p50 at F=%d (ms)" % F, dg_p50[i], obj["P_C1_latency"]["dg"][i]["p50_ms"], W)
        check("sect6.2 AR p50 at F=%d (ms)" % F, ar_p50[i], obj["P_C1_latency"]["ar"][i]["p50_ms"], W)
        check("sect6.2 AR/DG ratio at F=%d" % F, ratio[i], obj["P_C1_latency"]["ratio_AR_over_DG"][i], W)
        check("sect6.2 DG consistency at F=%d (%%)" % F, dg_cons[i],
              obj["P_C2_consistency"]["dg"][i]["consistency_pct"], W)
        check("sect6.2 AR consistency at F=%d (%%)" % F, ar_cons[i],
              obj["P_C2_consistency"]["ar"][i]["consistency_pct"], W)
        check("sect6.2 DG validity at F=%d (%%)" % F, dg_valid[i],
              obj["P_C3_validity"]["dg"][i]["validity"], W)

    check("sect6.2 DG mean tokens at F=100", 1406.4, obj["P_C1_latency"]["dg"][3]["mean_tokens"], W)
    check("sect6.2 AR mean tokens at F=100", 950.7, obj["P_C1_latency"]["ar"][3]["mean_tokens"], W)

    if obj["P_C2_consistency"]["kill_condition_triggered"]:
        fail("sect6.2 kill condition not triggered",
             "artifact now reports kill_condition_triggered=True")
    else:
        ok("sect6.2 kill condition not triggered (DG consistency stays >= 80% at F=50)")

    # AR wins consistency at every width -- the caveat NUMBERS.md insists on
    losses = [i for i in range(len(levels))
              if obj["P_C2_consistency"]["ar"][i]["consistency_pct"]
              <= obj["P_C2_consistency"]["dg"][i]["consistency_pct"]]
    if losses:
        fail("sect6.2 AR consistency > DG at every width",
             "AR does not lead at F=%s" % [levels[i] for i in losses])
    else:
        ok("sect6.2 AR consistency > DG at every width")

    # sect.2.2 natural task output length
    check("sect2.2 natural task output (mean tokens, 12 fields)", 176.455,
          obj["P_C1_latency"]["dg"][0]["mean_tokens"], W)
    warn(
        "sect2.2 natural-output provenance",
        "NUMBERS.md sect.2.2 cites chain_analysis.json for the 12-field "
        "mean_tokens = 176.455; that key does not exist there. The value is in "
        "wide_schema_analysis.json (P_C1_latency/dg[0]/mean_tokens). "
        "Fix the citation before submission.",
    )


# ==========================================================================
# main
# ==========================================================================


def main():
    global VERBOSE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the artifact value on PASS lines too")
    args = ap.parse_args()
    VERBOSE = args.verbose

    if not os.path.isdir(RESULTS):
        print("FATAL: results/ not found at %s" % RESULTS)
        return 2

    print("verify_numbers.py -- checking paper/NUMBERS.md against results/*.json")
    print("repo root: %s" % ROOT)
    print("tolerance: %.1f%% relative, or half a unit in the last printed decimal"
          % (100 * REL_TOL))

    check_ar_baseline()
    check_cost_laws()
    check_fit_discrepancy()
    check_quality_invariance()
    check_batching()
    check_chains()
    check_wide_schema()

    section("SUMMARY")
    print("checked : %d" % (STATE["pass"] + STATE["fail"]))
    print("passed  : %d" % STATE["pass"])
    print("failed  : %d" % STATE["fail"])
    print("warnings: %d" % STATE["warn"])
    if STATE["skip"]:
        print("skipped : %d (unreadable artifacts)" % STATE["skip"])

    if WARNINGS:
        print("")
        print("WARNINGS (documented discrepancies -- do not block this script, "
              "but block submission):")
        for w in WARNINGS:
            print("  - %s" % w)

    if FAILURES:
        print("")
        print("FAILURES:")
        for f in FAILURES:
            print("  - %s" % f)
        print("")
        print("RESULT: FAIL -- NUMBERS.md no longer matches results/*.json.")
        print("Fix NUMBERS.md and paper/macros.tex together; never edit a "
              "number directly into a .tex file.")
        return 1

    print("")
    print("RESULT: PASS -- every checked value in NUMBERS.md matches its artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
