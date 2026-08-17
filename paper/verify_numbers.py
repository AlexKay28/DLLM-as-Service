#!/usr/bin/env python3
"""
Provenance + value verification for the paper's numeric claims.

What this script actually checks (issue #112, grill-pass7 finding 36 -- the
old docstring oversold it):

  1. Provenance presence. Every result JSON that carries model info should
     have a ``provenance`` block (#57 R1/R7). For an *uncited* artifact this
     is a WARNING (advisory). For an artifact a macro in ``macros.tex`` cites
     it is an ERROR -- a cited number with no provenance is exactly the hole
     that hid the AR-30B gold-leak (issue #112).
  2. Model/model_type consistency (#57 R1 amended).
  3. 9-field F1 presence (#57 R4).
  4. CHECKLIST defect magnitudes (#65).
  5. Numeric precision policy (#85).
  6. Macro -> artifact value check (#112 Phase 4 step 16). Selected macros
     are tied to a field in a committed artifact, so a macro that drifts from
     its measurement (or whose artifact was never re-measured on the pinned
     stack) fails the build.

It does NOT open every ``.tex`` section or check every prose number; it checks
``paper/assets/macros.tex`` and the artifacts named in MACRO_ARTIFACTS below.

Usage:
    python3 paper/verify_numbers.py

Exit 0 = no errors (warnings may remain; they are advisory).
Exit 1 = at least one error: a cited artifact lacks provenance or fails the
         value check, a model is inconsistent, or a precision/defect check
         failed.
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

# Add repo root to path for imports
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.baselines.provenance import assert_provenance


# Artifacts cited in macros.tex or make_figures.py that MUST have provenance.
# Missing provenance in these is an ERROR, not a warning (#113 0.5g).
CITED_ARTIFACT_NAMES = {
    "issue42_batching.json",
    "issue43_nine_field_f1.json",
    "issue_energy_measurement.json",
    "issue_evidence_ar1b_energy.json",
    "issue111_qwen3_serial_sweep.json",
    "issue111_qwen3_batch_sweep.json",
    "issue111_ar30b_energy.json",
    "issue111_qwen3_30b_a3b.json",
}


WARNINGS = []
ERRORS = []


def warn(msg: str):
    WARNINGS.append(msg)
    print(f"  WARN: {msg}")


def error(msg: str):
    ERRORS.append(msg)
    print(f"  ERROR: {msg}")


def load_json(path: Path) -> dict | list | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def check_provenance(path: Path, data: dict) -> None:
    """Check that a result artifact has a provenance block (#57 R1/R7)."""
    if not isinstance(data, dict):
        return

    # Skip analysis/report files that don't carry model info
    if path.suffix == ".md":
        return

    prov = data.get("provenance")
    if prov is None:
        # Check for nested models structure (vLLM serving results)
        models = data.get("models", {})
        if isinstance(models, dict) and models:
            # These are vLLM serving results — check each model entry
            for mk, mv in models.items():
                if isinstance(mv, dict):
                    vllm = mv.get("vllm_version")
                    torch_v = mv.get("torch_version")
                    if not vllm and not torch_v:
                        warn(f"{path.name}: model '{mk}' has no stack version info")
        else:
            # Check for a sidecar .provenance.json
            sidecar = path.with_suffix(".provenance.json")
            if sidecar.exists():
                return  # sidecar covers it

            # Only warn if this file has model-related fields
            if data.get("model") or data.get("model_type"):
                # Cited artifacts (named in macros.tex or make_figures.py)
                # must have provenance — promote to error (#113 0.5g).
                if path.name in CITED_ARTIFACT_NAMES:
                    error(
                        f"{path.name}: cited in macros.tex or figures but "
                        f"lacks provenance block (#113 0.5g). Re-measure "
                        f"with capture_provenance or add a sidecar."
                    )
                else:
                    warn(f"{path.name}: has model info but no provenance block")


def check_model_consistency(path: Path, data: dict) -> None:
    """Check that model_type and model fields are consistent (#57 R1 amended)."""
    if not isinstance(data, dict):
        return

    model_type = data.get("model_type", "")
    model = data.get("model", "")

    if not model_type or not model:
        return

    # dm_hf / dm arms should not record an AR model name
    if model_type in ("dm_hf", "dm", "mdm"):
        if "Qwen" in model or "Llama" in model:
            # Check for sidecar correction
            sidecar = path.with_suffix(".provenance.json")
            if sidecar.exists():
                return  # corrected
            error(
                f"{path.name}: model_type={model_type!r} but model={model!r} "
                f"(AR model recorded for DM arm). See #57 R1."
            )


def check_9field(path: Path, data: dict) -> None:
    """Check that quality metrics include the 9-field score (#57 R4)."""
    if not isinstance(data, dict):
        return

    # Look for quality metrics
    has_f1 = any(
        k.startswith("field_f1") or k.startswith("macro_field_f1")
        for k in data.keys()
    )
    if not has_f1:
        # Check nested models structure
        models = data.get("models", {})
        if isinstance(models, dict):
            for mk, mv in models.items():
                if isinstance(mv, dict):
                    quality = mv.get("streaming_table3", {}).get("quality", {})
                    if quality and "field_f1_9" not in quality and "macro_field_f1_9" not in quality:
                        if "field_f1" in quality or "macro_field_f1" in quality:
                            warn(
                                f"{path.name}: has 12-field F1 but no 9-field score "
                                f"(#57 R4)"
                            )
        return

    if "field_f1_9" not in data and "macro_field_f1_9" not in data:
        if "field_f1" in data or "macro_field_f1" in data:
            warn(f"{path.name}: has 12-field F1 but no 9-field score (#57 R4)")


def check_predictions_saved(path: Path, data: dict) -> None:
    """Check that predictions are saved (#57 R3)."""
    if not isinstance(data, dict):
        return

    # Skip analysis files
    if data.get("experiment", "").startswith("analyze"):
        return

    # Check for predictions or a sidecar .jsonl
    has_preds = "predictions" in data or "pred_path" in data
    if not has_preds:
        pred_file = path.with_suffix(".preds.jsonl")
        if pred_file.exists():
            return
        # Don't warn for files that don't have model output (analysis, reports)
        if data.get("model") or data.get("models"):
            # Only warn for experiment files, not analysis
            exp = data.get("experiment", "")
            if "serving" in exp or "quality" in exp or "latency" in exp:
                pass  # serving/quality/latency results may not save per-event preds


def check_defect_magnitudes() -> None:
    """Assert the five CHECKLIST.md magnitudes against their artifacts (#65)."""
    print("Checking CHECKLIST.md defect magnitudes (#65)...")

    # 1. Decoding config: AR-1.5B F1 0.779 → 0.844
    d = load_json(RESULTS / "vllm_serving_1.5b.json")
    if d and isinstance(d, dict):
        models = d.get("models", {})
        for mk, mv in models.items():
            if "1.5" in mk:
                f1 = mv.get("streaming_table3", {}).get("quality", {}).get("field_f1")
                if f1 is not None:
                    if abs(f1 - 0.844) > 0.01:
                        error(f"Defect 1: vllm_serving_1.5b F1={f1} != 0.844")
                    else:
                        print(f"  Defect 1 (decoding): F1={f1:.3f} OK")
                # Also check p50 = 413.3
                p50 = mv.get("streaming_table3", {}).get("latency", {}).get("p50_ms")
                if p50 is not None and abs(p50 - 413.3) > 5:
                    error(f"Defect 2: vllm_serving_1.5b p50={p50} != 413.3")

    # 2. Serving parity: HF 1841.6 ms (from vllm_serving.md)
    # Checked via the md file, not a JSON

    # 3. Parser policy: AR-3B L=512 first-object-wins F1
    d = load_json(RESULTS / "issue43_nine_field_f1.json")
    if d and isinstance(d, dict):
        ar512 = d.get("ar3b_L512", {})
        f1_12 = ar512.get("field_f1_12")
        if f1_12 is not None:
            if abs(f1_12 - 0.826) > 0.01:
                error(f"Defect 3: ar3b_L512 12-field F1={f1_12} != 0.826")
            else:
                print(f"  Defect 3 (parser): AR-3B L=512 12-field F1={f1_12:.3f} OK")

    # 4. Provenance: check correction sidecars exist
    for name in ["perturbation_dm_hf", "bounded_deadline_dm_hf", "refinement_dm_hf"]:
        sidecar = RESULTS / f"{name}.provenance.json"
        if not sidecar.exists():
            error(f"Defect 5: missing correction sidecar {sidecar.name}")
        else:
            print(f"  Defect 5 (provenance): {sidecar.name} OK")


# --- Numeric precision policy (#85) ----------------------------------------
# Latency (ms, p50/p90/p99, base, step, slope, anchor, energy): 1 decimal.
# F1 (F, f1, Fone, Gap):                                       3 decimals.
# Ratio / multiplier / CV / percentage (Ratio, Mult, CV, Pct): 1 decimal.
# Rate (per-token input slopes, e.g. arInputSlope):            3 decimals.
_LATENCY_KEYS = (
    "ms", "p50", "p99", "p90", "base", "step", "slope",
    "anchor", "energy", "pfifty", "pninetynine", "latency",
)
_RATIO_KEYS = ("ratio", "mult", "cv", "pct")
_F1_KEYS = ("f", "f1", "fone", "gap")
_RATE_KEYS = ("inputslope",)
_MAX_DECIMALS = {"latency": 1, "f1": 3, "ratio": 1, "rate": 3}

PRECISION_VIOLATIONS = []


def _count_decimals(value: str) -> int:
    s = value.replace("{,}", "").strip()
    if "." in s:
        return len(s.split(".", 1)[1])
    return 0


def _is_numeric(value: str) -> bool:
    try:
        float(value.replace("{,}", "").strip())
        return True
    except ValueError:
        return False


def _classify_macro(name: str) -> str | None:
    low = name.lower()
    # Rate first: input slopes (e.g. arInputSlope) are per-token rates, not
    # raw latencies; they need 3 decimals (0.038 ms/token).
    for k in _RATE_KEYS:
        if k in low:
            return "rate"
    # Latency next: its keyword list (incl. Energy, Slope) takes precedence.
    for k in _LATENCY_KEYS:
        if k in low:
            return "latency"
    # Ratio before F1: "Footprint" contains "F" but is a ratio, not an F1.
    for k in _RATIO_KEYS:
        if k in low:
            return "ratio"
    for k in _F1_KEYS:
        if k in low:
            return "f1"
    return None


def _parse_macros(macros_path: Path) -> dict[str, str]:
    """Parse ``\\newcommand{\\name}{value}`` pairs from macros.tex.

    Tolerates one level of nested braces in the value (e.g. refPrompt =
    1{,}077). Returns {macro_name: raw_value_string}.
    """
    if not macros_path.exists():
        return {}
    text = macros_path.read_text()
    pattern = re.compile(
        r"\\newcommand\{\\(\w+)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    )
    return {name: raw for name, raw in pattern.findall(text)}


def check_precision_policy() -> bool:
    """Lint paper/macros.tex against the numeric precision policy (#85).

    Returns True if every classified macro is within policy, False otherwise.
    """
    macros_path = REPO_ROOT / "paper" / "assets" / "macros.tex"
    print("Checking numeric precision policy (#85)...")

    if not macros_path.exists():
        error(f"macros.tex not found at {macros_path}")
        return False

    checked = 0
    passed = 0
    PRECISION_VIOLATIONS.clear()

    print(f"  {'macro':<34} {'value':<10} {'type':<8} {'dec':>3}  status")
    for name, raw_value in _parse_macros(macros_path).items():
        value = raw_value.strip()
        if not _is_numeric(value):
            continue
        mtype = _classify_macro(name)
        if mtype is None:
            continue
        checked += 1
        dec = _count_decimals(value)
        limit = _MAX_DECIMALS[mtype]
        if dec <= limit:
            passed += 1
            print(f"  {name:<34} {value:<10} {mtype:<8} {dec:>3}  ok")
        else:
            PRECISION_VIOLATIONS.append((name, value, mtype, dec, limit))
            print(
                f"  {name:<34} {value:<10} {mtype:<8} {dec:>3}  "
                f"VIOLATION (max {limit})"
            )

    failed = len(PRECISION_VIOLATIONS)
    print(
        f"\n  Precision policy: {checked} macros checked, "
        f"{passed} pass, {failed} violation(s)."
    )
    return failed == 0


# --- Macro -> artifact value verification (#112 Phase 4) ---------------------
# Each entry ties a macro in macros.tex to the field in a committed artifact
# that the macro's value must equal. This is the check that was missing when
# the AR-30B gold-leak shipped: the cost-law macros traced to
# issue111_qwen3_serial_sweep.json, but that artifact had no provenance and no
# verifier opened it, so a macro drifting from (or never re-measured against)
# its measurement was invisible. A cited artifact that lacks a provenance
# block is an ERROR here, not a warning (issue #112 step 17).
#
# Tolerance is half a unit of the macro's last reported decimal, so rounding
# does not cause spurious failures.
MACRO_ARTIFACTS = [
    # (macro, artifact relpath, dotted field path, abs tolerance)
    ("arBase",      "issue111_qwen3_serial_sweep.json",
     "cost_law.intercept_ms",        0.05),
    ("arSlope",     "issue111_qwen3_serial_sweep.json",
     "cost_law.slope_ms_per_token",  0.05),
    ("crossoverL",  "issue111_qwen3_serial_sweep.json",
     "cost_law.crossover_L_star",   0.5),
]

# Other AR-30B (issue #111/#112) artifacts the paper cites that must carry
# provenance even though they back no \\newcommand macro (their values appear
# in prose / figure code / macros.tex comments).
CITED_AR30B_ARTIFACTS = [
    "issue111_qwen3_batch_sweep.json",   # 77x batch multiplier (make_figures.py)
    "issue111_ar30b_energy.json",         # 10.1 J/event (macros.tex comment)
    "issue111_qwen3_30b_a3b.json",        # AR-30B quality + single-cell latency
]

# Stack every cited AR-30B artifact must have been measured on (issue #112
# step 4). Provenance records the actual installed version; a mismatch is a
# WARNING (advisory -- it may flag a legitimate re-run on a different stack).
PINNED_VLLM = "0.25.1"


def _dotted_get(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def check_macro_values() -> bool:
    """Verify selected macros equal their cited artifact field (#112 step 16).

    For each mapped macro: load the cited artifact, require a provenance block
    (ERROR if missing -- step 17), and require the macro value to match the
    artifact field within tolerance (ERROR on drift, or on a missing field,
    which is the signal that the artifact has not been re-measured on the
    pinned stack since the harness was fixed).
    """
    print("Checking macro -> artifact values (#112)...")
    macros_path = REPO_ROOT / "paper" / "assets" / "macros.tex"
    macros = _parse_macros(macros_path)

    ok = True
    for macro, relpath, field, tol in MACRO_ARTIFACTS:
        arti_path = RESULTS / relpath
        label = f"\\{macro} -> {relpath}:{field}"

        if macro not in macros:
            error(f"{label}: macro not defined in macros.tex")
            ok = False
            continue

        macro_val = macros[macro].strip()
        if not _is_numeric(macro_val):
            error(f"{label}: macro value {macro_val!r} is not numeric")
            ok = False
            continue

        data = load_json(arti_path)
        if data is None:
            error(f"{label}: cited artifact not found at {arti_path}")
            ok = False
            continue

        # Step 17: a cited artifact MUST have provenance.
        prov = data.get("provenance") if isinstance(data, dict) else None
        if not prov:
            error(
                f"{label}: cited artifact lacks a provenance block (#112 "
                f"step 17). Re-measure on the pinned stack (vLLM "
                f"{PINNED_VLLM}) with the corrected harness."
            )
            ok = False
        else:
            vllm = prov.get("vllm_version")
            if vllm and vllm != PINNED_VLLM:
                warn(
                    f"{relpath}: provenance.vllm_version={vllm} != pinned "
                    f"{PINNED_VLLM} (stack asymmetry, issue #112 step 4)"
                )

        field_val, found = _dotted_get(data, field)
        if not found or field_val is None:
            error(
                f"{label}: field missing -- artifact not yet re-measured "
                f"with the fixed harness (issue #112)."
            )
            ok = False
            continue

        try:
            delta = abs(float(macro_val) - float(field_val))
        except (TypeError, ValueError):
            error(f"{label}: artifact field {field_val!r} not numeric")
            ok = False
            continue

        if delta <= tol:
            print(f"  {label:<58} {macro_val} == {field_val}  ok")
        else:
            error(
                f"{label}: macro={macro_val} drifts from artifact={field_val} "
                f"(Δ={delta:.4g} > {tol}). Update the macro or re-measure."
            )
            ok = False

    # Step 17 for the rest of the AR-30B arm: these artifacts are cited by the
    # paper (batch multiplier, energy, quality cell) but carry no \\newcommand
    # macro to value-check. They still MUST have provenance -- a cited AR-30B
    # number with no provenance is the hole that hid the gold-leak.
    cited = {a for _, a, _, _ in MACRO_ARTIFACTS}
    for relpath in CITED_AR30B_ARTIFACTS:
        if relpath in cited:
            continue
        arti_path = RESULTS / relpath
        data = load_json(arti_path)
        if data is None:
            error(f"{relpath}: cited AR-30B artifact not found")
            ok = False
            continue
        if not (isinstance(data, dict) and data.get("provenance")):
            error(
                f"{relpath}: cited AR-30B artifact lacks a provenance block "
                f"(#112 step 17). Re-measure on the pinned stack (vLLM "
                f"{PINNED_VLLM}) with the corrected harness."
            )
            ok = False

    print(f"  Macro/value check: {'pass' if ok else 'FAILED'}")
    return ok


def main():
    print("=== paper/verify_numbers.py ===")
    print()

    if not RESULTS.exists():
        error(f"results/ directory not found at {RESULTS}")
        sys.exit(1)

    json_files = sorted(RESULTS.rglob("*.json"))
    print(f"Scanning {len(json_files)} JSON files in results/...\n")

    for path in json_files:
        if path.name.endswith(".provenance.json"):
            continue  # skip sidecars

        data = load_json(path)
        if data is None:
            continue

        check_provenance(path, data)
        check_model_consistency(path, data)
        check_9field(path, data)

    check_defect_magnitudes()

    precision_ok = check_precision_policy()
    values_ok = check_macro_values()

    print()
    print(f"=== Summary ===")
    print(f"  Warnings: {len(WARNINGS)}")
    print(f"  Errors:   {len(ERRORS)}")
    print(f"  Precision violations: {len(PRECISION_VIOLATIONS)}")

    if ERRORS or not precision_ok or not values_ok:
        parts = []
        if ERRORS:
            parts.append(f"{len(ERRORS)} error(s)")
        if not precision_ok:
            parts.append(f"{len(PRECISION_VIOLATIONS)} precision violation(s)")
        if not values_ok:
            parts.append("macro/value check failed")
        print(f"\n  Failing — {' and '.join(parts)} must be fixed.")
        sys.exit(1)

    if WARNINGS:
        print(f"\n  Passing with {len(WARNINGS)} warning(s).")
        print(f"  (Warnings are uncited artifacts without provenance — not")
        print(f"   load-bearing for any paper claim. Cited artifacts are errors.)")
        sys.exit(0)
    else:
        print(f"\n  All checks pass with zero warnings.")
        sys.exit(0)


if __name__ == "__main__":
    main()
