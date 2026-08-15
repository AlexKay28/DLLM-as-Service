#!/usr/bin/env python3
"""
Audit all results/*.json for:

  1. Missing provenance — files without a top-level "provenance" key.
  2. model/model_type inconsistency — the issue #57 bug where model_type is
     "dm_hf"/"dm" but the recorded "model" is an AR model (Qwen) because
     run_perturbation.py serialized args.model instead of the model actually
     loaded (GSAI-ML/LLaDA-8B-Instruct).
  3. Missing 9-field F1 — files that carry quality metrics but no
     "field_f1_9" / "macro_field_f1_9" key (per #57 R4, the 9-field score).

Files with a nested "models" dict/list (vLLM serving results, quality_table2)
store model info differently and are handled explicitly.

Usage:
    python3 models/experiments/audit_artifacts.py
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# --- family classification -------------------------------------------------
# AR = autoregressive (Qwen family); DM = diffusion / masked-diffusion
# (LLaDA, DiffusionGemma).
AR_MODEL_MARKERS = ("qwen",)
DM_MODEL_MARKERS = ("llada", "diffusiongemma")

AR_TYPE_MARKERS = ("ar3b", "ar7b", "ar1b", "ar", "AR")
DM_TYPE_MARKERS = ("dm_hf", "dm", "DM", "dinfer", "dg", "mdm")

# For the #57 correction: dm_hf runners load the HF MDM (LLaDA) model.
DM_HF_CORRECT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"

NINE_KEYS = ("field_f1_9", "macro_field_f1_9")
QUALITY_KEYS = ("field_f1", "per_field_f1")


def rec_has(obj, keys):
    """True if any of *keys* appears anywhere (recursively) in obj."""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return True
        for v in obj.values():
            if rec_has(v, keys):
                return True
    elif isinstance(obj, list):
        for v in obj:
            if rec_has(v, keys):
                return True
    return False


def model_family(model):
    if not model:
        return None
    m = str(model).lower()
    if any(x in m for x in AR_MODEL_MARKERS):
        return "AR"
    if any(x in m for x in DM_MODEL_MARKERS):
        return "DM"
    return None


def type_family(mt):
    if mt is None:
        return None
    s = str(mt)
    if s in AR_TYPE_MARKERS or s.lower().startswith("ar"):
        return "AR"
    if s in DM_TYPE_MARKERS or s.lower() in ("dm_hf", "dm"):
        return "DM"
    return None


def consistency(mt, model):
    """ok / mismatch / n/a based on AR-vs-DM family of type and model."""
    mf = model_family(model)
    tf = type_family(mt)
    if mf is None or tf is None:
        return "n/a"
    return "mismatch" if mf != tf else "ok"


def extract_nested_models(data):
    """Return list of model-name strings from a nested 'models' dict/list."""
    models = data.get("models")
    names = []
    if isinstance(models, dict):
        for k, v in models.items():
            if isinstance(v, dict) and v.get("model"):
                names.append(str(v["model"]))
            else:
                names.append(str(k))
    elif isinstance(models, list):
        for v in models:
            if isinstance(v, dict) and v.get("model"):
                names.append(str(v["model"]))
    return names


def audit_file(path):
    rec = {"file": str(path.relative_to(RESULTS_DIR))}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        rec.update(parse_error=True, error=str(e), has_provenance="?",
                   model_type="?", model="?", consistency="?", has_9field="?",
                   nested=False)
        return rec
    if not isinstance(data, dict):
        rec.update(parse_error=True, error="top-level not a JSON object",
                   has_provenance="?", model_type="?", model="?",
                   consistency="?", has_9field="?", nested=False)
        return rec

    rec["parse_error"] = False
    rec["has_provenance"] = "yes" if "provenance" in data else "no"

    nested = isinstance(data.get("models"), (dict, list))
    rec["nested"] = nested

    if nested:
        rec["model_type"] = "(nested)"
        names = extract_nested_models(data)
        rec["model"] = "; ".join(names) if names else "(none)"
        # No top-level model_type to mismatch against; mark per-model below.
        fams = {model_family(n) for n in names}
        fams.discard(None)
        if len(fams) > 1:
            rec["consistency"] = "mismatch"   # mixes AR + DM
        else:
            rec["consistency"] = "n/a"
    else:
        rec["model_type"] = data.get("model_type")
        rec["model"] = data.get("model")
        rec["consistency"] = consistency(data.get("model_type"),
                                         data.get("model"))

    has9 = rec_has(data, NINE_KEYS)
    hasq = rec_has(data, QUALITY_KEYS)
    if has9:
        rec["has_9field"] = "yes"
    elif hasq:
        rec["has_9field"] = "no"
    else:
        rec["has_9field"] = "n/a"
    return rec


def trunc(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: max(0, n - 1)] + "\u2026"


def main():
    if not RESULTS_DIR.is_dir():
        print(f"ERROR: results dir not found: {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    files = sorted(RESULTS_DIR.rglob("*.json"))
    records = [audit_file(p) for p in files]

    # ---- per-file table --------------------------------------------------
    cols = [
        ("file", 40), ("prov", 4), ("model_type", 10), ("model", 40),
        ("consist", 8), ("9field", 6),
    ]
    header = "  ".join(name.ljust(w) for name, w in cols)
    print("=" * len(header))
    print("ARTIFACT AUDIT — results/*.json")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in records:
        if r["parse_error"]:
            print(f"{trunc(r['file'],40):<40}  PARSE_ERROR: {r['error']}")
            continue
        print("  ".join([
            trunc(r["file"], 40).ljust(40),
            r["has_provenance"].ljust(4),
            trunc(r["model_type"], 10).ljust(10),
            trunc(r["model"], 40).ljust(40),
            r["consistency"].ljust(8),
            r["has_9field"].ljust(6),
        ]))

    # ---- summary ---------------------------------------------------------
    n = len(records)
    ok = [r for r in records if not r["parse_error"]]
    n_missing_prov = sum(1 for r in ok if r["has_provenance"] == "no")
    n_has_prov = sum(1 for r in ok if r["has_provenance"] == "yes")
    n_nested = sum(1 for r in ok if r["nested"])
    n_mismatch = [r for r in ok if r["consistency"] == "mismatch"]
    n_no9 = sum(1 for r in ok if r["has_9field"] == "no")
    n_yes9 = sum(1 for r in ok if r["has_9field"] == "yes")
    n_na9 = sum(1 for r in ok if r["has_9field"] == "n/a")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  total json files scanned : {n}")
    print(f"  parse errors             : {n - len(ok)}")
    print(f"  has provenance           : {n_has_prov}")
    print(f"  MISSING provenance       : {n_missing_prov}")
    print(f"  nested-models files      : {n_nested}")
    print(f"  model_type/model mismatch: {len(n_mismatch)}")
    print(f"  has 9-field F1 (yes)     : {n_yes9}")
    print(f"  MISSING 9-field F1 (no)  : {n_no9}")
    print(f"  no quality metrics (n/a) : {n_na9}")

    print()
    print("-" * 60)
    print(f"MODEL/MODEL_TYPE MISMATCHES (issue #57 bug): {len(n_mismatch)}")
    print("-" * 60)
    if not n_mismatch:
        print("  (none)")
    for r in n_mismatch:
        tf = type_family(r["model_type"])
        correct = DM_HF_CORRECT_MODEL if tf == "DM" else "(unknown)"
        print(f"  {r['file']}")
        print(f"      model_type = {r['model_type']!r} ({tf})")
        print(f"      recorded   = {r['model']!r} (AR)")
        print(f"      correct    = {correct!r}")

    # ---- files needing *.provenance.json correction files ----------------
    print()
    print("=" * 60)
    print("FILES NEEDING *.provenance.json CORRECTION FILES")
    print("=" * 60)
    needs = [r for r in ok if r["has_provenance"] == "no"]
    critical = [r for r in needs if r["consistency"] == "mismatch"]
    model_bearing = [r for r in needs
                     if r["consistency"] != "mismatch"
                     and r["model"] not in (None, "", "(none)")]
    rest = [r for r in needs
            if r not in critical and r not in model_bearing]

    print(f"\n  [CRITICAL] model mismatch — correction must state correct model "
          f"({len(critical)})")
    for r in critical:
        tf = type_family(r["model_type"])
        correct = DM_HF_CORRECT_MODEL if tf == "DM" else "(unknown)"
        base = r["file"][:-5] if r["file"].endswith(".json") else r["file"]
        print(f"    {base}.provenance.json")
        print(f"        model_type={r['model_type']!r} recorded={r['model']!r} "
              f"-> correct={correct!r}")

    print(f"\n  [HIGH] model-bearing artifact, missing provenance "
          f"({len(model_bearing)})")
    for r in model_bearing:
        base = r["file"][:-5] if r["file"].endswith(".json") else r["file"]
        mt = r["model_type"] if r["model_type"] is not None else "(none)"
        print(f"    {base}.provenance.json   "
              f"[model_type={mt}, model={trunc(r['model'],30)}]")

    print(f"\n  [LOW] aggregate/analysis, no model field ({len(rest)})")
    for r in rest:
        base = r["file"][:-5] if r["file"].endswith(".json") else r["file"]
        print(f"    {base}.provenance.json")

    total_corr = len(needs)
    print(f"\n  TOTAL correction files needed: {total_corr} / {n} files "
          f"({n_has_prov} already have provenance)")

    if n_missing_prov == n:
        print("\n  NOTE: 0/{n} result files carry a provenance block — the "
              "entire results/ tree predates the provenance helper (#57 R1).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
