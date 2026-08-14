#!/usr/bin/env python3
"""
Issue #43 Step 2 — re-score saved predictions under three parser variants
to test whether the quality length-invariance result is a forced-min_tokens
+ last-wins-parser artifact.

Parser variants:
  (a) current   — rfind('}') last-wins, regex fallback last-occurrence-wins
  (b) first_obj — first balanced-brace object, regex fallback first-occurrence-wins
  (c) strict    — whole stripped output must parse as JSON

Also reports the empty-prediction floor (score of returning {}).

Usage:
    python models/experiments/rescore_parsers.py results/p2_ar3b_L512_preds.jsonl
    python models/experiments/rescore_parsers.py results/diffusiongemma_preds.jsonl --gold-n 2000

The script loads gold from datasets/splits/test/bgl.jsonl (first N records,
matching the order the predictions were generated).
"""

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES, FieldType, FIELD_BY_NAME
from metrics.quality_metrics import field_f1, schema_validity
from models.baselines.ar_wrapper import _dict_to_record

GOLD_PATH = _REPO_ROOT / "datasets" / "splits" / "test" / "bgl.jsonl"

_KV_PATTERN = re.compile(
    r'"(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)"\s*:\s*'
    r'(?:"(?P<str_val>(?:[^"\\]|\\.)*)"'
    r'|(?P<bool_val>true|false)'
    r'|(?P<num_val>-?\d+\.?\d*))'
)


# ─── Parser variants ─────────────────────────────────────────────────────

def parse_current(text: str) -> dict:
    """Current parser: last-wins (rfind), regex fallback last-occurrence-wins.
    Mirrors models/baselines/ar_model.py:parse_json_output exactly."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        block = text[start:end + 1]
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass
        result = {}
        for m in _KV_PATTERN.finditer(block):
            key = m.group("key")
            if m.group("str_val") is not None:
                result[key] = m.group("str_val")
            elif m.group("bool_val") is not None:
                result[key] = m.group("bool_val") == "true"
            elif m.group("num_val") is not None:
                try:
                    result[key] = float(m.group("num_val"))
                except ValueError:
                    pass
        return result
    return {}


def parse_first_obj(text: str) -> dict:
    """First balanced-brace object wins; regex fallback first-occurrence-wins."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first balanced { } object
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    block = text[start:i + 1]
                    try:
                        return json.loads(block)
                    except json.JSONDecodeError:
                        pass
                    # Regex fallback — first-occurrence-wins
                    result = {}
                    for m in _KV_PATTERN.finditer(block):
                        key = m.group("key")
                        if key in result:
                            continue  # first occurrence wins
                        if m.group("str_val") is not None:
                            result[key] = m.group("str_val")
                        elif m.group("bool_val") is not None:
                            result[key] = m.group("bool_val") == "true"
                        elif m.group("num_val") is not None:
                            try:
                                result[key] = float(m.group("num_val"))
                            except ValueError:
                                pass
                    return result
    return {}


def parse_strict(text: str) -> dict:
    """Strict: whole stripped output (after fence removal) must parse."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


PARSERS = {
    "current (last-wins)": parse_current,
    "first-object-wins": parse_first_obj,
    "strict (whole-output)": parse_strict,
}


# ─── Loading ──────────────────────────────────────────────────────────────

def load_gold(n):
    recs = []
    with open(GOLD_PATH) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                recs.append(Record(fields=json.loads(line)))
    return recs


def load_preds(path):
    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                preds.append(obj.get("raw", ""))
    return preds


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds_path", help="path to .jsonl with raw predictions")
    ap.add_argument("--gold-n", type=int, default=None,
                    help="number of gold records to load (default: len(preds))")
    args = ap.parse_args()

    preds_raw = load_preds(args.preds_path)
    n = args.gold_n or len(preds_raw)
    gold = load_gold(n)
    print(f"Loaded {len(preds_raw)} predictions, {len(gold)} gold records")
    if len(preds_raw) != len(gold):
        print(f"WARNING: count mismatch — using min={min(len(preds_raw), len(gold))}")
    n = min(len(preds_raw), len(gold))

    # Empty-prediction floor
    empty_preds = [Record(fields={}) for _ in range(n)]
    res_empty = field_f1(empty_preds, gold)
    print(f"\nEmpty-prediction floor: macro F1 = {res_empty['macro_field_f1']:.4f}")

    # Score each parser variant
    print(f"\n{'parser':25s}  {'macro F1':>10s}  {'validity':>10s}  {'diff vs current':>15s}")
    print("-" * 65)
    base_f1 = None
    for name, parser in PARSERS.items():
        records = [_dict_to_record(parser(raw)) for raw in preds_raw[:n]]
        res = field_f1(records, gold[:n])
        macro = res["macro_field_f1"]
        validity = schema_validity(records)
        diff = f"{macro - base_f1:+.4f}" if base_f1 is not None else "—"
        print(f"  {name:23s}  {macro:>10.4f}  {validity:>9.2f}%  {diff:>15s}")
        if base_f1 is None:
            base_f1 = macro

    # Per-field breakdown for current vs first-object
    print(f"\nPer-field breakdown (current vs first-object-wins):")
    print(f"  {'field':22s}  {'current':>8s}  {'first-obj':>10s}  {'delta':>8s}")
    print("  " + "-" * 52)
    recs_current = [_dict_to_record(parse_current(raw)) for raw in preds_raw[:n]]
    recs_first = [_dict_to_record(parse_first_obj(raw)) for raw in preds_raw[:n]]
    res_cur = field_f1(recs_current, gold[:n])
    res_first = field_f1(recs_first, gold[:n])
    for field_name in FIELD_NAMES:
        f_cur = res_cur[f"f1_{field_name}"]
        f_first = res_first[f"f1_{field_name}"]
        delta = f_first - f_cur
        flag = " ←" if abs(delta) > 0.01 else ""
        print(f"  {field_name:22s}  {f_cur:>8.4f}  {f_first:>10.4f}  {delta:>+8.4f}{flag}")

    # Count how many predictions have multiple JSON objects (ramble indicator)
    multi_obj = 0
    has_trailing = 0
    for raw in preds_raw[:n]:
        brace_opens = raw.count("{")
        brace_closes = raw.count("}")
        if brace_opens > 1:
            multi_obj += 1
        # Check if there's substantial text after the first }
        first_close = raw.find("}")
        if first_close != -1 and first_close < len(raw) - 10:
            trailing = raw[first_close + 1:].strip()
            if len(trailing) > 5:
                has_trailing += 1
    print(f"\nRamble indicators:")
    print(f"  predictions with >1 '{{'   : {multi_obj}/{n} ({100*multi_obj/n:.1f}%)")
    print(f"  predictions with trailing text after first '}}': {has_trailing}/{n} ({100*has_trailing/n:.1f}%)")


if __name__ == "__main__":
    main()
