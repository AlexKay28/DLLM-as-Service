#!/usr/bin/env python3
"""
Issue #52 — minimal Drain implementation to test whether the "drain-only"
results would change if real Drain were used instead of the regex fallback.

The regex fallback (drain_only.py:187-282) replaces all numbers/IDs with <*>
and takes the first 10 tokens. Real Drain clusters similar lines and merges
differing positions. This script implements Drain's core clustering algorithm
(minimal, no tree for efficiency) and compares template F1.

Drain algorithm (He et al. 2017):
  1. Tokenize log content
  2. Find matching cluster (highest token similarity, same length)
  3. If similarity ≥ threshold: update template (merge differing tokens to <*>)
  4. Else: create new cluster
"""

import json
import re
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES
from metrics.quality_metrics import field_f1
from models.baselines.ar_wrapper import _dict_to_record

GOLD_PATH = _REPO_ROOT / "datasets" / "splits" / "test" / "bgl.jsonl"
RAW_PATH = _REPO_ROOT / "datasets" / "splits" / "test" / "bgl.raw"
N = 2000


class MinimalDrain:
    def __init__(self, depth=4, sim_th=0.5):
        self.depth = depth
        self.sim_th = sim_th
        self.clusters = []  # list of {template: [tokens], count: int}

    def parse(self, tokens):
        if not tokens:
            return []
        # Only match clusters with same token count
        candidates = [c for c in self.clusters if len(c["template"]) == len(tokens)]
        best_match = None
        best_sim = 0.0
        for c in candidates:
            sim = sum(1 for a, b in zip(c["template"], tokens) if a == b) / len(tokens)
            if sim > best_sim:
                best_sim = sim
                best_match = c

        if best_match and best_sim >= self.sim_th:
            best_match["template"] = [
                a if a == b else "<*>" for a, b in zip(best_match["template"], tokens)
            ]
            best_match["count"] += 1
            return best_match["template"]
        else:
            new_c = {"template": list(tokens), "count": 1}
            self.clusters.append(new_c)
            return list(tokens)


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


def load_raw(n):
    lines = []
    with open(RAW_PATH) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            lines.append(line.strip())
    return lines


# Regex fallback template (mirrors drain_only.py:233-248)
NUM_RE = re.compile(r"^\d+\.?\d*$")
BLK_RE = re.compile(r"^blk_-?\d+$")
NODE_RE = re.compile(r"^(node|R\d|M\d|N\d|task|proc)\d+")
IP_RE = re.compile(r"^(/|)([0-9]+\.){3}[0-9]+")


def regex_fallback_template(content):
    tokens = content.split()
    template_tokens = []
    for token in tokens:
        if NUM_RE.match(token) or BLK_RE.match(token) or NODE_RE.match(token) or IP_RE.match(token):
            template_tokens.append("<*>")
        else:
            template_tokens.append(token)
    return template_tokens[:10]


def extract_content_bgl(line):
    """Extract content from BGL log line (same as drain_only.py:210-222)."""
    parts = line.split(None, 9)
    if len(parts) >= 9:
        return parts[9] if len(parts) > 9 else ""
    return line


def main():
    gold = load_gold(N)
    raw_lines = load_raw(N)
    print(f"Loaded {len(gold)} gold records, {len(raw_lines)} raw lines")

    # Extract content from raw lines
    contents = [extract_content_bgl(line) for line in raw_lines]

    # --- Regex fallback templates ---
    print("\nGenerating regex fallback templates...")
    regex_templates = [regex_fallback_template(c) for c in contents]
    n_regex_clusters = len(set(" ".join(t) for t in regex_templates))
    print(f"  {n_regex_clusters} unique templates (regex fallback)")

    # --- Drain templates ---
    print("Running minimal Drain (depth=4, sim_th=0.5)...")
    drain = MinimalDrain(depth=4, sim_th=0.5)
    t0 = time.perf_counter()
    drain_templates = []
    for content in contents:
        tokens = content.split()
        template = drain.parse(tokens)
        drain_templates.append(template)
    drain_time = time.perf_counter() - t0
    n_drain_clusters = len(drain.clusters)
    print(f"  {n_drain_clusters} clusters (Drain)")
    print(f"  Time: {drain_time:.2f}s for {N} events ({1000*drain_time/N:.3f} ms/event)")

    # --- Build records for each method ---
    # Both methods: structural fields from str.split (same as fallback),
    # only message_template differs
    SEVERITY_MAP = {"err": "ERROR", "error": "ERROR", "warn": "WARN",
                    "info": "INFO", "debug": "DEBUG", "fatal": "FATAL",
                    "crit": "CRITICAL", "critical": "CRITICAL"}
    ERR_RE = re.compile(r"(E\d+|err_\d+|ERR_\d+|blk_-?\d+|KERN_\d+|nfs\d+)")
    ENTITY_RE = re.compile(r"(blk_-?\d+|node\d+|R\d+-M\d+-N\d+|tbird-\d+|spirit-\d+|task\d+)")
    NUMERIC_RE = re.compile(r"\d+\.?\d*")

    def build_records(templates):
        records = []
        for line, template_tokens in zip(raw_lines, templates):
            parts = line.split(None, 9)
            rec = Record(fields={})
            if len(parts) >= 9:
                label = parts[0]
                rec.set("timestamp_norm", parts[1])
                rec.set("host", parts[3])
                rec.set("component", parts[7])
                sev = SEVERITY_MAP.get(parts[8].lower(), parts[8].upper())
                rec.set("severity", sev if sev in ("DEBUG","INFO","WARN","ERROR","FATAL","CRITICAL") else "INFO")
                content = parts[9] if len(parts) > 9 else ""
            else:
                content = line
                label = "-"

            rec.set("message_template", " ".join(template_tokens[:10]))
            err_m = ERR_RE.search(content)
            rec.set("error_code", err_m.group(1) if err_m else "")
            ent_m = ENTITY_RE.search(content)
            rec.set("affected_entity", ent_m.group(1) if ent_m else "")
            num_m = NUMERIC_RE.search(content)
            rec.set("metric_value", num_m.group(0) if num_m else "0.0")
            sev = rec.get("severity", "INFO")
            is_anom = sev in ("ERROR", "FATAL", "CRITICAL") or label != "-"
            rec.set("is_anomaly", "true" if is_anom else "false")
            rec.set("anomaly_score", "1.0" if is_anom else "0.0")
            rec.set("event_type", rec.get("message_template", "")[:30] or "unknown")
            rec.set("remediation_hint", "none")
            for name in FIELD_NAMES:
                if not rec.has(name):
                    rec.set(name, "")
            records.append(_dict_to_record(rec.fields))
        return records

    regex_recs = build_records(regex_templates)
    drain_recs = build_records(drain_templates)

    # --- Score ---
    print("\n" + "=" * 60)
    print("SCORING: regex fallback vs minimal Drain vs gold")
    print("=" * 60)

    res_regex = field_f1(regex_recs, gold)
    res_drain = field_f1(drain_recs, gold)

    print(f"\n  {'field':22s}  {'regex':>8s}  {'drain':>8s}  {'delta':>8s}")
    print("  " + "-" * 50)
    for name in FIELD_NAMES:
        fr = res_regex[f"f1_{name}"]
        fd = res_drain[f"f1_{name}"]
        delta = fd - fr
        flag = " ←" if abs(delta) > 0.01 else ""
        print(f"  {name:22s}  {fr:>8.4f}  {fd:>8.4f}  {delta:>+8.4f}{flag}")

    print(f"\n  {'macro F1':22s}  {res_regex['macro_field_f1']:>8.4f}  "
          f"{res_drain['macro_field_f1']:>8.4f}  "
          f"{res_drain['macro_field_f1']-res_regex['macro_field_f1']:>+8.4f}")

    # Template comparison
    print(f"\n  Template statistics:")
    print(f"    regex fallback: {n_regex_clusters} unique templates")
    print(f"    minimal Drain:  {n_drain_clusters} clusters")
    # Template F1 specifically
    print(f"    message_template F1:  regex={res_regex['f1_message_template']:.4f}  "
          f"drain={res_drain['f1_message_template']:.4f}")

    # Show some template examples
    print(f"\n  Sample Drain templates (top 5 by cluster size):")
    drain.clusters.sort(key=lambda c: -c["count"])
    for c in drain.clusters[:5]:
        print(f"    [{c['count']:4d}x]  {' '.join(c['template'][:15])}")

    print(f"\n  Sample regex templates (top 5 by frequency):")
    from collections import Counter
    regex_counts = Counter(" ".join(t) for t in regex_templates)
    for template, cnt in regex_counts.most_common(5):
        print(f"    [{cnt:4d}x]  {template[:80]}")


if __name__ == "__main__":
    main()
