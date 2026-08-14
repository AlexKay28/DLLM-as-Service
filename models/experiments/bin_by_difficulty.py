#!/usr/bin/env python3
"""
Issue #24: Difficulty binning by template frequency.

Reads the train split to count message_template frequencies, then
labels test events as routine / medium / novel based on how often
their template appeared in training.
"""

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = REPO / "datasets" / "splits" / "train" / "bgl.jsonl"
TEST_PATH = REPO / "datasets" / "splits" / "test" / "bgl.jsonl"
OUT_PATH = REPO / "results" / "difficulty_bins.json"

ROUTINE_THRESHOLD = 1000
NOVEL_THRESHOLD = 10


def count_train_templates():
    counts = Counter()
    with open(TRAIN_PATH) as f:
        for line in f:
            rec = json.loads(line)
            tpl = rec.get("message_template", "")
            counts[tpl] += 1
    return counts


def bin_event(template, train_counts):
    freq = train_counts.get(template, 0)
    if freq >= ROUTINE_THRESHOLD:
        return "routine"
    elif freq >= NOVEL_THRESHOLD:
        return "medium"
    else:
        return "novel"


def main():
    n = None
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    print("Counting train template frequencies...", flush=True)
    train_counts = count_train_templates()
    print(f"  {len(train_counts)} unique templates in train")

    print("Binning test events...", flush=True)
    bins = {"routine": [], "medium": [], "novel": []}
    with open(TEST_PATH) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            rec = json.loads(line)
            tpl = rec.get("message_template", "")
            b = bin_event(tpl, train_counts)
            bins[b].append(i)

    for b, indices in bins.items():
        print(f"  {b}: {len(indices)} ({100*len(indices)/sum(len(v) for v in bins.values()):.1f}%)")

    out = {
        "routine_indices": bins["routine"],
        "medium_indices": bins["medium"],
        "novel_indices": bins["novel"],
        "n_routine": len(bins["routine"]),
        "n_medium": len(bins["medium"]),
        "n_novel": len(bins["novel"]),
        "thresholds": {"routine": ROUTINE_THRESHOLD, "novel": NOVEL_THRESHOLD},
    }

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
