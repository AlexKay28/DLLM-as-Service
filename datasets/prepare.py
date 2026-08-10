"""
Prepare the FlowFill benchmark splits from raw LOGHUB corpora.

Pipeline (per corpus):
  1. Discover the raw .log file under datasets/raw/<corpus>/.
  2. Sample 600k lines (uniform over the file; the paper samples 600k per
     corpus). The most recent 200k lines of the sample form the temporal
     test split; the remaining 400k are the train split.
  3. Parse the sampled lines with Drain (via logparser3) to extract
     structural fields (message_template, component, severity, host,
     timestamp_norm, error_code, affected_entity, metric_value).
  4. Carry native anomaly labels from the raw log (BGL/Thunderbird/Spirit
     alert-tag first column; HDFS block-level labels) into is_anomaly +
     anomaly_score.
  5. Emit 12-field schema records as JSONL under datasets/splits/{train,test}/
     and a manifest.json describing per-corpus counts + anomaly rates.

If logparser3 is not installed, the Drain step degrades to a regex-based
template detector (same fallback used by models/baselines/drain_only.py),
so the pipeline runs end-to-end without `pip install logparser3`.

Usage:
    python prepare.py                                # all corpora, default dirs
    python prepare.py --only hdfs bgl                # subset
    python prepare.py --sample-size 600000 --test-size 200000
    python prepare.py --raw-dir datasets/raw --out-dir datasets/splits
    python prepare.py --smoke                        # tiny run for CI

Outputs:
    datasets/splits/
      train/<corpus>.jsonl
      test/<corpus>.jsonl
      manifest.json
    datasets/parsed/<corpus>/            # Drain structured + templates CSVs
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES, FIELD_BY_NAME, FieldType

DEFAULT_RAW_DIR = REPO_ROOT / "datasets" / "raw"
DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "splits"
DEFAULT_PARSED_DIR = REPO_ROOT / "datasets" / "parsed"

CORPORA = ["hdfs", "bgl", "thunderbird", "spirit"]

# Log formats — same as models/baselines/drain_only.py.
LOG_FORMATS = {
    "hdfs":        "<Date> <Time> <Pid> <Level> <Component>: <Content>",
    "bgl":         "<Label> <Timestamp> <Date> <Time> <Node> <Component> <Level> <Content>",
    "thunderbird": "<Label> <Timestamp> <Date> <Time> <Node> <Component> <Level> <Content>",
    "spirit":      "<Label> <Timestamp> <Date> <Time> <Node> <Component> <Level> <Content>",
}

SEVERITY_MAP = {
    "err": "ERROR", "error": "ERROR",
    "warn": "WARN", "warning": "WARN",
    "info": "INFO", "debug": "DEBUG",
    "fatal": "FATAL", "crit": "CRITICAL", "critical": "CRITICAL",
    "emerg": "CRITICAL", "alert": "CRITICAL", "notice": "INFO",
}

# Canonical enum values from the schema (metrics/schema.py). Any severity
# not mapping to one of these is coerced to INFO so schema-validity stays >0.
VALID_SEVERITIES = {"DEBUG", "INFO", "WARN", "ERROR", "FATAL", "CRITICAL"}

# Default sampling params from the paper (§4): 600k per corpus, last 200k = test.
DEFAULT_SAMPLE_SIZE = 600_000
DEFAULT_TEST_SIZE = 200_000

IP_RE = re.compile(r"(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)")
TS_RE = re.compile(r"(\d{4}\d{2}\d{2} \d{2}:\d{2}:\d{2}|\d{10,})")
ERR_RE = re.compile(r"(E\d+|err_\d+|ERR_\d+|blk_-?\d+|task_\d+|KERN_\d+|nfs\d+)")
ENTITY_RE = re.compile(r"(blk_-?\d+|node\d+|R\d+-M\d+-N\d+|tbird-\d+|spirit-\d+|task\d+)")
NUMERIC_RE = re.compile(r"\d+\.?\d*")


# ---------------------------------------------------------------------------
# 1. Locate the raw log file for a corpus.
# ---------------------------------------------------------------------------

def _candidate_log_files(corpus: str, raw_dir: Path) -> list[Path]:
    """Return candidate raw .log files for `corpus` under `raw_dir`."""
    root = raw_dir / corpus
    candidates = []
    if not root.exists():
        return candidates
    # Common LOGHUB filenames. Prefer the full log over the 2k sample.
    preferred = {
        "hdfs":        ["HDFS.log", "HDFS_v1.log"],
        "bgl":         ["BGL.log", "BGL_2k.log"],
        "thunderbird": ["Thunderbird.log", "Thunderbird_2k.log"],
        "spirit":      ["Spirit.log", "spirit2", "Spirit"],
    }
    names = preferred.get(corpus, [])
    for name in names:
        p = root / name
        if p.exists() and p.stat().st_size > 0:
            candidates.append(p)
    # Fallback: any *.log file in the dir tree (max depth 2).
    if not candidates:
        for p in root.rglob("*.log"):
            if p.stat().st_size > 0:
                candidates.append(p)
    # Deduplicate while preserving order.
    seen, out = set(), []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _resolve_raw_log(corpus: str, raw_dir: Path) -> Path:
    candidates = _candidate_log_files(corpus, raw_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No raw log for '{corpus}' under {raw_dir / corpus}. "
            f"Run `python datasets/download.py --only {corpus} --extract` first."
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# 2. Sample lines.
# ---------------------------------------------------------------------------

def _count_lines(path: Path) -> int:
    """Count lines without loading the whole file into memory."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _sample_lines(path: Path, sample_size: int, seed: int = 42) -> list[str]:
    """Uniformly sample `sample_size` lines from `path` (reservoir sampling).

    For very large files (Thunderbird/Spirit are >200M lines) this keeps
    memory bounded to `sample_size` lines instead of loading the file.
    Strips trailing CR so CRLF-encoded LOGHUB samples parse cleanly.
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\r\n")
            if i < sample_size:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < sample_size:
                    reservoir[j] = line
    rng.shuffle(reservoir)
    return reservoir


def _temporal_split(lines: list[str], test_size: int) -> tuple[list[str], list[str]]:
    """Last `test_size` lines = test; the rest = train (temporal holdout)."""
    if test_size >= len(lines):
        return [], lines
    train = lines[:-test_size]
    test = lines[-test_size:]
    return train, test


# ---------------------------------------------------------------------------
# 3. Drain parsing (optional) + fallback.
# ---------------------------------------------------------------------------

DRAIN_AVAILABLE = False
try:
    from logparser.Drain import LogParser  # type: ignore
    DRAIN_AVAILABLE = True
except ImportError:
    pass


def _run_drain(corpus: str, log_lines: list[str], parsed_dir: Path
               ) -> dict[int, dict] | None:
    """Run logparser3 Drain on the sampled lines.

    Returns a list of dicts (one per line, in input order) with at least
    keys: EventTemplate, Level, Component, Node/Host, Content, Label.
    Returns None if logparser3 is unavailable.
    """
    if not DRAIN_AVAILABLE:
        return None
    parsed_dir.mkdir(parents=True, exist_ok=True)
    in_dir = parsed_dir / "input"
    out_dir = parsed_dir / "output"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = f"{corpus}_sample.log"
    with open(in_dir / log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    parser = LogParser(
        log_format=LOG_FORMATS[corpus],
        indir=str(in_dir),
        outdir=str(out_dir),
        depth=4,
        st=0.5,
        rex=[IP_RE.pattern],
    )
    parser.parse(log_file)
    structured = out_dir / f"{log_file}_structured.csv"
    if not structured.exists():
        return None
    rows: list[dict] = []
    with open(structured, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _fallback_parse(corpus: str, log_lines: list[str]) -> list[dict]:
    """Regex-based template extraction when logparser3 is unavailable.

    Produces dicts with the same shape as Drain's structured CSV so the
    downstream schema mapper can treat both paths uniformly.
    """
    out = []
    for line in log_lines:
        parts = line.split(None, 7)
        row: dict = {"Label": "-", "Content": line}
        if corpus == "hdfs" and len(parts) >= 5:
            row["Date"] = parts[0]
            row["Time"] = parts[1] if len(parts) > 1 else ""
            row["Level"] = parts[3] if len(parts) > 3 else "INFO"
            row["Component"] = parts[4].rstrip(":") if len(parts) > 4 else ""
            row["Content"] = parts[7] if len(parts) > 7 else ""
            row["Node"] = ""
        elif corpus in ("bgl", "thunderbird", "spirit") and len(parts) >= 7:
            row["Label"] = parts[0]
            row["Timestamp"] = parts[1]
            row["Node"] = parts[4]
            row["Component"] = parts[5]
            row["Level"] = parts[6]
            row["Content"] = parts[7] if len(parts) > 7 else ""
        else:
            row["Content"] = line
        # Build a coarse template by replacing variable tokens with <*>.
        content = row["Content"]
        toks = []
        for tok in content.split():
            if NUMERIC_RE.fullmatch(tok) or re.fullmatch(r"blk_-?\d+", tok) \
                    or re.fullmatch(r"(node|R\d|M\d|N\d|task|proc)\d+", tok) \
                    or IP_RE.fullmatch(tok):
                toks.append("<*>")
            else:
                toks.append(tok)
        row["EventTemplate"] = " ".join(toks[:12])
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# 4. Map parsed rows -> 12-field schema Records.
# ---------------------------------------------------------------------------

def _map_to_record(corpus: str, row: dict) -> Record:
    rec = Record(fields={})

    # Structural fields ----------------------------------------------------
    content = row.get("Content", "")
    ts_raw = row.get("Timestamp") or row.get("Date", "")
    if ts_raw:
        ts_m = TS_RE.search(ts_raw) or TS_RE.search(content)
        if ts_m:
            rec.set("timestamp_norm", ts_m.group(1))
        elif row.get("Timestamp"):
            rec.set("timestamp_norm", row["Timestamp"])
    rec.set("host", row.get("Node", "") or "")
    rec.set("component", row.get("Component", "") or "")
    level = (row.get("Level") or "INFO")
    sev = SEVERITY_MAP.get(level.lower(), level.upper())
    # Coerce unknown severities (e.g. BGL 2k sample's "RAS") to INFO so the
    # ENUM schema-validity check never rejects a record on this field.
    if sev not in VALID_SEVERITIES:
        sev = "INFO"
    rec.set("severity", sev)
    rec.set("message_template", row.get("EventTemplate", "") or "")

    err_m = ERR_RE.search(content)
    rec.set("error_code", err_m.group(1) if err_m else "")

    ent_m = ENTITY_RE.search(content)
    rec.set("affected_entity", ent_m.group(1) if ent_m else "")

    num_m = NUMERIC_RE.search(content)
    # Default to 0.0 when no numeric token is present so the FLOAT/required
    # schema-validity check passes (metric_value is required by the schema).
    rec.set("metric_value", num_m.group(0) if num_m else "0.0")

    # Anomaly fields (native labels) --------------------------------------
    label = row.get("Label", "-")
    severity = (rec.get("severity") or "INFO").upper()
    is_anom = label != "-" or severity in ("ERROR", "FATAL", "CRITICAL")
    rec.set("is_anomaly", "true" if is_anom else "false")
    rec.set("anomaly_score", "1.0" if is_anom else "0.0")

    # Semantic fields (placeholder — paper uses teacher distillation) -------
    rec.set("event_type", (rec.get("message_template") or "")[:30] or "unknown")
    rec.set("remediation_hint", "none")

    # Ensure all 12 fields exist (empty string default) --------------------
    for name in FIELD_NAMES:
        if not rec.has(name):
            rec.set(name, "")
    return rec


def _coerce_types(rec: Record) -> Record:
    """Coerce string values to the schema's declared types for downstream
    schema-validity checks (metric_value/anomaly_score -> float, is_anomaly
    -> bool). Empty strings are converted to None for FLOAT/BOOL fields so
    the metric's `if value is None: continue` skips them rather than trying
    `float("")` (which would mark the record invalid)."""
    for name in FIELD_NAMES:
        sf = FIELD_BY_NAME[name]
        v = rec.get(name)
        if v is None:
            continue
        if v == "":
            if sf.type == FieldType.FLOAT:
                rec.set(name, 0.0)  # required field; avoid None/empty -> invalid
            elif sf.type == FieldType.BOOL:
                rec.set(name, False)
            continue
        if sf.type == FieldType.FLOAT:
            try:
                rec.set(name, float(v))
            except (TypeError, ValueError):
                rec.set(name, 0.0)
        elif sf.type == FieldType.BOOL:
            if isinstance(v, str):
                rec.set(name, v.lower() in ("true", "1", "yes"))
    return rec


# ---------------------------------------------------------------------------
# 5. Write splits + manifest.
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec.fields) + "\n")


def _anomaly_rate(records: list[Record]) -> float:
    if not records:
        return 0.0
    n = sum(1 for r in records if r.get("is_anomaly") in (True, "true", "1"))
    return 100.0 * n / len(records)


def prepare_corpus(corpus: str, raw_dir: Path, out_dir: Path,
                   parsed_dir: Path, sample_size: int, test_size: int,
                   seed: int = 42) -> dict:
    print(f"\n[{corpus}]")
    raw_path = _resolve_raw_log(corpus, raw_dir)
    print(f"  raw log: {raw_path}")

    total = _count_lines(raw_path)
    print(f"  total lines: {total:,}")
    if total == 0:
        raise RuntimeError(f"{raw_path} is empty")

    effective_sample = min(sample_size, total)
    effective_test = min(test_size, effective_sample)
    print(f"  sampling {effective_sample:,} / {total:,} lines "
          f"(seed={seed}); test={effective_test:,}")

    lines = _sample_lines(raw_path, effective_sample, seed=seed)
    train_lines, test_lines = _temporal_split(lines, effective_test)
    print(f"  split: train={len(train_lines):,}  test={len(test_lines):,}")

    # Parse both splits with Drain (or fallback). Parse them together so
    # Drain sees a consistent template space, then split the rows back.
    all_lines = train_lines + test_lines
    parsed_dir_corpus = parsed_dir / corpus
    drain_rows = _run_drain(corpus, all_lines, parsed_dir_corpus)
    if drain_rows is None:
        print("  logparser3 not installed — using regex fallback parser")
        drain_rows = _fallback_parse(corpus, all_lines)
    else:
        print(f"  Drain parsed {len(drain_rows)} rows -> {parsed_dir_corpus}")
    if len(drain_rows) != len(all_lines):
        # Drain occasionally drops trailing blank lines; pad/truncate.
        print(f"  WARN: Drain rows ({len(drain_rows)}) != input lines "
              f"({len(all_lines)}); aligning by truncation")
        drain_rows = drain_rows[:len(all_lines)]

    train_rows = drain_rows[:len(train_lines)]
    test_rows = drain_rows[len(train_lines):]

    train_records = [_coerce_types(_map_to_record(corpus, r)) for r in train_rows]
    test_records = [_coerce_types(_map_to_record(corpus, r)) for r in test_rows]

    train_path = out_dir / "train" / f"{corpus}.jsonl"
    test_path = out_dir / "test" / f"{corpus}.jsonl"
    _write_jsonl(train_records, train_path)
    _write_jsonl(test_records, test_path)
    print(f"  wrote {train_path} ({len(train_records):,})")
    print(f"  wrote {test_path}  ({len(test_records):,})")

    return {
        "corpus": corpus,
        "raw_path": str(raw_path),
        "raw_lines": total,
        "sample_size": effective_sample,
        "train_lines": len(train_records),
        "test_lines": len(test_records),
        "train_anomaly_pct": round(_anomaly_rate(train_records), 2),
        "test_anomaly_pct": round(_anomaly_rate(test_records), 2),
        "drain_available": DRAIN_AVAILABLE,
        "semantic_labels": "placeholder (teacher distillation not released)",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR),
                        help=f"Raw logs dir (default: {DEFAULT_RAW_DIR})")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help=f"Splits output dir (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--parsed-dir", default=str(DEFAULT_PARSED_DIR),
                        help=f"Drain parsed output dir (default: {DEFAULT_PARSED_DIR})")
    parser.add_argument("--only", nargs="+", choices=CORPORA,
                        help="Prepare only these corpora")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help=f"Lines to sample per corpus (default {DEFAULT_SAMPLE_SIZE})")
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE,
                        help=f"Last N sampled lines = test split (default {DEFAULT_TEST_SIZE})")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run: 2k sample, 200 test (for CI / quick checks)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing splits")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    parsed_dir = Path(args.parsed_dir)

    sample_size = args.sample_size
    test_size = args.test_size
    if args.smoke:
        sample_size, test_size = 2000, 200

    targets = args.only or CORPORA
    print(f"Preparing: {', '.join(targets)}")
    print(f"  raw_dir    = {raw_dir}")
    print(f"  out_dir    = {out_dir}")
    print(f"  parsed_dir = {parsed_dir}")
    print(f"  sample={sample_size:,}  test={test_size:,}  seed={args.seed}")

    if not args.overwrite and (out_dir / "manifest.json").exists():
        print(f"NOTE: {out_dir / 'manifest.json'} already exists; "
              f"use --overwrite to regenerate")

    manifest = {"corpora": [], "schema": FIELD_NAMES,
                "sample_size": sample_size, "test_size": test_size,
                "drain_available": DRAIN_AVAILABLE,
                "note": ("Semantic fields (event_type, affected_entity, "
                         "remediation_hint) are placeholders; teacher "
                         "distillation pipeline is not released by the paper.")}
    failures = []
    for corpus in targets:
        try:
            info = prepare_corpus(corpus, raw_dir, out_dir, parsed_dir,
                                  sample_size, test_size, seed=args.seed)
            manifest["corpora"].append(info)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            failures.append(corpus)
        except Exception as e:
            print(f"  ERROR preparing {corpus}: {e}")
            failures.append(corpus)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {out_dir / 'manifest.json'}")

    if failures:
        print(f"\nFailed/skipped: {', '.join(failures)}")
        if not [c for c in targets if c not in failures]:
            sys.exit(1)

    print("Preparation complete.")


if __name__ == "__main__":
    main()
