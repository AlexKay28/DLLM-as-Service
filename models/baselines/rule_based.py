"""
Rule-based log extractor using regex patterns.

Simulates what a production engineer would write: per-source regex
rules to extract structured fields from raw log lines.

This is the "grep/awk/regex" baseline — fast, deterministic, zero GPU,
but requires hand-crafted patterns per log source and breaks on
format drift.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES, FIELD_BY_NAME, FieldType


def _coerce_types(rec: Record) -> Record:
    """Coerce string values to the schema's declared types so predictions
    match gold records (which are coerced in prepare.py). Without this,
    BOOL fields like is_anomaly compare 'True' vs 'true' -> F1=0."""
    for name in FIELD_NAMES:
        sf = FIELD_BY_NAME[name]
        v = rec.get(name)
        if v is None or v == "":
            if sf.type == FieldType.FLOAT:
                rec.set(name, 0.0)
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


SEVERITY_MAP = {
    "err": "ERROR", "error": "ERROR",
    "warn": "WARN", "warning": "WARN",
    "info": "INFO", "debug": "DEBUG",
    "fatal": "FATAL", "crit": "CRITICAL", "critical": "CRITICAL",
    "emerg": "CRITICAL", "alert": "CRITICAL", "notice": "INFO",
}

HDFS_PATTERNS = {
    # HDFS format: "<Date> <Time> <Pid> <Level> <Component>: <Content>"
    # No host column — leave empty (host is not extractable from HDFS logs).
    "timestamp_norm": r"^(\d{6} \d{6}|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})",
    "host": r"",
    "component": r"^(?:\S+\s){3}\S+\s+([A-Za-z]\w*):",
    "severity": r"^(?:\S+\s){3}(INFO|ERROR|WARN|DEBUG|FATAL|CRITICAL)\b",
    "message_template": r"^(?:\S+\s){4}\S+:\s+(.+)$",
    "error_code": r"(E\d+|err_\d+|ERR_\d+|blk_-?\d+|task_\d+)",
    "affected_entity": r"(blk_-?\d+|task_\d+|node\d+)",
    "metric_value": r"(?:\S+\s){4}\S+:\s+.*\b(\d+(?:\.\d+)?)\s*(bytes?|KB|MB|GB|ms|s)?\s*$",
    "is_anomaly": r"^(?:\S+\s){3}(ERROR|FATAL|CRITICAL)\b|\b(exception|failed|failure|recv fail|timeout)\b",
    "anomaly_score": r"",
    "event_type": r"",
    "remediation_hint": r"",
}

BGL_PATTERNS = {
    "timestamp_norm": r"^(\d+)",
    "host": r"(R\d+-M\d+-N\d+|node\d+|midplane\d+)",
    "component": r"(MMCS|KERNEL|BGLMODULE|CFD|PE|IO|CMCS|GW|ML|APP)",
    "severity": r"\b(INFO|ERROR|WARN|FATAL|CRITICAL|DEBUG|SEVERE|WARNING)\b",
    "message_template": r"^(.*?)(?::|\s{2,})",
    "error_code": r"(err_\d+|ERR_\d+|KERN_\d+|CFD_\d+|MMCS_\d+)",
    "affected_entity": r"(R\d+-M\d+-N\d+|node\d+|task\d+|proc\d+)",
    "metric_value": r"(\d+\.?\d*)",
    "is_anomaly": r"^(?!-)(\S+)",
    "anomaly_score": r"",
    "event_type": r"",
    "remediation_hint": r"",
}

THUNDERBIRD_PATTERNS = {
    "timestamp_norm": r"^(\d+)",
    "host": r"(tbird-\d+|node\d+|c\d+-\d+)",
    "component": r"(nfs|kernel| Lustre|MMCS|MPI|GM|VFS|TCP|eth\d+)",
    "severity": r"\b(INFO|ERROR|WARN|FATAL|CRITICAL|DEBUG|WARNING|ERR)\b",
    "message_template": r"^(.*?)(?::|\s{2,})",
    "error_code": r"(err_\d+|ERR_\d+|nfs\d+| Lustre\d*)",
    "affected_entity": r"(node\d+|tbird-\d+|c\d+-\d+|task\d+)",
    "metric_value": r"(\d+\.?\d*)",
    "is_anomaly": r"^(?!-)(\S+)",
    "anomaly_score": r"",
    "event_type": r"",
    "remediation_hint": r"",
}

SPIRIT_PATTERNS = {
    "timestamp_norm": r"^(\d+)",
    "host": r"(spirit-\d+|node\d+|c\d+-\d+|s\d+)",
    "component": r"(kernel|nfs|MPI|GM|VFS|TCP|eth\d+|CRAY|XT3|SeaStar)",
    "severity": r"\b(INFO|ERROR|WARN|FATAL|CRITICAL|DEBUG|WARNING|ERR)\b",
    "message_template": r"^(.*?)(?::|\s{2,})",
    "error_code": r"(err_\d+|ERR_\d+|nfs\d+|lustre\d*)",
    "affected_entity": r"(node\d+|spirit-\d+|c\d+-\d+|s\d+|task\d+)",
    "metric_value": r"(\d+\.?\d*)",
    "is_anomaly": r"^(?!-)(\S+)",
    "anomaly_score": r"",
    "event_type": r"",
    "remediation_hint": r"",
}

PATTERNS_BY_SOURCE = {
    "hdfs": HDFS_PATTERNS,
    "bgl": BGL_PATTERNS,
    "thunderbird": THUNDERBIRD_PATTERNS,
    "spirit": SPIRIT_PATTERNS,
}

FIELDS_RULES_CAN_EXTRACT = [
    "timestamp_norm", "host", "component", "severity",
    "message_template", "error_code", "affected_entity", "metric_value",
    "is_anomaly",
]

FIELDS_RULES_CANNOT_EXTRACT = [
    "event_type", "anomaly_score", "remediation_hint",
]


@dataclass
class RuleBasedResult:
    records: list[Record]
    latencies_ms: list[float]
    fields_extracted: list[str]
    fields_skipped: list[str]


def extract(log_lines: list[str], source: str = "hdfs") -> RuleBasedResult:
    """
    Extract structured records from raw log lines using regex patterns.

    Args:
        log_lines: raw log lines
        source: one of 'hdfs', 'bgl', 'thunderbird', 'spirit'

    Returns:
        RuleBasedResult with records, latencies, and field info
    """
    patterns = PATTERNS_BY_SOURCE.get(source, HDFS_PATTERNS)
    records: list[Record] = []
    latencies: list[float] = []

    compiled = {}
    for field_name, pattern_str in patterns.items():
        if pattern_str:
            compiled[field_name] = re.compile(pattern_str, re.IGNORECASE)
        else:
            compiled[field_name] = None

    for line in log_lines:
        t0 = time.perf_counter()
        rec = Record(fields={})
        line_stripped = line.strip()

        # Pass 1: regex-extractable fields (structural + anomaly flag).
        for field_name, regex in compiled.items():
            if regex is None:
                continue
            match = regex.search(line_stripped)
            if not match:
                continue
            value = match.group(1) if match.groups() else match.group(0)

            if field_name == "severity":
                value = SEVERITY_MAP.get(value.lower(), value.upper())
            elif field_name == "is_anomaly":
                value = "true" if value and value != "-" else "false"
            elif field_name in ("metric_value", "anomaly_score"):
                try:
                    value = str(float(value))
                except (ValueError, TypeError):
                    pass

            rec.set(field_name, value)

        # Pass 2: derive fields that depend on Pass 1, then fill defaults.
        severity = (rec.get("severity") or "").upper()
        if not rec.has("is_anomaly"):
            rec.set("is_anomaly", "true" if severity in ("ERROR", "FATAL", "CRITICAL") else "false")
        rec.set("anomaly_score", "1.0" if rec.get("is_anomaly") == "true" else "0.0")
        msg = rec.get("message_template") or ""
        rec.set("event_type", msg[:30] if msg else "unknown")
        rec.set("remediation_hint", "none")

        for field_name in FIELD_NAMES:
            if not rec.has(field_name):
                rec.set(field_name, "")

        records.append(_coerce_types(rec))
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    return RuleBasedResult(
        records=records,
        latencies_ms=latencies,
        fields_extracted=FIELDS_RULES_CAN_EXTRACT,
        fields_skipped=FIELDS_RULES_CANNOT_EXTRACT,
    )


def extract_field_from_line(line: str, field_name: str, source: str = "hdfs") -> Optional[str]:
    """Extract a single field from a log line."""
    patterns = PATTERNS_BY_SOURCE.get(source, HDFS_PATTERNS)
    pattern_str = patterns.get(field_name, "")
    if not pattern_str:
        return None
    match = re.search(pattern_str, line, re.IGNORECASE)
    if match:
        return match.group(1) if match.groups() else match.group(0)
    return None
