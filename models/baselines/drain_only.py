"""
Drain-only baseline: log parsing without any LLM.

Uses the Drain algorithm (via logparser3) to extract event templates
from raw logs, then maps templates to schema fields.

This shows what classical log parsing alone achieves — high on structural
fields, zero on semantic fields.
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES

DRAIN_AVAILABLE = False
try:
    from logparser.Drain import LogParser
    DRAIN_AVAILABLE = True
except ImportError:
    pass


@dataclass
class DrainResult:
    records: list[Record]
    latencies_ms: list[float]
    templates: list[str]
    drain_available: bool


LOG_FORMATS = {
    "hdfs": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
    "bgl": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <RAS> <Component> <Level> <Content>",
    "thunderbird": "<Label> <Timestamp> <Date> <Time> <Node> <Component> <Level> <Content>",
    "spirit": "<Label> <Timestamp> <Date> <Time> <Node> <Component> <Level> <Content>",
}

SEVERITY_MAP = {
    "err": "ERROR", "error": "ERROR",
    "warn": "WARN", "warning": "WARN",
    "info": "INFO", "debug": "DEBUG",
    "fatal": "FATAL", "crit": "CRITICAL", "critical": "CRITICAL",
}


def extract(log_lines: list[str], source: str = "hdfs",
            input_dir: str = "/var/tmp/opencode/drain_input",
            output_dir: str = "/var/tmp/opencode/drain_output") -> DrainResult:
    """
    Extract structured records using Drain log parser.

    Args:
        log_lines: raw log lines
        source: one of 'hdfs', 'bgl', 'thunderbird', 'spirit'
        input_dir: temp dir for input file
        output_dir: temp dir for Drain output

    Returns:
        DrainResult with records, latencies, templates
    """
    if not DRAIN_AVAILABLE:
        return _extract_fallback(log_lines, source)

    import os
    import csv
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    log_file = f"{source}_input.log"
    log_path = os.path.join(input_dir, log_file)
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    log_format = LOG_FORMATS.get(source, LOG_FORMATS["hdfs"])
    regex = [r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)']

    parser = LogParser(
        log_format=log_format,
        indir=input_dir,
        outdir=output_dir,
        depth=4,
        st=0.5,
        rex=regex,
    )

    t0 = time.perf_counter()
    parser.parse(log_file)
    parse_time = (time.perf_counter() - t0) * 1000

    structured_path = os.path.join(output_dir, f"{log_file}_structured.csv")
    templates_path = os.path.join(output_dir, f"{log_file}_templates.csv")

    records: list[Record] = []
    templates: list[str] = []

    if os.path.exists(templates_path):
        with open(templates_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                templates.append(row.get("EventTemplate", ""))

    if os.path.exists(structured_path):
        with open(structured_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = Record(fields={})
                rec.set("message_template", row.get("EventTemplate", ""))
                rec.set("severity", SEVERITY_MAP.get(
                    row.get("Level", "").lower(),
                    row.get("Level", "INFO").upper()
                ))
                rec.set("component", row.get("Component", ""))
                rec.set("host", row.get("Node", row.get("Host", "")))
                content = row.get("Content", "")

                import re
                ts_match = re.search(r"(\d{4}\d{2}\d{2} \d{2}:\d{2}:\d{2}|\d{10,})", content)
                if ts_match:
                    rec.set("timestamp_norm", ts_match.group(1))

                err_match = re.search(r"(E\d+|err_\d+|ERR_\d+|blk_-?\d+)", content)
                if err_match:
                    rec.set("error_code", err_match.group(1))

                entity_match = re.search(r"(blk_-?\d+|node\d+|R\d+-M\d+-N\d+)", content)
                if entity_match:
                    rec.set("affected_entity", entity_match.group(1))

                metric_match = re.search(r"(\d+\.?\d*)\s*(bytes?|KB|MB|GB|ms|s)?", content)
                if metric_match:
                    rec.set("metric_value", metric_match.group(1))

                severity = rec.get("severity", "INFO")
                is_anom = severity in ("ERROR", "FATAL", "CRITICAL") or row.get("Label", "-") != "-"
                rec.set("is_anomaly", "true" if is_anom else "false")
                rec.set("anomaly_score", "1.0" if is_anom else "0.0")
                rec.set("event_type", rec.get("message_template", "")[:30])
                rec.set("remediation_hint", "none")

                for field_name in FIELD_NAMES:
                    if not rec.has(field_name):
                        rec.set(field_name, "")

                records.append(rec)

    per_event_latency = parse_time / len(log_lines) if log_lines else 0
    latencies = [per_event_latency] * len(records)

    return DrainResult(
        records=records,
        latencies_ms=latencies,
        templates=templates,
        drain_available=True,
    )


def _extract_fallback(log_lines: list[str], source: str = "hdfs") -> DrainResult:
    """Fallback when logparser3 is not installed — uses regex-based template detection."""
    import re
    from collections import defaultdict

    records: list[Record] = []
    latencies: list[float] = []
    template_counts: dict[str, int] = defaultdict(int)

    for line in log_lines:
        t0 = time.perf_counter()
        line_stripped = line.strip()
        rec = Record(fields={})

        parts = line_stripped.split(None, 7)
        label = "-"

        if source == "hdfs" and len(parts) >= 5:
            rec.set("timestamp_norm", f"{parts[0]} {parts[1]}")
            rec.set("host", parts[2] if len(parts) > 2 else "")
            rec.set("severity", SEVERITY_MAP.get(parts[3].lower(), parts[3].upper()) if len(parts) > 3 else "INFO")
            rec.set("component", parts[4].rstrip(":") if len(parts) > 4 else "")
            content = parts[7] if len(parts) > 7 else ""
        elif source == "bgl":
            # BGL full log: <Label> <Timestamp> <Date> <Node> <Time>
            #   <NodeRepeat> <RAS> <Component> <Level> <Content>
            bgl_parts = line_stripped.split(None, 9)
            if len(bgl_parts) >= 9:
                label = bgl_parts[0]
                rec.set("timestamp_norm", bgl_parts[1])
                rec.set("host", bgl_parts[3])
                rec.set("component", bgl_parts[7])
                rec.set("severity", SEVERITY_MAP.get(bgl_parts[8].lower(), bgl_parts[8].upper()))
                content = bgl_parts[9] if len(bgl_parts) > 9 else ""
            else:
                content = line_stripped
        elif source in ("thunderbird", "spirit") and len(parts) >= 7:
            label = parts[0]
            rec.set("timestamp_norm", parts[1])
            rec.set("host", parts[4])
            rec.set("component", parts[5])
            rec.set("severity", SEVERITY_MAP.get(parts[6].lower(), parts[6].upper()) if len(parts) > 6 else "INFO")
            content = parts[7] if len(parts) > 7 else ""
        else:
            content = line_stripped

        tokens = content.split()
        template_tokens = []
        for token in tokens:
            if re.match(r"^\d+\.?\d*$", token):
                template_tokens.append("<*>")
            elif re.match(r"^blk_-?\d+$", token):
                template_tokens.append("<*>")
            elif re.match(r"^(node|R\d|M\d|N\d|task|proc)\d+", token):
                template_tokens.append("<*>")
            elif re.match(r"^(/|)([0-9]+\.){3}[0-9]+", token):
                template_tokens.append("<*>")
            else:
                template_tokens.append(token)

        template = " ".join(template_tokens[:10])
        rec.set("message_template", template)
        template_counts[template] += 1

        err_match = re.search(r"(E\d+|err_\d+|ERR_\d+|blk_-?\d+)", content)
        if err_match:
            rec.set("error_code", err_match.group(1))

        entity_match = re.search(r"(blk_-?\d+|node\d+|R\d+-M\d+-N\d+)", content)
        if entity_match:
            rec.set("affected_entity", entity_match.group(1))

        metric_match = re.search(r"(\d+\.?\d*)", content)
        if metric_match:
            rec.set("metric_value", metric_match.group(1))

        severity = rec.get("severity", "INFO")
        is_anom = severity in ("ERROR", "FATAL", "CRITICAL") or label != "-"
        rec.set("is_anomaly", "true" if is_anom else "false")
        rec.set("anomaly_score", "1.0" if is_anom else "0.0")
        rec.set("event_type", template[:30])
        rec.set("remediation_hint", "none")

        for field_name in FIELD_NAMES:
            if not rec.has(field_name):
                rec.set(field_name, "")

        records.append(rec)
        latencies.append((time.perf_counter() - t0) * 1000)

    return DrainResult(
        records=records,
        latencies_ms=latencies,
        templates=list(template_counts.keys()),
        drain_available=False,
    )
