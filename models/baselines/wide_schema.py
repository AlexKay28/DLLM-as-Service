#!/usr/bin/env python3
"""
Issue #19 (App C): wide-schema ladder + deterministic gold labeler.

Every field beyond the original 12 is a deterministic function of the 12
gold fields plus the raw line.  No new hand-labeling, no teacher model.

Schema ladder (each level is a strict superset):
  L12  = 12 base fields (existing schema)
  L25  = L12 + 13 derived structural fields
  L50  = L25 + 25 ECS/OCSF envelope fields
  L100 = L50 + 50 normalization-variant fields
"""

import hashlib
import json
import re
from metrics.schema import FIELD_NAMES

SEVERITY_ORDINAL = {
    "DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "FATAL": 4, "CRITICAL": 5,
}


def _hash_str(s: str, n: int = 8) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:n]


def _decompose_bgl_host(host: str) -> dict:
    """R13-M0-N7-C:J04-U01 → rack, midplane, node, card, jtag, unit."""
    out = {"rack": "", "midplane": "", "node": "", "card": "", "jtag": "", "unit": ""}
    if not host:
        return out
    m = re.match(r"R(\d+)-M(\d+)-N(\d+)-([A-Z])(?::J(\d+)-U(\d+))?", host)
    if m:
        out = {
            "rack": f"R{m.group(1)}", "midplane": f"M{m.group(2)}",
            "node": f"N{m.group(3)}", "card": m.group(4),
            "jtag": f"J{m.group(5)}" if m.group(5) else "",
            "unit": f"U{m.group(6)}" if m.group(6) else "",
        }
    return out


def _parse_ts(ts: str) -> dict:
    out = {"ts_year": "", "ts_month": "", "ts_day": "",
           "ts_hour": "", "ts_minute": "", "ts_second": ""}
    if not ts:
        return out
    m = re.match(r"(\d{4})\.(\d{2})\.(\d{2})", ts)
    if m:
        out["ts_year"] = m.group(1)
        out["ts_month"] = m.group(2)
        out["ts_day"] = m.group(3)
    m2 = re.search(r"(\d{2}):(\d{2}):(\d{2})", ts)
    if m2:
        out["ts_hour"] = m2.group(1)
        out["ts_minute"] = m2.group(2)
        out["ts_second"] = m2.group(3)
    return out


def derive_fields(gold: dict, raw_line: str, level: int) -> dict:
    """Deterministically derive wider fields from the 12 gold + raw line."""
    out = dict(gold)

    if level <= 12:
        return out

    # L25: +13 structural derivations
    out.update(_parse_ts(gold.get("timestamp_norm", "")))
    out.update(_decompose_bgl_host(gold.get("host", "")))
    sev = gold.get("severity", "")
    out["severity_ordinal"] = SEVERITY_ORDINAL.get(sev, 0)

    if level <= 25:
        return out

    # L50: +25 ECS/OCSF envelope mappings
    out["event.created"] = gold.get("timestamp_norm", "")
    out["event.category"] = "log"
    out["event.type"] = gold.get("event_type", "")
    out["event.severity"] = sev
    out["event.action"] = gold.get("event_type", "")

    out["host.name"] = gold.get("host", "")
    out["host.id"] = _hash_str(gold.get("host", ""))
    out["host.hostname"] = gold.get("host", "")
    hd = _decompose_bgl_host(gold.get("host", ""))
    out["host.rack"] = hd["rack"]
    out["host.midplane"] = hd["midplane"]

    out["log.level"] = sev
    out["log.source"] = gold.get("component", "")
    out["log.original"] = raw_line[:200]
    out["log.facility"] = gold.get("component", "")
    out["log.logger"] = gold.get("component", "")

    out["observer.name"] = gold.get("component", "")
    out["observer.type"] = "system"
    out["observer.vendor"] = "ibm"
    out["observer.version"] = "unknown"
    out["observer.product"] = "bgl"

    out["related.hosts"] = gold.get("host", "")
    out["related.ip"] = ""
    out["related.entities"] = gold.get("affected_entity", "")
    out["related.templates"] = gold.get("message_template", "")
    out["related.users"] = ""

    if level <= 50:
        return out

    # L100: +50 normalization variants
    mv = gold.get("metric_value", 0.0)
    out["metric_value_normalized"] = round(float(mv) / 100.0, 4) if mv else 0.0
    out["metric_value_log"] = round(max(0, float(mv)), 4) if mv else 0.0
    out["metric_value_zscore"] = 0.0

    for s in SEVERITY_ORDINAL:
        out[f"severity_cat_{s.lower()}"] = 1 if sev == s else 0

    out["component_hash"] = _hash_str(gold.get("component", ""))
    out["event_type_hash"] = _hash_str(gold.get("event_type", ""))
    out["template_hash"] = _hash_str(gold.get("message_template", ""))
    out["error_code_hash"] = _hash_str(gold.get("error_code", ""))

    for fn in FIELD_NAMES:
        out[f"confidence_{fn}"] = 1.0 if gold.get(fn) else 0.5

    out["cross_ref_host_component"] = f"{gold.get('host','')}:{gold.get('component','')}"
    out["cross_ref_host_severity"] = f"{gold.get('host','')}:{sev}"
    out["cross_ref_component_severity"] = f"{gold.get('component','')}:{sev}"
    out["cross_ref_template_event"] = f"{gold.get('message_template','')}:{gold.get('event_type','')}"
    out["cross_ref_entity_component"] = f"{gold.get('affected_entity','')}:{gold.get('component','')}"

    filled = sum(1 for v in gold.values() if v not in ("", 0, 0.0, False, None))
    out["field_group_structural_count"] = 7
    out["field_group_semantic_count"] = 3
    out["field_group_anomaly_count"] = 2
    out["field_group_total_count"] = 12
    out["field_group_empty_count"] = 12 - filled

    out["normalized_host_upper"] = gold.get("host", "").upper()
    out["normalized_component_upper"] = gold.get("component", "").upper()
    out["normalized_event_type_lower"] = gold.get("event_type", "").lower()
    out["normalized_template_compact"] = re.sub(r"\s+", "", gold.get("message_template", ""))
    out["normalized_severity_upper"] = sev.upper()

    out["extraction_time_epoch"] = "0"
    out["extraction_time_iso"] = gold.get("timestamp_norm", "")
    out["extraction_source"] = "bgl"
    out["extraction_confidence"] = round(filled / 12.0, 2)
    out["extraction_version"] = "1.0"

    base_id = _hash_str(raw_line)
    out["ref_event_id"] = f"evt_{base_id}"
    out["ref_record_id"] = f"rec_{base_id}"
    out["ref_correlation_id"] = f"corr_{base_id[:4]}"
    out["ref_parent_id"] = f"par_{base_id[:4]}"
    out["ref_trace_id"] = f"trc_{base_id}"

    return out


SCHEMA_LEVELS = {12, 25, 50, 100}


def field_names_for_level(level: int) -> list[str]:
    """Return the ordered field names for a given schema level."""
    dummy_gold = {fn: "" for fn in FIELD_NAMES}
    full = derive_fields(dummy_gold, "dummy", 100)
    base = list(FIELD_NAMES)
    if level == 12:
        return base
    derived = [k for k in full if k not in base]
    if level == 25:
        return base + derived[:13]
    if level == 50:
        return base + derived[:38]
    return base + derived


def build_schema_description(level: int) -> str:
    """Build a JSON schema description string for the prompt."""
    names = field_names_for_level(level)
    dummy = derive_fields({fn: "" for fn in FIELD_NAMES}, "dummy", 100)
    lines = []
    for fn in names:
        val = dummy.get(fn, "")
        if isinstance(val, bool):
            t = "boolean"
        elif isinstance(val, (int, float)):
            t = "float"
        else:
            t = "string"
        lines.append(f'  "{fn}": "{t}"')
    return "{\n" + ",\n".join(lines) + "\n}"


def build_wide_system_prompt(level: int) -> str:
    """Build the system prompt for a given schema level."""
    schema = build_schema_description(level)
    return f"""You are a log analysis expert. Extract structured fields from raw log lines.

Given a raw log line, output a JSON object with exactly these {len(field_names_for_level(level))} fields:
{schema}

Rules:
- Output ONLY valid JSON, no explanations.
- If a field is not present in the log, use empty string "" for strings, 0.0 for floats, false for booleans, 0 for integers.
- All field names must match exactly as shown above.
- Ensure all fields are present and internally consistent (e.g. host.rack must match the rack portion of host.name)."""


def build_wide_chat_messages(log_line: str, source: str, level: int,
                             few_shot: bool = True) -> list[dict]:
    from models.baselines.ar_model import FEW_SHOT_EXAMPLES
    system = build_wide_system_prompt(level)
    user = ""
    if few_shot:
        user += "Examples:\n"
        for ex in FEW_SHOT_EXAMPLES:
            wide = derive_fields(ex["output"], ex["input"], level)
            wide = {k: wide[k] for k in field_names_for_level(level)}
            user += f'  Input: "{ex["input"]}"\n'
            user += f"  Output: {json.dumps(wide)}\n"
        user += "\n"
    user += f"Source: {source}\n"
    user += f'Input: "{log_line}"\n'
    user += "Output:"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]
