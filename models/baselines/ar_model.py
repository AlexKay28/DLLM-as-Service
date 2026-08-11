"""
Prompt builder for AR log-extraction models.

Constructs the instruction prompt that asks an AR LLM (Qwen2.5, Llama, etc.)
to extract the 12-field schema from a raw log line, returning strict JSON.

This is the AR-side equivalent of FlowFill's denoising task — both produce the
same 12-field Record, but AR does it autoregressively (128 steps) while
FlowFill does it in K=8 parallel denoising steps.
"""

import json
from metrics.schema import FIELD_NAMES, FIELD_BY_NAME, FieldType, FieldCategory

# The 12-field schema description given to the model.
SCHEMA_DESCRIPTION = """{
  "timestamp_norm":  "string  — normalized timestamp (epoch seconds or ISO 8601)",
  "host":            "string  — hostname / node identifier",
  "component":       "string  — software component (e.g. KERNEL, APP, MMCS, DataNode)",
  "severity":        "enum    — one of: DEBUG, INFO, WARN, ERROR, FATAL, CRITICAL",
  "event_type":      "string  — semantic event category (e.g. cache_parity, io_timeout)",
  "message_template":"string  — log template with variable tokens replaced by <*>",
  "error_code":      "string  — error/exception code if present, else empty",
  "affected_entity": "string  — resource/entity affected (block id, node, task)",
  "metric_value":    "float   — numeric metric value if present, else 0.0",
  "is_anomaly":      "boolean — true if this log line indicates an anomaly",
  "anomaly_score":   "float   — 0.0 (normal) to 1.0 (anomaly)",
  "remediation_hint": "string — suggested remediation action or 'none'"
}"""

SYSTEM_PROMPT = """You are a log analysis expert. Extract structured fields from raw log lines.

Given a raw log line, output a JSON object with exactly these 12 fields:
{schema}

Rules:
- Output ONLY valid JSON, no explanations.
- If a field is not present in the log, use empty string "" for strings, 0.0 for floats, false for booleans.
- severity must be one of: DEBUG, INFO, WARN, ERROR, FATAL, CRITICAL.
- message_template should replace variable tokens (IPs, numbers, IDs) with <*>
- is_anomaly: true for ERROR/FATAL/CRITICAL severity or explicit alert labels.
- anomaly_score: 1.0 for anomalies, 0.0 for normal, 0.5 for warnings.
- event_type: use plain words with spaces, NOT underscores (e.g. "data storage interrupt", NOT "data_storage_interrupt").
- affected_entity: only fill if a distinct non-host resource is mentioned (block id, task id); use "" if none.
- remediation_hint: "none" if no specific action is needed.""".format(schema=SCHEMA_DESCRIPTION)


FEW_SHOT_EXAMPLES = [
    {
        "input": "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating",
        "output": {
            "timestamp_norm": "081109 203615",
            "host": "",
            "component": "DataNode",
            "severity": "INFO",
            "event_type": "packet responder terminate",
            "message_template": "PacketResponder <*> for block <*> terminating",
            "error_code": "",
            "affected_entity": "blk_38865049064139660",
            "metric_value": 0.0,
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "remediation_hint": "none"
        }
    },
    {
        "input": "KERNSTOR 1118709471 2005.06.13 R13-M0-N7-C:J04-U01 2005-06-13-17.37.51.375330 R13-M0-N7-C:J04-U01 RAS KERNEL FATAL data storage interrupt",
        "output": {
            "timestamp_norm": "1118709471",
            "host": "R13-M0-N7-C:J04-U01",
            "component": "KERNEL",
            "severity": "FATAL",
            "event_type": "data storage interrupt",
            "message_template": "data storage interrupt",
            "error_code": "",
            "affected_entity": "",
            "metric_value": 0.0,
            "is_anomaly": True,
            "anomaly_score": 1.0,
            "remediation_hint": "none"
        }
    },
]


def build_prompt(log_line: str, source: str = "hdfs",
                 few_shot: bool = True) -> str:
    """Build the full prompt for a single log line.

    Args:
        log_line: raw log line to extract fields from.
        source: log source (hdfs, bgl, thunderbird, spirit) for context.
        few_shot: include few-shot examples in the prompt.

    Returns:
        The full prompt string (system + examples + query).
    """
    parts = [SYSTEM_PROMPT]

    if few_shot:
        parts.append("\nExamples:")
        for ex in FEW_SHOT_EXAMPLES:
            parts.append(f'  Input: "{ex["input"]}"')
            parts.append(f'  Output: {json.dumps(ex["output"])}')

    parts.append(f'\nSource: {source}')
    parts.append(f'Input: "{log_line}"')
    parts.append("Output:")

    return "\n".join(parts)


def build_chat_messages(log_line: str, source: str = "hdfs",
                        few_shot: bool = True) -> list[dict]:
    """Build chat-format messages for models that use chat templates.

    Args:
        log_line: raw log line.
        source: log source.
        few_shot: include few-shot examples.

    Returns:
        List of {"role": "system"/"user", "content": str} messages.
    """
    user_content = ""
    if few_shot:
        user_content += "Examples:\n"
        for ex in FEW_SHOT_EXAMPLES:
            user_content += f'  Input: "{ex["input"]}"\n'
            user_content += f'  Output: {json.dumps(ex["output"])}\n'
        user_content += "\n"

    user_content += f"Source: {source}\n"
    user_content += f'Input: "{log_line}"\n'
    user_content += "Output:"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_json_output(raw_output: str) -> dict:
    """Parse the model's JSON output, handling common failure modes.

    - Strips markdown code fences (```json ... ```).
    - Extracts the first {...} block if extra text is present.
    - Returns empty dict on parse failure.

    Args:
        raw_output: raw model output string.

    Returns:
        Parsed dict with field names as keys, or empty dict on failure.
    """
    text = raw_output.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        block = text[start:end + 1]
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass

        # Regex fallback: recover well-formed "key": value pairs even if the
        # overall document is malformed (missing colon on one line, stray
        # comment, unbalanced quote, etc.) — common MDM denoising artifacts
        # where most fields decode correctly but one is garbled.
        recovered = _extract_key_value_pairs(block)
        if recovered:
            return recovered

    return {}


import re as _re

_KV_PATTERN = _re.compile(
    r'"(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)"\s*:\s*'
    r'(?:"(?P<str_val>(?:[^"\\]|\\.)*)"'
    r'|(?P<bool_val>true|false)'
    r'|(?P<num_val>-?\d+\.?\d*))'
)


def _extract_key_value_pairs(text: str) -> dict:
    """Recover "key": value pairs via regex from a malformed JSON blob.

    Only matches syntactically well-formed individual pairs (quoted key,
    colon, quoted string / bool / number value) — does not guess or repair
    broken pairs. Used as a last-resort fallback when a document has one
    or two garbled fields but the rest are recoverable.
    """
    result = {}
    for m in _KV_PATTERN.finditer(text):
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
