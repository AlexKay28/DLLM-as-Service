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


def _looks_like_gold_record(log_line: str) -> bool:
    """Return True if ``log_line`` is a gold JSON record rather than a raw log
    line.

    Raw BGL/HDFS log lines are whitespace-delimited text and never start with
    ``{``; a gold record (from ``datasets/splits/test/*.jsonl``) always does.
    This guards against the issue-#112 defect class: a harness that loads the
    gold file and feeds it here as ``log_line`` prompts the model with the
    answer it was supposed to extract. The bug is otherwise silent -- the model
    scores perfect F1 by copying the input -- so this assertion is the only
    detector at the shared entry point.
    """
    s = log_line.strip()
    if not s or s[0] != "{":
        return False
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(obj, dict) and any(f in obj for f in FIELD_NAMES)


def build_chat_messages(log_line: str, source: str = "hdfs",
                        few_shot: bool = True, n_few_shot: int = None) -> list[dict]:
    """Build chat-format messages for models that use chat templates.

    Args:
        log_line: raw log line.
        source: log source.
        few_shot: include few-shot examples.
        n_few_shot: if set, use only the first N few-shot examples (for
                   input-length sweeping, issue #44). None = all examples.

    Returns:
        List of {"role": "system"/"user", "content": str} messages.
    """
    if _looks_like_gold_record(log_line):
        raise AssertionError(
            f"build_chat_messages received a gold JSON record instead of a raw "
            f"log line (issue #112): the model would be prompted with the "
            f"answer. Pass a line from datasets/splits/test/*.raw, not *.jsonl. "
            f"Got: {log_line[:120]!r}"
        )
    user_content = ""
    if few_shot:
        examples = FEW_SHOT_EXAMPLES
        if n_few_shot is not None:
            examples = examples[:n_few_shot]
        user_content += "Examples:\n"
        for ex in examples:
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


def parse_json_output(raw_output: str, policy: str = "first_object_wins") -> dict:
    """Parse the model's JSON output, handling common failure modes.

    Policies (#57 R5):
    - "first_object_wins" (primary, per #43): extract the first {...} block
      and parse it. Tolerant of trailing text, extra prose, markdown fences.
      This is the default and the policy that should be recorded in result JSONs.
    - "strict": require the entire output (after fence stripping) to be valid
      JSON. Used as a secondary column for comparison.

    Args:
        raw_output: raw model output string.
        policy: "first_object_wins" (default) or "strict".

    Returns:
        Parsed dict with field names as keys, or empty dict on failure.
    """
    text = raw_output.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if policy == "strict":
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            return {}
        return {}

    # first_object_wins: try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Extract first complete {...} block by tracking brace depth.
    # The old code used text.rfind("}") which captures first-{ to last-},
    # effectively scoring the last (most-drifted) copy of each field (#43).
    start = text.find("{")
    if start != -1:
        depth = 0
        end = -1
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            block = text[start:end + 1]
            try:
                result = json.loads(block)
                if isinstance(result, dict):
                    return result
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
