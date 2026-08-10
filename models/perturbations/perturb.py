"""
Log perturbation engine: inject subtle changes that break regex rules
but are trivially readable by an LLM.

Each perturbation simulates a real-world scenario:
  - Whitespace variations (tabs, extra spaces, no spaces)
  - Field reordering (same content, different column order)
  - Separator changes (| -> ; -> , -> :)
  - Abbreviation/synonym (ERROR -> ERR, WARNING -> WARN)
  - Timestamp format shifts (epoch -> ISO 8601 -> RFC3339)
  - Missing fields (some logs omit a field — rules crash, LLM infers)
  - Case variation (UPPER -> lower -> Mixed)
  - Wrapping noise (extra prefix/suffix, log level moved into content)
"""

import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PerturbationType(Enum):
    WHITESPACE = "whitespace"
    FIELD_REORDER = "field_reorder"
    SEPARATOR_CHANGE = "separator_change"
    ABBREVIATION = "abbreviation"
    TIMESTAMP_FORMAT = "timestamp_format"
    MISSING_FIELD = "missing_field"
    CASE_VARIATION = "case_variation"
    WRAPPING_NOISE = "wrapping_noise"
    SYNONYM_SUBSTITUTION = "synonym_substitution"
    ENCODING_NOISE = "encoding_noise"


ABBREVIATIONS = {
    "ERROR": "ERR",
    "WARNING": "WARN",
    "CRITICAL": "CRIT",
    "INFORMATION": "INFO",
    "DEBUG": "DBG",
    "FATAL": "FTL",
    "exception": "exc",
    "failure": "fail",
    "received": "recv",
    "terminated": "term",
    "starting": "start",
    "connection": "conn",
    "response": "resp",
    "request": "req",
    "block": "blk",
    "packet": "pkt",
    "message": "msg",
    "node": "nd",
}

SYNONYMS = {
    "ERROR": {"FAILED", "BROKEN", "FAULT"},
    "WARN": {"CAUTION", "ALERT", "NOTICE"},
    "INFO": {"STATUS", "NOTICE", "LOG"},
    "block": {"chunk", "segment", "unit"},
    "node": {"host", "server", "machine"},
    "received": {"got", "accepted", "acquired"},
    "failed": {"errored", "aborted", "lost"},
    "connection": {"link", "channel", "session"},
}

TIMESTAMP_FORMATS = [
    ("0813 09:21:14", "2025-08-13T09:21:14Z"),          # HDFS -> ISO 8601
    ("0813 09:21:14", "2025-08-13 09:21:14+00:00"),      # HDFS -> RFC3339
    ("1117817541", "2025-08-13T09:21:14Z"),              # epoch -> ISO 8601
    ("1117817541", "2025/08/13 09:21:14"),               # epoch -> slash format
    ("0813 09:21:14", "Aug 13 09:21:14"),                # HDFS -> month name
]


@dataclass
class PerturbationConfig:
    perturbation_type: PerturbationType
    intensity: float = 1.0  # 0.0 = none, 1.0 = full
    seed: int = 42


def perturb_logs(
    log_lines: list[str],
    config: PerturbationConfig,
) -> list[str]:
    """
    Apply a perturbation to a list of log lines.

    Args:
        log_lines: raw log lines
        config: perturbation type and intensity

    Returns:
        perturbed log lines
    """
    rng = random.Random(config.seed)
    perturbed = []

    for line in log_lines:
        if rng.random() > config.intensity:
            perturbed.append(line)
            continue

        ptype = config.perturbation_type

        if ptype == PerturbationType.WHITESPACE:
            perturbed.append(_perturb_whitespace(line, rng))
        elif ptype == PerturbationType.FIELD_REORDER:
            perturbed.append(_perturb_field_reorder(line, rng))
        elif ptype == PerturbationType.SEPARATOR_CHANGE:
            perturbed.append(_perturb_separator_change(line, rng))
        elif ptype == PerturbationType.ABBREVIATION:
            perturbed.append(_perturb_abbreviation(line, rng))
        elif ptype == PerturbationType.TIMESTAMP_FORMAT:
            perturbed.append(_perturb_timestamp(line, rng))
        elif ptype == PerturbationType.MISSING_FIELD:
            perturbed.append(_perturb_missing_field(line, rng))
        elif ptype == PerturbationType.CASE_VARIATION:
            perturbed.append(_perturb_case_variation(line, rng))
        elif ptype == PerturbationType.WRAPPING_NOISE:
            perturbed.append(_perturb_wrapping_noise(line, rng))
        elif ptype == PerturbationType.SYNONYM_SUBSTITUTION:
            perturbed.append(_perturb_synonym_substitution(line, rng))
        elif ptype == PerturbationType.ENCODING_NOISE:
            perturbed.append(_perturb_encoding_noise(line, rng))
        else:
            perturbed.append(line)

    return perturbed


def perturb_mixed(
    log_lines: list[str],
    intensity: float = 0.5,
    seed: int = 42,
) -> tuple[list[str], list[PerturbationType]]:
    """
    Apply a mix of all perturbations at a given intensity level.

    Returns perturbed lines and the list of perturbation types applied.
    """
    rng = random.Random(seed)
    all_types = list(PerturbationType)
    perturbed = []
    applied_types = []

    for line in log_lines:
        if rng.random() > intensity:
            perturbed.append(line)
            continue

        ptype = rng.choice(all_types)
        if ptype == PerturbationType.WHITESPACE:
            perturbed.append(_perturb_whitespace(line, rng))
        elif ptype == PerturbationType.FIELD_REORDER:
            perturbed.append(_perturb_field_reorder(line, rng))
        elif ptype == PerturbationType.SEPARATOR_CHANGE:
            perturbed.append(_perturb_separator_change(line, rng))
        elif ptype == PerturbationType.ABBREVIATION:
            perturbed.append(_perturb_abbreviation(line, rng))
        elif ptype == PerturbationType.TIMESTAMP_FORMAT:
            perturbed.append(_perturb_timestamp(line, rng))
        elif ptype == PerturbationType.MISSING_FIELD:
            perturbed.append(_perturb_missing_field(line, rng))
        elif ptype == PerturbationType.CASE_VARIATION:
            perturbed.append(_perturb_case_variation(line, rng))
        elif ptype == PerturbationType.WRAPPING_NOISE:
            perturbed.append(_perturb_wrapping_noise(line, rng))
        elif ptype == PerturbationType.SYNONYM_SUBSTITUTION:
            perturbed.append(_perturb_synonym_substitution(line, rng))
        elif ptype == PerturbationType.ENCODING_NOISE:
            perturbed.append(_perturb_encoding_noise(line, rng))

        applied_types.append(ptype)

    return perturbed, applied_types


def _perturb_whitespace(line: str, rng: random.Random) -> str:
    """Replace consistent spacing with irregular whitespace."""
    parts = line.split()
    if len(parts) < 2:
        return line
    result = parts[0]
    for part in parts[1:]:
        sep = rng.choice([" ", "  ", "\t", " \t ", "   ", " "])
        result += sep + part
    return result


def _perturb_field_reorder(line: str, rng: random.Random) -> str:
    """Reorder fields while preserving content."""
    parts = line.split()
    if len(parts) < 4:
        return line
    n_keep = max(2, len(parts) // 3)
    head = parts[:n_keep]
    tail = parts[n_keep:]
    rng.shuffle(tail)
    return " ".join(head + tail)


def _perturb_separator_change(line: str, rng: random.Random) -> str:
    """Change field separators (| -> ; -> , -> :)."""
    separators = ["|", ";", ",", ":", "->"]
    for sep in ["|", "\t"]:
        if sep in line:
            return line.replace(sep, rng.choice(separators))
    parts = line.split()
    if len(parts) > 2:
        sep = rng.choice(separators)
        return parts[0] + sep + sep.join(parts[1:])
    return line


def _perturb_abbreviation(line: str, rng: random.Random) -> str:
    """Replace full words with common abbreviations."""
    result = line
    for full, abbrev in ABBREVIATIONS.items():
        if full in result and rng.random() > 0.3:
            result = result.replace(full, abbrev, 1)
    return result


def _perturb_timestamp(line: str, rng: random.Random) -> str:
    """Change timestamp format."""
    for old_fmt, new_fmt in TIMESTAMP_FORMATS:
        if old_fmt in line:
            return line.replace(old_fmt, new_fmt, 1)
    ts_match = re.search(r"(\d{10,})", line)
    if ts_match:
        import datetime
        try:
            ts = int(ts_match.group(1))
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            new_ts = rng.choice([
                dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                dt.strftime("%Y/%m/%d %H:%M:%S"),
                dt.strftime("%b %d %H:%M:%S"),
            ])
            return line[:ts_match.start()] + new_ts + line[ts_match.end():]
        except (ValueError, OSError):
            pass
    return line


def _perturb_missing_field(line: str, rng: random.Random) -> str:
    """Remove a field from the log line."""
    parts = line.split()
    if len(parts) < 4:
        return line
    drop_idx = rng.randint(1, min(len(parts) - 2, 6))
    parts.pop(drop_idx)
    return " ".join(parts)


def _perturb_case_variation(line: str, rng: random.Random) -> str:
    """Change case of the log line."""
    style = rng.choice(["lower", "upper", "title", "mixed"])
    if style == "lower":
        return line.lower()
    elif style == "upper":
        return line.upper()
    elif style == "title":
        return line.title()
    else:
        return "".join(
            c.lower() if rng.random() > 0.5 else c.upper()
            for c in line
        )


def _perturb_wrapping_noise(line: str, rng: random.Random) -> str:
    """Add prefix/suffix noise around the log line."""
    prefixes = ["[DEBUG] ", ">> ", "### ", "", "[syslog] "]
    suffixes = [" [end]", " <<", " ###", "", " [ok]"]
    prefix = rng.choice(prefixes)
    suffix = rng.choice(suffixes)
    return prefix + line + suffix


def _perturb_synonym_substitution(line: str, rng: random.Random) -> str:
    """Substitute words with synonyms."""
    result = line
    for word, synonyms in SYNONYMS.items():
        if word in result and rng.random() > 0.5:
            replacement = rng.choice(list(synonyms))
            result = result.replace(word, replacement, 1)
    return result


def _perturb_encoding_noise(line: str, rng: random.Random) -> str:
    """Inject encoding-like noise (unicode lookalikes, stray bytes)."""
    replacements = {
        "a": "\u0430",  # Cyrillic a
        "e": "\u0435",  # Cyrillic e
        "o": "\u043e",  # Cyrillic o
        "p": "\u0440",  # Cyrillic p
        "c": "\u0441",  # Cyrillic c
        " ": "\u00a0",  # Non-breaking space
    }
    result = []
    for char in line:
        if char in replacements and rng.random() > 0.95:
            result.append(replacements[char])
        else:
            result.append(char)
    return "".join(result)
