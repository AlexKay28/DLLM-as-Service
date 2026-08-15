"""
Shared provenance helper for all experiment runners.

Every result JSON emitted by a runner under models/experiments/ must include
a provenance block produced by this module. The block records:

- The software stack (vLLM, torch, transformers, CUDA driver, GPU model)
- The model actually loaded (not an argparse default — see #57 R1)
- The parser policy used (see #57 R5)
- The scoring fields used (9-field vs 12-field, see #57 R4)

Usage:
    from models.baselines.provenance import Provenance, capture_provenance

    # At the point where the model is actually loaded:
    prov = capture_provenance(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        model_type="ar3b",
        backend="vllm",
        max_new_tokens=256,
        parser_policy="first_object_wins",
        scoring_fields="9",
    )

    # In the result dict:
    result["provenance"] = prov.to_dict()

Design principle (from #57):
    "A runner must not be able to name a model it did not run."
    The provenance is emitted from the object that was constructed, not from
    an argparse default.
"""

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


def _try_import_version(mod_name: str) -> Optional[str]:
    """Try to import a module and return its __version__, or None."""
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", None)
    except ImportError:
        return None


def _nvidia_smi_gpu() -> Optional[str]:
    """Get GPU model name via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode().strip()
        return out.split("\n")[0] if out else None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _cuda_version() -> Optional[str]:
    """Get CUDA version from torch if available."""
    try:
        import torch
        return torch.version.cuda
    except (ImportError, AttributeError):
        return None


def _torch_version() -> Optional[str]:
    return _try_import_version("torch")


def _transformers_version() -> Optional[str]:
    return _try_import_version("transformers")


def _vllm_version() -> Optional[str]:
    return _try_import_version("vllm")


@dataclass
class Provenance:
    """Provenance block emitted by every runner into its result JSON."""

    model_name: str
    model_type: str
    backend: str
    max_new_tokens: int
    parser_policy: str = "first_object_wins"
    scoring_fields: str = "9"
    vllm_version: Optional[str] = None
    torch_version: Optional[str] = None
    transformers_version: Optional[str] = None
    cuda_version: Optional[str] = None
    gpu_model: Optional[str] = None
    python_version: str = platform.python_version()
    platform: str = platform.platform()
    timestamp: str = ""
    repo_commit: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self.vllm_version is None and self.backend == "vllm":
            self.vllm_version = _vllm_version()
        if self.torch_version is None:
            self.torch_version = _torch_version()
        if self.transformers_version is None:
            self.transformers_version = _transformers_version()
        if self.cuda_version is None:
            self.cuda_version = _cuda_version()
        if self.gpu_model is None:
            self.gpu_model = _nvidia_smi_gpu()
        if self.repo_commit is None:
            self.repo_commit = _git_commit()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def capture_provenance(
    model_name: str,
    model_type: str,
    backend: str = "vllm",
    max_new_tokens: int = 256,
    parser_policy: str = "first_object_wins",
    scoring_fields: str = "9",
    **extra,
) -> Provenance:
    """Capture provenance at the point where the model is actually loaded.

    This is the single entry point all runners should call. It records the
    model that was *actually constructed*, not an argparse default.

    Args:
        model_name: the HuggingFace model ID that was loaded (e.g.
            "Qwen/Qwen2.5-3B-Instruct"). Must come from the loaded object,
            not from argparse.
        model_type: short label ("ar3b", "dm_hf", "dg", etc.).
        backend: "vllm", "hf_transformers", "dinfer", or "mock".
        max_new_tokens: max output tokens configured.
        parser_policy: "first_object_wins" (primary, per #43) or "strict".
        scoring_fields: "9" (7 structural + 2 anomaly) or "12" (all fields).

    Returns:
        Provenance dataclass ready for .to_dict() or .to_json().
    """
    prov = Provenance(
        model_name=model_name,
        model_type=model_type,
        backend=backend,
        max_new_tokens=max_new_tokens,
        parser_policy=parser_policy,
        scoring_fields=scoring_fields,
    )
    for k, v in extra.items():
        setattr(prov, k, v)
    return prov


def assert_provenance(result_dict: dict) -> None:
    """Assert that a result dict contains a valid provenance block.

    Called by paper/verify_numbers.py to fail when a cited artifact
    lacks provenance (#57 R7 acceptance criterion).
    """
    prov = result_dict.get("provenance")
    if prov is None:
        raise AssertionError(
            f"Result artifact '{result_dict.get('experiment', '?')}' "
            f"lacks a provenance block (#57 R1)."
        )
    required = {"model_name", "model_type", "backend", "vllm_version",
                "torch_version", "gpu_model"}
    missing = required - set(prov.keys())
    if missing:
        raise AssertionError(
            f"Provenance block in '{result_dict.get('experiment', '?')}' "
            f"missing fields: {missing}"
        )
    # Check model/model_type consistency (#57 amended R1)
    model = result_dict.get("model", "")
    model_type = result_dict.get("model_type", "")
    if model and prov["model_name"] and model != prov["model_name"]:
        # model field may be a short name; check if model_type disambiguates
        if model_type and prov["model_type"] and model_type != prov["model_type"]:
            raise AssertionError(
                f"Provenance mismatch: result 'model'={model!r} / 'model_type'="
                f"{model_type!r} but provenance says model_name="
                f"{prov['model_name']!r} / model_type={prov['model_type']!r}. "
                f"The runner may be recording an argparse default instead of "
                f"the model actually loaded (see #57 R1)."
            )


STACK_LABELS = {
    # Campaign D (#3-#8): HF transformers, no vLLM
    ("hf_transformers", None): "D",
    # Campaign A (#12): vLLM 0.19.1, H100
    ("vllm", "0.19.1"): "A",
    # Campaign B (#14-#21): vLLM 0.25.1, A100
    ("vllm", "0.25.1"): "B",
    # Campaign C (#22): dInfer + vLLM 0.10.2
    ("dinfer", "0.10.2"): "C",
    # Campaign new (#45+): vLLM 0.7.2, A100
    ("vllm", "0.7.2"): "new",
}


def stack_label(prov: Provenance | dict) -> str:
    """Map a provenance block to a campaign label (A/B/C/D/new).

    This identifies which of the five historical stacks produced an artifact.
    """
    if isinstance(prov, Provenance):
        prov = prov.to_dict()
    backend = prov.get("backend", "")
    vllm_ver = prov.get("vllm_version")
    key = (backend, vllm_ver)
    if key in STACK_LABELS:
        return STACK_LABELS[key]
    if backend == "vllm" and vllm_ver:
        return f"vllm-{vllm_ver}"
    if backend == "dinfer":
        return "C"
    if backend == "hf_transformers":
        return "D"
    return "unknown"


# ---- 9-field and 12-field definitions ----

# Per #57 R4: score on 9 fields (7 structural + 2 anomaly).
# Drop event_type, affected_entity, remediation_hint until #53's gold
# problem is fixed. Emit the 12-field macro alongside for continuity.

NINE_FIELD_NAMES = [
    # 7 structural
    "timestamp_norm",
    "host",
    "component",
    "severity",
    "message_template",
    "error_code",
    "metric_value",
    # 2 anomaly
    "is_anomaly",
    "anomaly_score",
]

TWELVE_FIELD_NAMES = [
    "timestamp_norm",
    "host",
    "component",
    "severity",
    "event_type",
    "message_template",
    "error_code",
    "affected_entity",
    "metric_value",
    "is_anomaly",
    "anomaly_score",
    "remediation_hint",
]

# Fields dropped from the 9-field score and why
DROPPED_FIELDS = {
    "event_type": "gold has near-zero variance (#53: 2 unique values across 2000 events)",
    "affected_entity": "gold has near-zero variance (#53: 1 unique value)",
    "remediation_hint": "gold has near-zero variance (#53: 1 unique value); a model emitting {} scores 1.000",
}
