"""
AR model wrapper for log-extraction experiments.

Loads an open AR model (Qwen2.5-1.5B, Qwen2.5-7B, or any HuggingFace model),
runs inference on raw log lines, and returns structured Records with per-event
latency measurements.

This is the AR-side baseline that replaces the paper's unreleased AR-1.5B /
AR-7B models. Used by issues #3 (quality), #4 (latency), and #5 (throughput).

Usage:
    from models.baselines.ar_wrapper import ARModel

    # Load model (GPU required)
    model = ARModel("Qwen/Qwen2.5-1.5B-Instruct")
    result = model.extract(log_lines, source="bgl")

    # result.records       — list[Record]
    # result.latencies_ms  — list[float] per-event wall-clock latency
    # result.token_counts  — list[int] generated tokens per event
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES, FIELD_BY_NAME, FieldType, FieldCategory
from models.baselines.ar_model import build_chat_messages, parse_json_output, SYSTEM_PROMPT


@dataclass
class ARResult:
    """Result of AR model extraction."""
    records: list[Record]
    latencies_ms: list[float]
    token_counts: list[int]
    model_name: str
    n_input_tokens: int = 0  # total input tokens consumed
    n_output_tokens: int = 0  # total output tokens generated


def _coerce_types(rec: Record) -> Record:
    """Coerce string values to the schema's declared types so predictions
    match gold records (which are coerced in prepare.py)."""
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


def _dict_to_record(data: dict) -> Record:
    """Convert a parsed JSON dict to a schema-compliant Record."""
    rec = Record(fields={})
    for name in FIELD_NAMES:
        if name in data:
            rec.set(name, data[name])
        else:
            # Fill defaults
            sf = FIELD_BY_NAME[name]
            if sf.type == FieldType.FLOAT:
                rec.set(name, 0.0)
            elif sf.type == FieldType.BOOL:
                rec.set(name, False)
            else:
                rec.set(name, "")
    return _coerce_types(rec)


class ARModel:
    """AR model wrapper for log extraction.

    Loads a HuggingFace model and runs inference with per-event timing.
    Supports both chat-formatted (instruct) and raw-text (base) models.

    Args:
        model_name: HuggingFace model ID (e.g. "Qwen/Qwen2.5-1.5B-Instruct")
        device: "cuda" (default), "cpu", or specific device ("cuda:0")
        dtype: "auto" (default), "float16", "bfloat16", "float32"
        max_new_tokens: max tokens to generate per event (default 256)
        grammar_constrained: if True, use XGrammar for JSON-constrained decoding
        batch_size: batch size for inference (default 1 = streaming mode m=1)
        repetition_penalty: must stay 1.0 (neutral) for this task. Qwen2.5's
            `generation_config.json` ships 1.1 (1.5B) / 1.05 (3B, 7B), and HF
            `generate()` applies it even under `do_sample=False` — so leaving it
            unset silently penalises exactly what extraction requires: copying
            spans out of the input. Measured on 100 BGL events, the model's own
            default emptied the `host` field on 80/100 events (F1 0.20 vs 0.83)
            and cost 0.075 macro Field-F1. vLLM defaults it to 1.0, which is
            why the two stacks disagreed until this was made explicit.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        grammar_constrained: bool = False,
        batch_size: int = 1,
        repetition_penalty: float = 1.0,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.grammar_constrained = grammar_constrained
        self.batch_size = batch_size
        self.repetition_penalty = repetition_penalty

        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    def load(self):
        """Load the model and tokenizer. Must be called before extract()."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, "auto")

        print(f"Loading {self.model_name} (device={self.device}, dtype={self.dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        self._is_loaded = True
        print(f"  loaded. {self.model.num_parameters():,} params")

    def _generate_single(self, log_line: str, source: str) -> tuple[str, int, int]:
        """Generate for a single log line. Returns (output_text, input_tokens, output_tokens)."""
        import torch

        messages = build_chat_messages(log_line, source=source, few_shot=True)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        output_tokens = output[0][input_len:]
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return output_text, input_len, len(output_tokens)

    def _generate_batch(self, log_lines: list[str], source: str
                        ) -> list[tuple[str, int, int]]:
        """Generate for a batch of log lines."""
        import torch

        # Build prompts
        all_texts = []
        for line in log_lines:
            messages = build_chat_messages(line, source=source, few_shot=True)
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            all_texts.append(text)

        # Tokenize with padding
        inputs = self.tokenizer(
            all_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=2048).to(self.model.device)
        input_lens = inputs["attention_mask"].sum(dim=1)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        results = []
        for i in range(len(log_lines)):
            input_len = input_lens[i].item()
            output_tokens = outputs[i][input_len:]
            output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
            results.append((output_text, input_len, len(output_tokens)))
        return results

    def extract(self, log_lines: list[str], source: str = "hdfs",
                show_progress: bool = True) -> ARResult:
        """Extract structured records from raw log lines.

        Args:
            log_lines: raw log lines to process.
            source: log source (hdfs, bgl, thunderbird, spirit).
            show_progress: print progress.

        Returns:
            ARResult with records, latencies_ms, token_counts.
        """
        if not self._is_loaded:
            self.load()

        records = []
        latencies = []
        token_counts = []
        total_input_tokens = 0
        total_output_tokens = 0

        n = len(log_lines)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, n, self.batch_size),
                                desc=f"  {self.model_name}", unit="batch")
            except ImportError:
                iterator = range(0, n, self.batch_size)
        else:
            iterator = range(0, n, self.batch_size)

        for start in iterator:
            batch = log_lines[start:start + self.batch_size]
            t0 = time.perf_counter()

            if self.batch_size == 1:
                output_text, in_tok, out_tok = self._generate_single(batch[0], source)
                results = [(output_text, in_tok, out_tok)]
            else:
                results = self._generate_batch(batch, source)

            elapsed = (time.perf_counter() - t0) * 1000

            for i, (output_text, in_tok, out_tok) in enumerate(results):
                data = parse_json_output(output_text)
                rec = _dict_to_record(data)
                records.append(rec)

                # Per-event latency: total batch time / batch size
                latencies.append(elapsed / len(batch))
                token_counts.append(out_tok)
                total_input_tokens += in_tok
                total_output_tokens += out_tok

            if show_progress and hasattr(iterator, 'set_postfix'):
                avg_lat = sum(latencies) / len(latencies)
                iterator.set_postfix(avg_ms=f"{avg_lat:.1f}")

        return ARResult(
            records=records,
            latencies_ms=latencies,
            token_counts=token_counts,
            model_name=self.model_name,
            n_input_tokens=total_input_tokens,
            n_output_tokens=total_output_tokens,
        )


class MockARModel:
    """Mock AR model for testing the experiment harness without GPU.

    Produces deterministic records using the same fallback parser as
    drain_only.py — useful for verifying the experiment pipeline end-to-end
    before running on real GPU models.
    """

    def __init__(self, model_name: str = "mock-ar-1.5b"):
        self.model_name = model_name
        self._is_loaded = True

    def load(self):
        pass  # no-op

    def extract(self, log_lines: list[str], source: str = "hdfs",
                show_progress: bool = True) -> ARResult:
        """Simulate AR extraction using regex fallback (for testing only)."""
        from models.baselines.drain_only import extract as drain_extract

        # Use drain fallback as a stand-in for AR output
        dr_result = drain_extract(log_lines, source=source)

        # Simulate AR latency: ~168ms per event (paper's AR-1.5B p50)
        import random
        rng = random.Random(42)
        latencies = [168.0 + rng.gauss(0, 30) for _ in log_lines]
        token_counts = [128] * len(log_lines)  # paper's AR step count

        return ARResult(
            records=dr_result.records,
            latencies_ms=latencies,
            token_counts=token_counts,
            model_name=self.model_name,
            n_input_tokens=len(log_lines) * 300,  # approx input tokens
            n_output_tokens=len(log_lines) * 128,
        )
