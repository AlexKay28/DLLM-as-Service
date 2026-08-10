"""
MDM (Masked-Diffusion Model) wrapper for log-extraction experiments.

This is the FlowFill-side wrapper: it loads a masked-diffusion LM and runs
parallel denoising to produce the same 12-field Records as the AR wrapper,
but in K steps instead of ~128.

The key difference from AR:
  - AR generates tokens left-to-right (128 serial steps)
  - MDM denoises all K mask tokens in parallel (K steps, K << 128)

Open MDM substitutes for FlowFill-MDM (weights not released):
  - LLaDA-8B (https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
  - Dream (https://huggingface.co/Dream-org/Dream-7B-Instruct)

Usage (GPU required):
    from models.baselines.mdm_wrapper import MDMModel

    model = MDMModel("GSAI-ML/LLaDA-8B-Instruct", k_steps=8)
    result = model.extract(log_lines, source="bgl")

    # With Semi-AR blocks (Table 5 ablation)
    result = model.extract_with_blocks(log_lines, source="bgl", blocks=2)

    # K-sweep (Table 5 / K=8 knee)
    result = model.extract_with_k(log_lines, source="bgl", k_steps=4)
"""

import json
import sys
import time
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES, FIELD_BY_NAME, FieldType, FieldCategory
from models.baselines.ar_model import build_chat_messages, parse_json_output

# Number of denoising steps. Paper's knee is K=8.
DEFAULT_K_STEPS = 8

# Paper's Table 3 numbers for FlowFill-MDM (for comparison / mock simulation)
PAPER_FLOWFILL = {
    "p50_ms": 23, "p99_ms": 31, "p999_ms": 36,
    "j_per_event": 0.7, "steps": 8,
    "field_f1": 0.907, "schema_valid_pct": 99.4,
    "consistency_pct": 97.8, "anomaly_f1": 0.865,
}

# Approximate per-step cost: paper's K=8 → 23ms → ~2.9ms/step
MS_PER_STEP = 23 / 8


@dataclass
class MDMResult:
    """Result of MDM extraction. Same shape as ARResult for compatibility."""
    records: list[Record]
    latencies_ms: list[float]
    token_counts: list[int]
    model_name: str
    k_steps: int = DEFAULT_K_STEPS
    blocks: int = 1
    n_input_tokens: int = 0
    n_output_tokens: int = 0


def _coerce_types(rec: Record) -> Record:
    """Coerce string values to the schema's declared types."""
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
            sf = FIELD_BY_NAME[name]
            if sf.type == FieldType.FLOAT:
                rec.set(name, 0.0)
            elif sf.type == FieldType.BOOL:
                rec.set(name, False)
            else:
                rec.set(name, "")
    return _coerce_types(rec)


class MDMModel:
    """Masked-Diffusion Model wrapper for log extraction.

    Loads an open MDM (LLaDA, Dream) and runs K-step parallel denoising
    to produce 12-field Records.

    The denoising process:
      1. Build the prompt with [MASK] tokens for all 12 field values.
      2. In each of K steps, unmask a subset of masked tokens in parallel.
      3. After K steps, parse the fully-unmasked output as JSON → Record.

    Args:
        model_name: HuggingFace model ID (e.g. "GSAI-ML/LLaDA-8B-Instruct")
        k_steps: number of denoising steps (default 8, paper's knee)
        device: "cuda" or "cpu"
        dtype: "auto", "float16", "bfloat16", "float32"
        max_new_tokens: max output tokens per record (~128 for 12-field schema)
    """

    def __init__(
        self,
        model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
        k_steps: int = DEFAULT_K_STEPS,
        device: str = "cuda",
        dtype: str = "auto",
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.k_steps = k_steps
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    def load(self):
        """Load the MDM model and tokenizer.

        TODO (GPU): The exact loading API depends on which MDM we use.
        LLaDA and Dream have different interfaces — both need transformers
        support or custom loaders. See integration notes below.
        """
        # --- LLaDA integration (primary substitute) ---
        # LLaDA-8B uses a custom inference loop (not standard HF generate).
        # See: https://github.com/ML-GSAI/LLaDA
        #
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        # import torch
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name, torch_dtype=torch.bfloat16,
        #     device_map=self.device, trust_remote_code=True)
        # self.model.eval()
        # self._is_loaded = True
        #
        # LLaDA's denoising is done via model.forward() + iterative unmasking,
        # NOT model.generate(). The key method is:
        #   logits = model(input_ids)  # [B, T, V]
        # Then sample tokens at mask positions, repeat K times.
        #
        # --- Dream integration (alternative substitute) ---
        # Dream-7B has a similar masked-diffusion interface.
        # See: https://github.com/bytedance/Dream
        #
        # --- FlowFill-MDM (the actual paper model) ---
        # Weights not released. If they become available, load here.

        raise NotImplementedError(
            "MDM model loading not yet implemented. "
            "See load() docstring for LLaDA/Dream integration notes. "
            "Use MockMDMModel for testing.")

    def _denoise(self, input_ids, k_steps: int, blocks: int = 1):
        """Run K-step parallel denoising on masked input.

        This is the core MDM inference loop. In each step:
          1. Forward pass: get logits at all masked positions.
          2. Sample tokens at a subset of mask positions.
          3. Unmask those positions (replace [MASK] with sampled tokens).
          4. Repeat until K steps done or all positions unmasked.

        Semi-AR blocks (Table 5):
          - B=1: all positions denoised in parallel (pure diffusion)
          - B=2: first half denoised, then second half (2 sequential passes)
          - B=4: 4 sequential quarters
          Each block does K/B steps of parallel denoising internally.

        TODO (GPU): implement using the specific MDM's forward() + mask API.
        """
        # Pseudocode:
        # mask_positions = (input_ids == self.tokenizer.mask_token_id)
        # if blocks > 1:
        #     # Split mask positions into B contiguous blocks
        #     block_chunks = split_blocks(mask_positions, blocks)
        #     for chunk in block_chunks:
        #         for step in range(k_steps // blocks):
        #             logits = self.model(input_ids)
        #             # Sample at chunk positions only
        #             sampled = sample(logits, chunk)
        #             input_ids[chunk] = sampled
        # else:
        #     for step in range(k_steps):
        #         logits = self.model(input_ids)
        #         n_to_unmask = max(1, mask_positions.sum() // (k_steps - step))
        #         sampled = sample(logits, n_to_unmask)
        #         input_ids[unmask_positions] = sampled
        #
        # return input_ids

        raise NotImplementedError("MDM denoising loop not yet implemented.")

    def _generate_single(self, log_line: str, source: str,
                         k_steps: int = None, blocks: int = 1
                         ) -> tuple[str, int, int]:
        """Generate for a single log line via masked denoising.

        Returns (output_text, input_tokens, output_tokens).
        """
        k = k_steps or self.k_steps

        # Build prompt — same format as AR, but the model will fill in masks
        messages = build_chat_messages(log_line, source=source, few_shot=True)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        # Append masked output template (the 12-field JSON with [MASK] tokens)
        # The model denoises this template in K steps.
        masked_output = self._build_masked_output()
        full_input = text + masked_output

        input_ids = self.tokenizer(full_input, return_tensors="pt").to(self.model.device)
        input_len = input_ids["input_ids"].shape[1]

        # Run denoising
        output_ids = self._denoise(input_ids["input_ids"], k_steps=k, blocks=blocks)

        output_text = self.tokenizer.decode(output_ids[0][input_len:],
                                            skip_special_tokens=True)
        return output_text, input_len, len(output_ids[0][input_len:])

    def _build_masked_output(self) -> str:
        """Build the masked JSON template that the MDM will denoise.

        The template is a JSON skeleton with [MASK] tokens for each field value.
        The model's job is to fill in the masks.
        """
        fields_json = {}
        for name in FIELD_NAMES:
            fields_json[name] = "[MASK]"
        return "\n" + json.dumps(fields_json, indent=2)

    def extract(self, log_lines: list[str], source: str = "hdfs",
                show_progress: bool = True,
                k_steps: int = None, blocks: int = 1) -> MDMResult:
        """Extract structured records via K-step parallel denoising.

        Args:
            log_lines: raw log lines.
            source: log source.
            show_progress: print progress.
            k_steps: override default K (for K-sweep ablation).
            blocks: Semi-AR blocks (1=parallel, 2/4=sequential blocks).

        Returns:
            MDMResult with records, latencies_ms, token_counts.
        """
        if not self._is_loaded:
            self.load()

        k = k_steps or self.k_steps
        records = []
        latencies = []
        token_counts = []
        total_input_tokens = 0
        total_output_tokens = 0

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(log_lines, desc=f"  {self.model_name} (K={k})", unit="ev")
            except ImportError:
                iterator = log_lines
        else:
            iterator = log_lines

        for line in iterator:
            t0 = time.perf_counter()
            output_text, in_tok, out_tok = self._generate_single(
                line, source, k_steps=k, blocks=blocks)
            elapsed = (time.perf_counter() - t0) * 1000

            data = parse_json_output(output_text)
            rec = _dict_to_record(data)
            records.append(rec)
            latencies.append(elapsed)
            token_counts.append(out_tok)
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            if show_progress and hasattr(iterator, 'set_postfix'):
                avg_lat = sum(latencies) / len(latencies)
                iterator.set_postfix(avg_ms=f"{avg_lat:.1f}")

        return MDMResult(
            records=records,
            latencies_ms=latencies,
            token_counts=token_counts,
            model_name=self.model_name,
            k_steps=k,
            blocks=blocks,
            n_input_tokens=total_input_tokens,
            n_output_tokens=total_output_tokens,
        )

    def extract_with_blocks(self, log_lines: list[str], source: str = "hdfs",
                            blocks: int = 1, show_progress: bool = True) -> MDMResult:
        """Semi-AR block denoising (Table 5 ablation).

        B=1: pure parallel (all 12 fields denoised simultaneously)
        B=2: first 6 fields, then next 6 (2 sequential passes)
        B=4: 3 fields per block (4 sequential passes)

        Each block does K/B parallel denoising steps internally.
        """
        return self.extract(log_lines, source=source, show_progress=show_progress,
                           k_steps=self.k_steps, blocks=blocks)

    def extract_with_k(self, log_lines: list[str], source: str = "hdfs",
                       k_steps: int = DEFAULT_K_STEPS,
                       show_progress: bool = True) -> MDMResult:
        """K-step sweep (Table 5 / K=8 knee ablation)."""
        return self.extract(log_lines, source=source, show_progress=show_progress,
                           k_steps=k_steps, blocks=1)


class MockMDMModel:
    """Mock MDM model for testing the experiment harness without GPU.

    Simulates FlowFill-MDM's behavior:
      - K-step latency: p50 ≈ K × 2.9ms (paper's ~23ms at K=8)
      - F1: near paper's 0.907 (using drain fallback as stand-in)
      - Content-independent latency (same K regardless of input length)

    This lets us verify the experiment pipeline end-to-end, including
    the ablation runners that call extract_with_blocks / extract_with_k.
    """

    def __init__(self, model_name: str = "mock-flowfill-mdm",
                 k_steps: int = DEFAULT_K_STEPS):
        self.model_name = model_name
        self.k_steps = k_steps
        self._is_loaded = True
        self.max_new_tokens = 256
        self.batch_size = 1

    def load(self):
        pass

    def extract(self, log_lines: list[str], source: str = "hdfs",
                show_progress: bool = True,
                k_steps: int = None, blocks: int = 1) -> MDMResult:
        """Simulate MDM extraction with paper-like latency characteristics."""
        from models.baselines.drain_only import extract as drain_extract

        k = k_steps or self.k_steps
        # Use drain fallback as a stand-in for MDM output (same as MockARModel)
        dr_result = drain_extract(log_lines, source=source)

        # Simulate MDM latency: K steps × ~2.9ms/step
        # B>1 adds sequential passes: total_steps = K, but in B chunks
        # so latency = K * ms_per_step (content-independent)
        # Add small jitter for realism
        rng = random.Random(42)
        ms_per_step = MS_PER_STEP
        latencies = []
        for _ in log_lines:
            base = k * ms_per_step
            # B>1 adds a small sequential overhead per block
            if blocks > 1:
                base += (blocks - 1) * 1.5  # ~1.5ms per extra block boundary
            jitter = rng.gauss(0, base * 0.08)
            latencies.append(max(base + jitter, 0.5))

        # Token count: MDM produces fixed-length output (~128 tokens)
        # regardless of K (K controls denoising steps, not output length)
        token_counts = [128] * len(log_lines)

        return MDMResult(
            records=dr_result.records,
            latencies_ms=latencies,
            token_counts=token_counts,
            model_name=self.model_name,
            k_steps=k,
            blocks=blocks,
            n_input_tokens=len(log_lines) * 300,
            n_output_tokens=len(log_lines) * 128,
        )

    def extract_with_blocks(self, log_lines: list[str], source: str = "hdfs",
                            blocks: int = 1, show_progress: bool = True) -> MDMResult:
        return self.extract(log_lines, source=source, show_progress=show_progress,
                           k_steps=self.k_steps, blocks=blocks)

    def extract_with_k(self, log_lines: list[str], source: str = "hdfs",
                       k_steps: int = DEFAULT_K_STEPS,
                       show_progress: bool = True) -> MDMResult:
        return self.extract(log_lines, source=source, show_progress=show_progress,
                           k_steps=k_steps, blocks=1)


# ---------------------------------------------------------------------------
# Integration notes for tomorrow's GPU work
# ---------------------------------------------------------------------------
#
# 1. LLaDA-8B (primary MDM substitute)
#    - Repo: https://github.com/ML-GSAI/LLaDA
#    - HF: https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct
#    - Loading: AutoModelForCausalLM + trust_remote_code=True
#    - Inference: NOT model.generate(). LLaDA uses a custom denoising loop:
#        - Input has [MASK] tokens at generation positions
#        - Each step: forward pass → sample at masked positions → unmask subset
#        - K steps total
#    - The denoising loop is in LLaDA's repo: llada/generate.py
#    - Key function: generate(model, prompt, steps=K, ...)
#
# 2. Dream-7B (alternative MDM substitute)
#    - Repo: https://github.com/bytedance/Dream
#    - HF: https://huggingface.co/Dream-org/Dream-7B-Instruct
#    - Similar masked-diffusion interface
#
# 3. FlowFill-MDM (the actual paper model)
#    - Weights NOT released (preprint under review)
#    - If released, load here and the denoising loop should match the paper's
#
# 4. Semi-AR blocks (Table 5)
#    - Block splitting: divide the 12 field positions into B contiguous groups
#    - Denoise block 1 fully, then block 2, etc. (sequential across blocks)
#    - Within each block: parallel denoising (K/B steps per block)
#    - Total latency: B × (K/B) × ms_per_step = K × ms_per_step (same as B=1)
#      BUT there's cross-block conditioning overhead → paper shows +6ms (B=2), +18ms (B=4)
#
# 5. K-sweep (K=8 knee)
#    - Run with K ∈ {2, 4, 6, 8, 10, 12, 16, 24}
#    - Quality should saturate around K=8
#    - Latency should be linear: p50 ≈ K × 2.9ms
