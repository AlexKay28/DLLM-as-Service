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
        """Load the MDM model and tokenizer (LLaDA-8B-Instruct).

        LLaDA's custom `modeling_llada.py` targets an older transformers
        internal API (~4.4x) and is incompatible with transformers>=5.x in
        two spots:
          1. `LLaDAModelLM` doesn't call `post_init()`, so it's missing the
             `all_tied_weights_keys` attribute that newer transformers'
             `get_total_byte_count()` expects during `from_pretrained()`.
          2. Its `tie_weights()` override has signature `(self)`, but newer
             transformers calls it with `missing_keys=`/`recompute_mapping=`
             kwargs.
        We monkey-patch both call sites narrowly (no behavior change for
        other models) to make loading work without modifying the cached
        model file or pinning an old transformers version (which would
        break the AR wrapper's torch.compile / newer-API usage).
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import transformers.modeling_utils as _mu
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        import torch

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, "auto")

        # --- Patch 1: get_total_byte_count tolerates missing all_tied_weights_keys ---
        # Only applies to transformers>=5.x where get_total_byte_count exists.
        if hasattr(_mu, "get_total_byte_count") and not getattr(_mu, "_llada_patch_gtbc_applied", False):
            _orig_gtbc = _mu.get_total_byte_count

            def _patched_gtbc(model, accelerator_device_map, hf_quantizer=None):
                if not hasattr(model, "all_tied_weights_keys"):
                    model.all_tied_weights_keys = {}
                return _orig_gtbc(model, accelerator_device_map, hf_quantizer)

            _mu.get_total_byte_count = _patched_gtbc
            _mu._llada_patch_gtbc_applied = True

        print(f"Loading {self.model_name} (device={self.device}, dtype={self.dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, local_files_only=True)

        # --- Patch 2: LLaDAModelLM.tie_weights() accept-and-ignore new kwargs ---
        try:
            LLaDAModelLM = get_class_from_dynamic_module(
                "modeling_llada.LLaDAModelLM", self.model_name,
                local_files_only=True)
            if not getattr(LLaDAModelLM, "_tie_weights_patched", False):
                _orig_tie = LLaDAModelLM.tie_weights

                def _patched_tie(self_model, *args, **kwargs):
                    return _orig_tie(self_model)

                LLaDAModelLM.tie_weights = _patched_tie
                LLaDAModelLM._tie_weights_patched = True
        except Exception as e:
            print(f"  warning: could not patch LLaDAModelLM.tie_weights: {e}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

        # LLaDA mask token: <|mdm_mask|> = token id 126336
        self.mask_token_id = 126336
        self._is_loaded = True
        print(f"  loaded. {self.model.num_parameters():,} params, "
              f"mask_token_id={self.mask_token_id}")

    def _denoise(self, input_ids, k_steps: int, blocks: int = 1):
        """Run K-step parallel denoising on masked input.

        Confidence-based unmasking (paper §2.2):
          - Each step, forward pass → get logits at all masked positions
          - Pick the highest-confidence tokens (by max logit), unmask them
          - Number unmasked per step = max(1, remaining_masks // steps_left)
          - Easy fields (host, severity) resolve early; hard ones use budget

        Semi-AR blocks (Table 5):
          - B=1: all positions denoised in parallel
          - B>1: split mask positions into B contiguous chunks, denoise sequentially
          - Each block does K/B parallel denoising steps internally
        """
        import torch
        import torch.nn.functional as F

        device = input_ids.device
        mask_id = self.mask_token_id

        # Find all masked positions
        mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            return input_ids  # nothing to denoise

        # Split into blocks if B > 1
        if blocks > 1:
            block_chunks = torch.tensor_split(mask_positions, blocks)
        else:
            block_chunks = [mask_positions]

        steps_per_block = max(1, k_steps // blocks) if blocks > 1 else k_steps

        for chunk in block_chunks:
            if len(chunk) == 0:
                continue

            remaining = list(chunk.tolist())
            steps = steps_per_block if blocks > 1 else k_steps

            for step in range(steps):
                if not remaining:
                    break

                with torch.no_grad():
                    logits = self.model(input_ids, use_cache=False).logits  # [1, T, V]

                # Get logits at remaining masked positions
                pos_tensor = torch.tensor(remaining, device=device)
                masked_logits = logits[0, pos_tensor, :]  # [n_masked, V]
                masked_probs = F.softmax(masked_logits.float(), dim=-1)

                # Confidence = max softmax probability at each position
                confidences, predicted_ids = masked_probs.max(dim=-1)  # [n_masked]

                # Number to unmask this step
                steps_left = steps - step
                n_to_unmask = max(1, len(remaining) // steps_left)

                # Pick top-n most confident positions
                top_conf, top_idx = confidences.topk(min(n_to_unmask, len(remaining)))

                # Unmask those positions
                unmask_positions = pos_tensor[top_idx]
                unmask_tokens = predicted_ids[top_idx]

                input_ids[0, unmask_positions] = unmask_tokens

                # Remove unmasked positions from remaining list
                unmasked_set = set(unmask_positions.tolist())
                remaining = [p for p in remaining if p not in unmasked_set]

        return input_ids

    def _generate_single(self, log_line: str, source: str,
                         k_steps: int = None, blocks: int = 1
                         ) -> tuple[str, int, int]:
        """Generate for a single log line via masked denoising.

        Returns (output_text, input_tokens, output_tokens).
        """
        import torch

        import torch

        k = k_steps or self.k_steps

        # Build the instruction prompt (few-shot examples + query), same
        # content as the AR wrapper, so both substitutes see identical
        # context/instructions.
        messages = build_chat_messages(log_line, source=source, few_shot=True)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True,
            max_length=2048 - self.max_new_tokens)["input_ids"]
        prompt_len = prompt_ids.shape[1]

        # Append a contiguous block of mask tokens (the completion to denoise).
        # Concatenating token IDs directly avoids any ambiguity in how the
        # tokenizer would re-tokenize a literal "<|mdm_mask|>" string.
        n_gen = self.max_new_tokens
        mask_block = torch.full((1, n_gen), self.mask_token_id, dtype=prompt_ids.dtype)
        input_ids = torch.cat([prompt_ids, mask_block], dim=1).to(self.model.device)

        # Run denoising (mutates input_ids in place, filling masked positions)
        output_ids = self._denoise(input_ids, k_steps=k, blocks=blocks)

        # Decode only the part after the prompt (the denoised completion)
        gen_ids = output_ids[0][prompt_len:]
        output_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        n_output = int((gen_ids != self.mask_token_id).sum())
        return output_text, prompt_len, n_output

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

    def extract_impute(self, log_lines: list[str], known_fields_list: list[dict],
                       missing_fields_list: list[list[str]], source: str = "hdfs",
                       show_progress: bool = True,
                       k_steps: int = None) -> MDMResult:
        """Impute missing fields via selective masking (issue #23).

        Instead of masking the entire completion, this builds a partial JSON
        with known fields filled as real tokens and missing fields as mask
        tokens. _denoise fills only the masked positions, conditioning on the
        known fields — DM's native operation.

        Args:
            log_lines: raw log lines.
            known_fields_list: per-event dict of known field values.
            missing_fields_list: per-event list of field names to impute.
            source: log source.
            k_steps: override default K.
        """
        import torch

        from models.baselines.imputation import (
            build_dm_imputation_prompt_text,
            build_dm_imputation_token_ids,
        )

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
                iterator = tqdm(zip(log_lines, known_fields_list, missing_fields_list),
                               desc=f"  {self.model_name} (impute K={k})", unit="ev")
            except ImportError:
                iterator = zip(log_lines, known_fields_list, missing_fields_list)
        else:
            iterator = zip(log_lines, known_fields_list, missing_fields_list)

        for line, known, missing in iterator:
            prompt_text = build_dm_imputation_prompt_text(
                line, source, known, missing)

            input_ids, prompt_len, n_mask = build_dm_imputation_token_ids(
                self.tokenizer, prompt_text, known, missing,
                self.mask_token_id, max_length=2048)
            input_ids = input_ids.to(self.model.device)

            t0 = time.perf_counter()
            output_ids = self._denoise(input_ids, k_steps=k, blocks=1)
            elapsed = (time.perf_counter() - t0) * 1000

            gen_ids = output_ids[0][prompt_len:]
            output_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            n_output = int((gen_ids != self.mask_token_id).sum())

            data = parse_json_output(output_text)
            merged = dict(known)
            merged.update(data)
            rec = _dict_to_record(merged)
            records.append(rec)
            latencies.append(elapsed)
            token_counts.append(n_output)
            total_input_tokens += prompt_len
            total_output_tokens += n_output

            if show_progress and hasattr(iterator, 'set_postfix'):
                avg_lat = sum(latencies) / len(latencies)
                iterator.set_postfix(avg_ms=f"{avg_lat:.1f}")

        return MDMResult(
            records=records,
            latencies_ms=latencies,
            token_counts=token_counts,
            model_name=self.model_name,
            k_steps=k,
            blocks=1,
            n_input_tokens=total_input_tokens,
            n_output_tokens=total_output_tokens,
        )


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

    def extract_impute(self, log_lines: list[str], known_fields_list: list[dict],
                       missing_fields_list: list[list[str]], source: str = "hdfs",
                       show_progress: bool = True,
                       k_steps: int = None) -> MDMResult:
        """Mock imputation: drain extract + merge known fields."""
        from models.baselines.drain_only import extract as drain_extract

        k = k_steps or self.k_steps
        dr_result = drain_extract(log_lines, source=source)

        records = []
        for dr_rec, known, missing in zip(dr_result.records, known_fields_list, missing_fields_list):
            merged = dict(known)
            for fname in FIELD_NAMES:
                if fname not in merged or merged[fname] is None:
                    merged[fname] = dr_rec.get(fname)
            records.append(_dict_to_record(merged))

        rng = random.Random(42)
        latencies = []
        for missing in missing_fields_list:
            n_mask = len(missing)
            base = k * MS_PER_STEP * (0.3 + 0.1 * n_mask)
            jitter = rng.gauss(0, base * 0.08)
            latencies.append(max(base + jitter, 0.5))

        token_counts = [sum(len(str(v).split()) for v in known.values()) // 2 + 10]
        token_counts = [128] * len(log_lines)

        return MDMResult(
            records=records,
            latencies_ms=latencies,
            token_counts=token_counts,
            model_name=self.model_name,
            k_steps=k,
            blocks=1,
            n_input_tokens=len(log_lines) * 300,
            n_output_tokens=len(log_lines) * 64,
        )
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
