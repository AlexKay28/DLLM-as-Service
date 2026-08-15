#!/usr/bin/env python3
"""#6 / #58 P5: Long-context memory probe for AR-1.5B.

Measures true GPU memory demand (not vLLM pool) at various context lengths
with enforce_eager and low gpu_memory_utilization.

Usage:
    .venv-dg/bin/python models/experiments/run_memory_probe.py
"""

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.baselines.provenance import capture_provenance

OUT_DIR = _REPO_ROOT / "results"


def main():
    import torch
    from vllm import LLM, SamplingParams

    ctx_lengths = [1024, 4096, 16384, 32768]
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"=== Memory probe: {model_name} ===", flush=True)

    results = []
    for ctx_len in ctx_lengths:
        print(f"\n  ctx_len={ctx_len}...", flush=True)
        try:
            llm = LLM(
                model=model_name, dtype="bfloat16",
                gpu_memory_utilization=0.3,
                max_model_len=ctx_len + 256,
                enable_prefix_caching=False,
                enforce_eager=True,
            )
            tok = llm.get_tokenizer()
            sp = SamplingParams(max_tokens=1, temperature=0.0)

            # Build a prompt of ~ctx_len tokens
            prompt = "x " * ctx_len

            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            outs = llm.generate([prompt], sp, use_tqdm=False)
            elapsed = time.perf_counter() - t0

            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            mem_reserved = torch.cuda.max_memory_reserved() / 1e9

            result = {
                "model": model_name,
                "ctx_len": ctx_len,
                "peak_memory_gb": round(mem_peak, 2),
                "reserved_memory_gb": round(mem_reserved, 2),
                "latency_ms": round(elapsed * 1000, 1),
                "enforce_eager": True,
            }
            results.append(result)
            print(f"    peak={mem_peak:.2f} GB  reserved={mem_reserved:.2f} GB  "
                  f"latency={elapsed*1000:.0f}ms", flush=True)

            del llm
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            results.append({"model": model_name, "ctx_len": ctx_len, "error": str(e)})

    # Also measure DG memory for comparison
    print(f"\n  DG-26B at L=256 (for comparison)...", flush=True)
    try:
        llm = LLM(
            model="google/diffusiongemma-26B-A4B-it", dtype="bfloat16",
            gpu_memory_utilization=0.3,
            max_model_len=4096,
            enable_prefix_caching=True,
            enforce_eager=True,
            diffusion_config={"canvas_length": 256, "max_denoising_steps": 8},
        )
        torch.cuda.reset_peak_memory_stats()
        tok = llm.get_tokenizer()
        sp = SamplingParams(max_tokens=256, min_tokens=256, temperature=0.0)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": "Extract: 081109 203615 148 INFO dfs.DataNode: PacketResponder terminating"}],
            tokenize=False, add_generation_prompt=True)
        outs = llm.generate([prompt], sp, use_tqdm=False)
        dg_peak = torch.cuda.max_memory_allocated() / 1e9
        results.append({
            "model": "google/diffusiongemma-26B-A4B-it",
            "ctx_len": 1047,
            "peak_memory_gb": round(dg_peak, 2),
            "note": "DG at standard prompt, K=8, enforce_eager",
        })
        print(f"    DG peak={dg_peak:.2f} GB", flush=True)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"    DG FAILED: {e}", flush=True)

    prov = capture_provenance(
        model_name=model_name, model_type="ar1.5b", backend="vllm",
        max_new_tokens=1, parser_policy="first_object_wins", scoring_fields="9",
    )

    output = {
        "experiment": "issue58_p5_memory_probe",
        "results": results,
        "provenance": prov.to_dict(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = OUT_DIR / "issue58_memory_probe.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")

    print("\n=== Summary ===")
    for r in results:
        if "error" in r:
            print(f"  {r.get('model','?')} ctx={r.get('ctx_len','?')}: ERROR")
        else:
            print(f"  {r['model']} ctx={r.get('ctx_len','?')}: peak={r.get('peak_memory_gb','?')} GB")


if __name__ == "__main__":
    main()
