#!/usr/bin/env python3
"""Issue #115: Cost-shape generalization on HumanEval (code generation).

Tests whether the AR-line vs DG-staircase cost law holds on a second task:
code generation. HumanEval (164 problems) is ideal: output is long,
truncation is catastrophic (SyntaxError), and batch-1 is the natural mode.

Arms:
  DG: DiffusionGemma-26B-A4B-it, K ∈ {2,4,8,16}, canvas=256
  AR: Qwen3-30B-A3B (MoE-matched) and Qwen2.5-3B-Instruct (dense baseline)

Metrics: pass@1, code_validity, completeness, p50/p99 latency, mean tokens.

Usage:
    # DG baseline (K=8)
    .venv/bin/python models/experiments/run_humaneval.py --arm dg --k-steps 8

    # AR-30B baseline
    .venv/bin/python models/experiments/run_humaneval.py --arm ar30b

    # AR-3B dense baseline
    .venv/bin/python models/experiments/run_humaneval.py --arm ar3b

    # K-sweep on DG
    for K in 2 4 8 16; do
      .venv/bin/python models/experiments/run_humaneval.py --arm dg --k-steps $K
    done
"""

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
exec(open(str(REPO / "scripts" / "vllm_patch.py")).read())

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from models.baselines.provenance import capture_provenance

HUMANEVAL_PATH = REPO / "datasets" / "humaneval" / "HumanEval.jsonl"

PROMPT_TEMPLATE = """Complete the following Python function. Return ONLY the function body (everything after the docstring), or the complete function if you prefer. The code must be syntactically valid Python.

{prompt}

Your code:"""


def load_humaneval():
    problems = []
    with open(HUMANEVAL_PATH) as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def build_prompt(problem):
    return PROMPT_TEMPLATE.format(prompt=problem["prompt"])


def extract_code(generated_text, problem):
    """Extract the function from generated text."""
    text = generated_text.strip()
    # Find the last ```python ... ``` block (some models prepend prose)
    if "```" in text:
        parts = text.split("```")
        # Take the longest code block
        best = ""
        for i, part in enumerate(parts):
            if i == 0 or i == len(parts) - 1:
                continue  # skip text before first fence and after last
            if part.startswith("python"):
                part = part[6:]  # strip language tag
            part = part.strip()
            if len(part) > len(best):
                best = part
        if best:
            text = best
    # The prompt already contains the function signature + docstring.
    # If the model returns just the body, indent and prepend the prompt.
    if text.startswith("def ") or text.startswith("from ") or text.startswith("import "):
        return text
    # Indent the body to match the function body level
    indented = "\n".join("    " + line if line.strip() else line for line in text.split("\n"))
    return problem["prompt"] + indented


def is_valid_python(code):
    """Check if code parses as valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def has_return(code):
    """Check if the function body has a return statement."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                return True
    except SyntaxError:
        pass
    return False


def run_tests(problem, code, timeout=10):
    """Execute the code against the problem's test cases. Returns True if pass."""
    test_code = code + "\n\n" + problem["test"] + "\n\n"
    test_code += f"check({problem['entry_point']})\n"
    test_code += 'print("PASS")\n'

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=timeout,
            )
            return "PASS" in result.stdout
        except (subprocess.TimeoutExpired, Exception):
            return False
        finally:
            os.unlink(f.name)


def run_experiment(arm, k_steps=8, n=None, max_tokens=512, deadline_ms=None):
    from vllm import LLM, SamplingParams

    problems = load_humaneval()
    if n:
        problems = problems[:n]
    print(f"Loaded {len(problems)} HumanEval problems")

    # Deadline mode (#116): AR gets truncated max_tokens, DG always full canvas
    if deadline_ms is not None:
        ar_slope = 7.2  # ms/token (from issue #112)
        deadline_max_tokens = max(1, int(deadline_ms / ar_slope))
        if arm != "dg":
            max_tokens = deadline_max_tokens
            print(f"Deadline mode: {deadline_ms}ms -> AR max_tokens={max_tokens}")
        else:
            print(f"Deadline mode: {deadline_ms}ms -> DG full canvas (K={k_steps})")

    models = {
        "dg": ("google/diffusiongemma-26B-A4B-it", True),
        "ar30b": ("Qwen/Qwen3-30B-A3B", False),
        "ar3b": ("Qwen/Qwen2.5-3B-Instruct", False),
    }
    if arm not in models:
        print(f"Unknown arm: {arm}. Choose from {list(models.keys())}")
        return

    model_name, is_diffusion = models[arm]
    print(f"Loading {model_name} (arm={arm})...")

    llm_kwargs = dict(
        model=model_name, dtype="bfloat16",
        gpu_memory_utilization=0.85, max_model_len=4096,
        enable_prefix_caching=True, trust_remote_code=True,
    )
    if is_diffusion:
        llm_kwargs["diffusion_config"] = {
            "canvas_length": 256, "max_denoising_steps": k_steps}

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    prompts = []
    for problem in problems:
        text = build_prompt(problem)
        messages = [{"role": "user", "content": text}]
        try:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        prompts.append(chat_text)

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=0)

    # Serial (batch-1) per-event timing
    latencies = []
    results = []
    print(f"\n=== Serial (batch-1), n={len(problems)} ===")
    for i, (problem, prompt) in enumerate(zip(problems, prompts)):
        t0 = time.perf_counter()
        outs = llm.generate([prompt], sp, use_tqdm=False)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        text = outs[0].outputs[0].text
        token_count = len(outs[0].outputs[0].token_ids)
        code = extract_code(text, problem)

        valid = is_valid_python(code)
        complete = has_return(code) if valid else False
        passed = run_tests(problem, code) if valid else False

        results.append({
            "task_id": problem["task_id"],
            "entry_point": problem["entry_point"],
            "latency_ms": round(lat, 1),
            "token_count": token_count,
            "code_validity": valid,
            "completeness": complete,
            "pass_at_1": passed,
            "generated": text[:500],
        })

        if (i + 1) % 20 == 0:
            pass_rate = sum(r["pass_at_1"] for r in results) / len(results)
            print(f"  {i+1}/{len(problems)}  pass@1={pass_rate:.3f}  "
                  f"p50={sorted(latencies)[len(latencies)//2]:.0f}ms", flush=True)

    import statistics
    import numpy as np

    pass_count = sum(r["pass_at_1"] for r in results)
    valid_count = sum(r["code_validity"] for r in results)
    complete_count = sum(r["completeness"] for r in results)

    prov = capture_provenance(
        model_name=model_name, model_type=arm, backend="vllm",
        max_new_tokens=max_tokens, parser_policy="ast_parse",
        scoring_fields="pass_at_1",
    )
    if is_diffusion:
        prov.denoising_steps = k_steps
        prov.canvas_length = 256

    result = {
        "experiment": "issue115_humaneval",
        "arm": arm,
        "model": model_name,
        "model_type": arm,
        "benchmark": "HumanEval",
        "n_problems": len(results),
        "k_steps": k_steps if is_diffusion else None,
        "canvas_length": 256 if is_diffusion else None,
        "max_new_tokens": max_tokens,
        "temperature": 0.0,
        "serving": "serial (batch-1)",
        "pass_at_1": round(pass_count / len(results), 4),
        "code_validity_pct": round(valid_count / len(results) * 100, 1),
        "completeness_pct": round(complete_count / len(results) * 100, 1),
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1),
            "p99": round(sorted(latencies)[int(len(latencies) * 0.99)], 1),
            "mean": round(statistics.fmean(latencies), 1),
        },
        "mean_output_tokens": round(sum(r["token_count"] for r in results) / len(results), 1),
        "predictions": results,
        "provenance": prov.to_dict(),
    }

    out_dir = REPO / "results" / "issue115_humaneval"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_K{k_steps}" if is_diffusion else ""
    if deadline_ms is not None:
        suffix += f"_D{deadline_ms}"
    out_path = out_dir / f"{arm}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {out_path}")

    print(f"\n=== RESULTS ({arm}, K={k_steps if is_diffusion else 'N/A'}) ===")
    print(f"  pass@1:         {result['pass_at_1']:.4f} ({pass_count}/{len(results)})")
    print(f"  code_validity:  {result['code_validity_pct']:.1f}%")
    print(f"  completeness:   {result['completeness_pct']:.1f}%")
    print(f"  p50 latency:    {result['latency_ms']['p50']:.1f}ms")
    print(f"  p99 latency:    {result['latency_ms']['p99']:.1f}ms")
    print(f"  mean tokens:    {result['mean_output_tokens']:.1f}")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["dg", "ar30b", "ar3b"], required=True)
    ap.add_argument("--k-steps", type=int, default=8)
    ap.add_argument("--n", type=int, default=None, help="limit problems (default: all 164)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--deadline", type=int, default=None,
                    help="deadline in ms (#116): AR truncates max_tokens, DG full canvas")
    args = ap.parse_args()
    run_experiment(args.arm, k_steps=args.k_steps, n=args.n,
                  max_tokens=args.max_tokens, deadline_ms=args.deadline)
