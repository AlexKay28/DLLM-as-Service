#!/usr/bin/env python3
"""Issue #117: Multi-turn code correction chain — generate → execute → fix → re-execute.

Tests the serial chain latency claim with a real multi-turn workflow on HumanEval.
Each round: generate code, execute it, if it fails feed the error back and ask
the model to fix it. Up to 5 rounds.

Arms: DG (DiffusionGemma-26B-A4B-it, K=8) and AR-30B (Qwen3-30B-A3B).
Both serial (batch-1) on vLLM 0.25.1.

Usage:
    .venv/bin/python models/experiments/run_code_chain.py --arm dg --k-steps 8
    .venv/bin/python models/experiments/run_code_chain.py --arm ar30b
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
MAX_ROUNDS = 5

GEN_PROMPT = """Complete the following Python function. Return ONLY the function body, or the complete function. The code must be valid Python.

{prompt}
Your code:"""

FIX_PROMPT = """The following Python code failed its tests. Fix it.

Problem:
{prompt}

Code that failed:
```python
{code}
```

Error:
{error}

Return ONLY the corrected code (the complete function or just the body):
```python
"""


def load_humaneval():
    problems = []
    with open(HUMANEVAL_PATH) as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def extract_code(generated_text, problem):
    text = generated_text.strip()
    if "```" in text:
        parts = text.split("```")
        best = ""
        for i, part in enumerate(parts):
            if i == 0 or i == len(parts) - 1:
                continue
            if part.startswith("python"):
                part = part[6:]
            part = part.strip()
            if len(part) > len(best):
                best = part
        if best:
            text = best
    if text.startswith("def ") or text.startswith("from ") or text.startswith("import "):
        return text
    indented = "\n".join("    " + line if line.strip() else line for line in text.split("\n"))
    return problem["prompt"] + indented


def is_valid_python(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def run_tests_capture_error(problem, code, timeout=10):
    """Run tests, return (passed, error_output)."""
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
            passed = "PASS" in result.stdout
            error = result.stderr[-500:] if result.stderr else ""
            if not passed and not error:
                error = "Test failed (no exception)"
            return passed, error
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
        finally:
            os.unlink(f.name)


def run_chain_experiment(arm, k_steps=8, n=None, max_tokens=512):
    from vllm import LLM, SamplingParams

    problems = load_humaneval()
    if n:
        problems = problems[:n]
    print(f"Loaded {len(problems)} HumanEval problems, max {MAX_ROUNDS} rounds each")

    models = {
        "dg": ("google/diffusiongemma-26B-A4B-it", True),
        "ar30b": ("Qwen/Qwen3-30B-A3B", False),
    }
    if arm not in models:
        print(f"Unknown arm: {arm}")
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
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=0)

    def generate(prompt_text):
        messages = [{"role": "user", "content": prompt_text}]
        try:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        t0 = time.perf_counter()
        outs = llm.generate([chat_text], sp, use_tqdm=False)
        lat = (time.perf_counter() - t0) * 1000.0
        text = outs[0].outputs[0].text
        tok = len(outs[0].outputs[0].token_ids)
        return text, lat, tok

    all_results = []
    print(f"\n=== Chain experiment (max {MAX_ROUNDS} rounds, n={len(problems)}) ===")

    for idx, problem in enumerate(problems):
        rounds = []
        current_prompt = GEN_PROMPT.format(prompt=problem["prompt"])
        cumulative_time = 0.0

        for round_num in range(1, MAX_ROUNDS + 1):
            gen_text, lat, tok = generate(current_prompt)
            cumulative_time += lat
            code = extract_code(gen_text, problem)
            valid = is_valid_python(code)

            if valid:
                passed, error = run_tests_capture_error(problem, code)
            else:
                passed, error = False, "SyntaxError: invalid Python"

            rounds.append({
                "round": round_num,
                "latency_ms": round(lat, 1),
                "cumulative_ms": round(cumulative_time, 1),
                "token_count": tok,
                "code_validity": valid,
                "pass_at_1": passed,
            })

            if passed:
                break

            # Build fix prompt for next round
            current_prompt = FIX_PROMPT.format(
                prompt=problem["prompt"], code=code, error=error)

        total_lat = sum(r["latency_ms"] for r in rounds)
        solved = rounds[-1]["pass_at_1"]
        solve_round = len(rounds) if solved else 0

        all_results.append({
            "task_id": problem["task_id"],
            "rounds": rounds,
            "n_rounds": len(rounds),
            "solved": solved,
            "solve_round": solve_round,
            "total_latency_ms": round(total_lat, 1),
        })

        if (idx + 1) % 20 == 0:
            solved_count = sum(r["solved"] for r in all_results)
            print(f"  {idx+1}/{len(problems)}  solved={solved_count}/{len(all_results)}  "
                  f"({solved_count/len(all_results)*100:.1f}%)", flush=True)

    import statistics

    pass_r1 = sum(1 for r in all_results if r["solve_round"] == 1) / len(all_results)
    pass_total = sum(1 for r in all_results if r["solved"]) / len(all_results)

    # Cumulative pass@1 per round
    cumulative = []
    for rnd in range(1, MAX_ROUNDS + 1):
        count = sum(1 for r in all_results if r["solve_round"] <= rnd and r["solve_round"] > 0)
        cumulative.append(count / len(all_results))

    # Per-round latency (only first-round latencies for p50)
    r1_lats = [r["rounds"][0]["latency_ms"] for r in all_results]

    prov = capture_provenance(
        model_name=model_name, model_type=arm, backend="vllm",
        max_new_tokens=max_tokens, parser_policy="ast_parse",
        scoring_fields="pass_at_1_cumulative",
    )
    if is_diffusion:
        prov.denoising_steps = k_steps
        prov.canvas_length = 256

    result = {
        "experiment": "issue117_code_chain",
        "arm": arm,
        "model": model_name,
        "model_type": arm,
        "benchmark": "HumanEval",
        "n_problems": len(all_results),
        "max_rounds": MAX_ROUNDS,
        "k_steps": k_steps if is_diffusion else None,
        "canvas_length": 256 if is_diffusion else None,
        "max_new_tokens": max_tokens,
        "serving": "serial (batch-1)",
        "pass_at_1_round1": round(pass_r1, 4),
        "pass_at_1_cumulative": round(pass_total, 4),
        "cumulative_by_round": [round(c, 4) for c in cumulative],
        "r1_latency_ms": {
            "p50": round(statistics.median(r1_lats), 1),
            "p99": round(sorted(r1_lats)[int(len(r1_lats) * 0.99)], 1),
            "mean": round(statistics.fmean(r1_lats), 1),
        },
        "total_latency_ms": {
            "p50": round(statistics.median([r["total_latency_ms"] for r in all_results]), 1),
            "p99": round(sorted([r["total_latency_ms"] for r in all_results])[int(len(all_results) * 0.99)], 1),
            "mean": round(statistics.fmean([r["total_latency_ms"] for r in all_results]), 1),
        },
        "predictions": all_results,
        "provenance": prov.to_dict(),
    }

    out_dir = REPO / "results" / "issue117_chain"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_K{k_steps}" if is_diffusion else ""
    out_path = out_dir / f"{arm}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {out_path}")

    print(f"\n=== RESULTS ({arm}, K={k_steps if is_diffusion else 'N/A'}) ===")
    print(f"  pass@1 (round 1):     {result['pass_at_1_round1']:.4f}")
    print(f"  pass@1 (cumulative):  {result['pass_at_1_cumulative']:.4f}")
    print(f"  cumulative by round:  {result['cumulative_by_round']}")
    print(f"  R1 p50 latency:       {result['r1_latency_ms']['p50']:.1f}ms")
    print(f"  Total p50 latency:    {result['total_latency_ms']['p50']:.1f}ms")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["dg", "ar30b"], required=True)
    ap.add_argument("--k-steps", type=int, default=8)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    run_chain_experiment(args.arm, k_steps=args.k_steps, n=args.n,
                         max_tokens=args.max_tokens)
