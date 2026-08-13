#!/usr/bin/env python3
"""
Issue #14 (P1): open-loop rate-vs-lag curve for a served model.

The closed-burst throughput in run_vllm_serving.py cannot measure queueing
lag: every burst arrives at t=0, so "p99" there is just the slowest request
in the burst. This script is the open-loop alternative. It serves the model
with `vllm serve` and drives it with Poisson arrivals at a fixed rate,
recording per request:

    t_arrive (client send) — t_first_token — t_finish

Lag = t_finish - t_arrive (arrival-referenced; no coordinated omission).
Requests still in flight `drain_timeout` after the send window closes are
recorded as abandoned — past capacity, that backlog IS the measurement.

Usage:
    .venv-dg/bin/python models/experiments/rate_lag_curve.py \
        --model google/diffusiongemma-26B-A4B-it \
        --rates 0.5 1 2 3 4 5 6 8 12 \
        --output results/rate_lag_dg.json
"""

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.latency_metrics import latency_percentiles
from models.baselines.ar_model import build_chat_messages

SPLIT_DIR = _REPO_ROOT / "datasets" / "splits" / "test"
VENV_PY = _REPO_ROOT / ".venv-dg" / "bin" / "python"


def load_log_lines(path, n=None):
    lines = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def start_server(model, port, gpu_mem, max_model_len, log_path, extra_args):
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    venv_bin = str(VENV_PY.parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    cmd = [str(VENV_PY), "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--port", str(port),
           "--dtype", "bfloat16", "--max-model-len", str(max_model_len),
           "--gpu-memory-utilization", str(gpu_mem)] + extra_args
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    return proc, log


async def wait_healthy(proc, port, timeout_s=900):
    import httpx
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (code {proc.returncode}); see log")
            try:
                r = await client.get(f"http://127.0.0.1:{port}/health",
                                     timeout=5)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(5)
    raise RuntimeError(f"server not healthy after {timeout_s}s; see log")


def kill_server(proc, log):
    try:
        os.killpg(proc.pid, 15)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        proc.wait(timeout=60)
    log.close()
    time.sleep(10)  # let CUDA memory release before the next engine loads


async def run_window(port, model, prompts, rate, seed, duration_s,
                     max_events, drain_timeout, pbar_label):
    """One offered rate: Poisson sends for min(duration_s, max_events/rate),
    then wait for completions up to drain_timeout."""
    import httpx

    rng = random.Random(seed)
    schedule = []
    t = 0.0
    while t <= duration_s and len(schedule) < max_events:
        schedule.append(t)
        t += rng.expovariate(rate)
    n_to_send = len(schedule)

    order = list(range(len(prompts)))
    rng.shuffle(order)

    records = []

    async def one(client, i, offset):
        await asyncio.sleep(offset)
        t0 = time.perf_counter()
        prompt = prompts[order[i % len(order)]]
        payload = {
            "model": model,
            "messages": build_chat_messages(prompt, source="bgl"),
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": True,
        }
        rec = {"arrive": t0, "ttft": None, "finish": None, "ok": False,
               "out_tokens": None}
        try:
            async with client.stream(
                    "POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                    json=payload) as resp:
                if resp.status_code != 200:
                    rec["error"] = f"http {resp.status_code}"
                    await resp.aread()
                else:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        if rec["ttft"] is None and '"content"' in line:
                            rec["ttft"] = time.perf_counter()
                        if '"usage"' in line and '"completion_tokens"' in line:
                            try:
                                usage = json.loads(line[5:])["usage"]
                                rec["out_tokens"] = usage["completion_tokens"]
                            except Exception:
                                pass
            rec["finish"] = time.perf_counter()
            rec["ok"] = True
        except asyncio.CancelledError:
            rec["finish"] = time.perf_counter()
        except Exception as e:
            rec["finish"] = time.perf_counter()
            rec["error"] = str(e)[:200]
        records.append(rec)
        if len(records) % 25 == 0:
            done = [r for r in records if r["ok"]]
            if done:
                lags = sorted(r["finish"] - r["arrive"] for r in done)
                print(f"      {pbar_label}: {len(records)}/{n_to_send} done, "
                      f"lag p50={lags[len(lags)//2]*1000:.0f}ms", flush=True)

    limits = httpx.Limits(max_connections=2048, max_keepalive_connections=2048)
    timeout = httpx.Timeout(None, connect=30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        tasks = [asyncio.create_task(one(client, i, off))
                 for i, off in enumerate(schedule)]
        send_end = schedule[-1] if schedule else 0
        await asyncio.sleep(send_end)
        done, pending = await asyncio.wait(tasks, timeout=drain_timeout)
        for t_ in pending:
            t_.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    n_abandoned = sum(1 for r in records if r["finish"] is None
                      or (not r["ok"] and "error" not in r))
    return records, n_to_send, n_abandoned


def top_errors(records, k=5):
    counts = {}
    for r in records:
        if not r["ok"] and r.get("error"):
            counts[r["error"]] = counts.get(r["error"], 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:k]


def summarize(records, n_sent, n_abandoned, rate):
    ok = [r for r in records if r["ok"]]
    lags = [(r["finish"] - r["arrive"]) * 1000 for r in ok]
    ttfts = [(r["ttft"] - r["arrive"]) * 1000 for r in ok if r["ttft"]]
    out_tok = [r["out_tokens"] for r in ok if r["out_tokens"]]
    return {
        "offered_rate_ev_s": rate,
        "sent": n_sent,
        "completed": len(ok),
        "errors": len(records) - len(ok) - n_abandoned,
        "error_samples": top_errors(records),
        "abandoned_at_drain": n_abandoned,
        "completion_ratio": round(len(ok) / n_sent, 4) if n_sent else None,
        "lag_ms": latency_percentiles(lags),
        "ttft_ms": latency_percentiles(ttfts),
        "mean_out_tokens": (sum(out_tok) / len(out_tok)) if out_tok else None,
    }


async def amain():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rates", type=float, nargs="+", required=True,
                    help="offered Poisson rates in events/s")
    ap.add_argument("--n-prompt-pool", type=int, default=400,
                    help="distinct BGL lines cycled through (prompt variety)")
    ap.add_argument("--duration-s", type=float, default=150.0,
                    help="max send window per rate")
    ap.add_argument("--max-events", type=int, default=300,
                    help="max sends per rate")
    ap.add_argument("--drain-timeout", type=float, default=45.0,
                    help="wait for in-flight completion after the send window")
    ap.add_argument("--seed", type=int, default=0,
                    help="arrival-process seed; identical across models "
                         "gives paired windows")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--server-arg", action="append", default=[],
                    help="extra raw args for vllm serve (repeatable)")
    ap.add_argument("--server-log", default="logs/rate_lag_server.log")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    prompts = load_log_lines(str(SPLIT_DIR / "bgl.raw"), n=args.n_prompt_pool)
    print(f"Prompt pool: {len(prompts)} BGL lines", flush=True)

    proc, log = start_server(args.model, args.port,
                             args.gpu_memory_utilization,
                             args.max_model_len, args.server_log,
                             args.server_arg)
    results = {
        "experiment": "issue14_rate_lag_curve",
        "model": args.model,
        "arrivals": "poisson, open-loop; lag = finish - arrive (client-side)",
        "duration_s": args.duration_s,
        "max_events_per_rate": args.max_events,
        "drain_timeout_s": args.drain_timeout,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rates": [],
    }
    try:
        print("waiting for server health...", flush=True)
        await wait_healthy(proc, args.port)
        print("server healthy; sweeping rates", flush=True)
        for rate in args.rates:
            print(f"  rate={rate} ev/s ...", flush=True)
            recs, n_sent, n_aband = await run_window(
                args.port, args.model, prompts, rate, args.seed,
                args.duration_s, args.max_events, args.drain_timeout,
                pbar_label=f"λ={rate}")
            row = summarize(recs, n_sent, n_aband, rate)
            results["rates"].append(row)
            lag = row["lag_ms"]
            print(f"    sent={row['sent']} done={row['completed']} "
                  f"abandoned={row['abandoned_at_drain']} "
                  f"lag p50={lag.get('p50_ms', float('nan')):.0f}ms "
                  f"p99={lag.get('p99_ms', float('nan')):.0f}ms", flush=True)
            # crash-proof: rewrite the whole JSON after every rate
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
    finally:
        kill_server(proc, log)

    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    asyncio.run(amain())
