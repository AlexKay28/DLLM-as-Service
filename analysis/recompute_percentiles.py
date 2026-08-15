#!/usr/bin/env python3
"""Recompute p50/p95/p99 for every latency cell from existing artifacts.

Issue #76: p50-only reporting in a latency paper.

Walks all JSON files in results/, finds every latency summary dict and
raw_latencies_ms array, and recomputes p50/p95/p99/p99.9/mean/min/max.
Outputs results/issue76_percentile_summary.json and .md.
"""

import json
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_JSON = RESULTS_DIR / "issue76_percentile_summary.json"
OUTPUT_MD = RESULTS_DIR / "issue76_percentile_summary.md"

LATENCY_SUBKEYS = [
    "latency", "latency_ms", "lag_ms", "ttft_ms",
    "vip_lag_ms", "background_lag_ms",
]

SKIP_RECURSE_KEYS = set(LATENCY_SUBKEYS) | {
    "raw_latencies_ms", "raw_out_tokens", "per_event_f1",
    "quality", "per_field_f1", "provenance", "sla_grid_ms",
    "power", "cost", "at_sla", "error_samples",
    "paper_table3", "throughput_table4", "denoising_steps_sweep",
    "comparator_ar3b_same_host", "tail_ratio_p999_over_p50",
    "tail_ratio", "mean_output_tokens",
    "vllm_version", "torch_version", "gpu_model",
    "timestamp", "experiment", "serving",
    "levels", "suffix_sizes",
    "peak_events_per_s", "peak_tokens_per_s",
    "peak_events_per_s_batch",
}

CONDITION_KEYS = [
    "max_denoising_steps", "denoising_steps", "k_steps",
    "max_new_tokens", "min_tokens", "canvas_length",
    "batch_size", "offered_rate_ev_s", "schema_repeats",
    "prompt_token_count", "source", "n_fields", "level",
    "k", "round", "suffix_size", "steps", "n", "n_events",
    "latency_source", "seed", "cold_start_s",
    "model_type", "stratification", "threshold",
    "completion_ratio", "sent", "completed", "errors",
    "abandoned_at_drain", "vip_n", "bg_cancelled_by_vip",
    "duration_s", "max_events_per_rate", "drain_timeout_s",
    "arrivals", "prefix_caching", "min_tokens",
    "field_f1_imputed", "field_f1_all",
    "suffix_identity_rate",
]

MODEL_HINTS = ("qwen", "diffusion", "llada", "flowfill", "gemma", "instruct")


def percentile(sorted_data, p):
    """Linear-interpolation percentile (standard / numpy-default method)."""
    n = len(sorted_data)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_data[0])
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * frac


def has_percentile(d):
    """Check if dict has latency percentile keys (p50 + at least one other)."""
    if not isinstance(d, dict):
        return False
    has_p50 = any(k in d for k in ("p50_ms", "p50"))
    has_other = any(k in d for k in (
        "p99_ms", "p99", "p999_ms", "p999",
        "mean_ms", "mean", "p90_ms", "p90",
    ))
    return has_p50 and has_other


def get_val(d, *keys):
    """Get first available numeric value for any of the keys."""
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                return v
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def round_val(v, digits=3):
    if v is None:
        return None
    return round(v, digits)


def make_cell(lat_dict, raw, model, condition, source_file, metric):
    """Build a latency cell from a latency summary dict and/or raw array."""
    has_raw = raw is not None and isinstance(raw, list) and len(raw) > 0

    if has_raw:
        s = sorted(raw)
        p50 = percentile(s, 50)
        p95 = percentile(s, 95)
        p99 = percentile(s, 99)
        p999 = percentile(s, 99.9)
        mean = statistics.mean(s)
        mn = s[0]
        mx = s[-1]
        n = len(s)
    else:
        p50 = get_val(lat_dict, "p50_ms", "p50")
        p95 = get_val(lat_dict, "p95_ms", "p95")
        p99 = get_val(lat_dict, "p99_ms", "p99")
        p999 = get_val(lat_dict, "p99_9_ms", "p999_ms", "p999", "p99_9")
        mean = get_val(lat_dict, "mean_ms", "mean")
        mn = get_val(lat_dict, "min_ms", "min")
        mx = get_val(lat_dict, "max_ms", "max")
        n = get_val(lat_dict, "n_samples", "n", "n_tested")
        if n is None:
            n = condition.get("n") or condition.get("n_events")

    if n is None:
        n = condition.get("n") or condition.get("n_events") or condition.get("completed")

    max_median_ratio = round_val(mx / p50) if (p50 and mx and p50 > 0) else None
    p99_p50_ratio = round_val(p99 / p50) if (p50 and p99 and p50 > 0) else None

    cond = {k: v for k, v in condition.items() if v is not None}

    return {
        "source_file": source_file,
        "model": model,
        "condition": cond,
        "metric": metric,
        "n": n,
        "p50_ms": round_val(p50),
        "p95_ms": round_val(p95),
        "p99_ms": round_val(p99),
        "p99_9_ms": round_val(p999),
        "mean_ms": round_val(mean),
        "min_ms": round_val(mn),
        "max_ms": round_val(mx),
        "max_over_median": max_median_ratio,
        "p99_over_p50": p99_p50_ratio,
        "has_raw_data": has_raw,
    }


def looks_like_model(key):
    """Check if a string looks like a HuggingFace model name."""
    return isinstance(key, str) and "/" in key


def extract_model(obj, path_key, current_model):
    """Try to extract model name from dict or path key."""
    if isinstance(obj, dict):
        m = obj.get("model")
        if isinstance(m, str) and m:
            return m
        prov = obj.get("provenance")
        if isinstance(prov, dict):
            pm = prov.get("model_name")
            if isinstance(pm, str) and pm:
                return pm
    if looks_like_model(path_key):
        return path_key
    return current_model


def scan(obj, model, condition, source_file, path_key, parent_obj=None):
    """Recursively scan for latency cells.

    parent_obj is the dict containing obj (for accessing siblings like
    raw_latencies_ms).
    """
    cells = []

    if isinstance(obj, dict):
        model = extract_model(obj, path_key, model)

        cond = dict(condition) if condition else {}
        for k in CONDITION_KEYS:
            if k in obj:
                cond[k] = obj[k]

        # 1. Named latency sub-dicts (latency, latency_ms, lag_ms, ttft_ms, ...)
        found_latency_subdict = False
        for lat_key in LATENCY_SUBKEYS:
            sub = obj.get(lat_key)
            if isinstance(sub, dict) and has_percentile(sub):
                raw = obj.get("raw_latencies_ms")
                cells.append(make_cell(sub, raw, model, cond, source_file, lat_key))
                found_latency_subdict = True

        # 2. Direct percentile keys (flat p50_ms / p99_ms)
        if not found_latency_subdict and has_percentile(obj):
            raw = obj.get("raw_latencies_ms")
            metric_label = path_key if path_key else "direct"
            cells.append(make_cell(obj, raw, model, cond, source_file, metric_label))

        # 3. paper_table3 (p50/p99/p999 without _ms suffix; model = key)
        pt3 = obj.get("paper_table3")
        if isinstance(pt3, dict):
            for mname, pdata in pt3.items():
                if isinstance(pdata, dict) and has_percentile(pdata):
                    cells.append(make_cell(
                        pdata, None, mname,
                        {"steps": pdata.get("steps")},
                        source_file, "paper_table3"))

        # 4. throughput_table4.sweep (p99_ms = burst wall clock)
        tt4 = obj.get("throughput_table4")
        if isinstance(tt4, dict):
            sweep = tt4.get("sweep")
            if isinstance(sweep, list):
                for item in sweep:
                    if isinstance(item, dict) and "p99_ms" in item:
                        cells.append(make_cell(
                            {"p99_ms": item["p99_ms"]},
                            None, model,
                            {**cond,
                             "batch_size": item.get("batch_size"),
                             "latency_source": item.get("latency_source")},
                            source_file, "throughput_sweep"))

        # 5. denoising_steps_sweep
        dss = obj.get("denoising_steps_sweep")
        if isinstance(dss, dict):
            for step_key, step_data in dss.items():
                if isinstance(step_data, dict) and has_percentile(step_data):
                    cells.append(make_cell(
                        step_data, None, model,
                        {**cond, "denoising_steps": step_key},
                        source_file, "denoising_steps_sweep"))

        # 6. comparator
        for ck in ("comparator_ar3b_same_host",):
            cdata = obj.get(ck)
            if isinstance(cdata, dict) and has_percentile(cdata):
                cells.append(make_cell(
                    cdata, None, model,
                    {**cond, "comparator": ck},
                    source_file, ck))

        # 7. Recurse into children
        for key, val in obj.items():
            if key in SKIP_RECURSE_KEYS:
                continue
            if key in LATENCY_SUBKEYS:
                continue
            if isinstance(key, str) and key.endswith("provenance"):
                continue

            child_model = model
            child_cond = dict(cond)

            # If key is a model name (contains /), use as model
            if looks_like_model(key):
                child_model = key
            # If sibling <key>_provenance exists, extract model
            elif isinstance(key, str):
                prov_key = key + "_provenance"
                if prov_key in obj and isinstance(obj[prov_key], dict):
                    pm = obj[prov_key].get("model_name")
                    if isinstance(pm, str) and pm:
                        child_model = pm

            cells.extend(
                scan(val, child_model, child_cond, source_file, key, obj))

    elif isinstance(obj, list):
        for item in obj:
            cells.extend(
                scan(item, model, condition, source_file, path_key, parent_obj))

    return cells


def classify_experiment(source_file):
    """Infer experiment type from filename for grouping."""
    f = source_file
    if "ksweep" in f:
        return "K-sweep (DG)"
    if "issue68" in f:
        return "Output-length sweep (DG)"
    if "issue66_canvas" in f:
        return "Canvas sweep (DG)"
    if "input_sweep" in f:
        return "Input-length sweep"
    if "issue42_batching" in f:
        return "Batching sweep"
    if "issue49_ar1b" in f or "issue_evidence_ar1b" in f:
        return "AR-1.5B variants"
    if "vllm_serving_1.5b" in f:
        return "AR-1.5B serving"
    if "issue49_ar3b" in f or "issue47_" in f or "issue45_" in f:
        return "AR-3B variants"
    if "issue50_hdfs" in f:
        return "AR-3B HDFS"
    if "vllm_serving_3b" in f or "comparator" in f:
        return "AR-3B serving"
    if "issue49_ar7b" in f or "vllm_serving_7b" in f:
        return "AR-7B variants"
    if "issue49_mdm" in f:
        return "MDM (LLaDA)"
    if "diffusiongemma_perf" in f or "dg_steps" in f:
        return "DG (DiffusionGemma)"
    if "rate_lag" in f or "vip_probe" in f:
        return "Rate-lag curve"
    if "wide_schema" in f:
        return "Wide-schema sweep"
    if "imputation" in f:
        return "Imputation"
    if "adaptive_effort" in f:
        return "Adaptive effort"
    if "refinement" in f:
        return "Iterative refinement"
    if "perturbation" in f:
        return "Perturbation robustness"
    if "issue59_infilling" in f:
        return "Infilling"
    if "latency_table3" in f:
        return "Paper Table 3"
    if "issue67" in f:
        return "HDFS output-length sweep"
    if "appD_" in f:
        return "Output-length sweep"
    if "p2_" in f:
        return "Output-length sweep"
    if "issue43" in f:
        return "Output-length sweep"
    if "issue_evidence" in f:
        return "AR-1.5B forced"
    if "diag_vllm" in f:
        return "vLLM diagnostics"
    if "issue58_memory" in f:
        return "Memory probe"
    if "issue48" in f:
        return "MMPP"
    if "vip_" in f:
        return "VIP probing"
    if "chain" in f:
        return "Chain analysis"
    if "baselines" in f:
        return "Baselines"
    if "difficulty_bins" in f:
        return "Difficulty bins"
    if "per_field_fig4" in f:
        return "Per-field analysis"
    if "ablation" in f:
        return "Ablation"
    if "quality_table2" in f:
        return "Quality table"
    return "Other"


def fmt_ms(v):
    """Format a millisecond value for the markdown table."""
    if v is None:
        return "-"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def fmt_ratio(v):
    if v is None:
        return "-"
    return f"{v:.2f}x"


def write_markdown(cells, path):
    """Write grouped markdown summary table."""
    groups = {}
    for c in cells:
        exp = classify_experiment(c["source_file"])
        groups.setdefault(exp, []).append(c)

    lines = []
    lines.append("# Issue #76: Percentile Summary")
    lines.append("")
    lines.append(
        f"Recomputed p50/p95/p99 for {len(cells)} latency cells "
        f"from {len(set(c['source_file'] for c in cells))} files.")
    raw_count = sum(1 for c in cells if c["has_raw_data"])
    lines.append(
        f"Cells with raw data: {raw_count}. "
        f"Summary-only: {len(cells) - raw_count}.")
    lines.append("")

    for exp in sorted(groups.keys()):
        group_cells = groups[exp]
        lines.append(f"## {exp}")
        lines.append("")
        lines.append(
            "| Model | Condition | n | p50 | p95 | p99 | p99.9 | "
            "max/med |")
        lines.append(
            "|-------|-----------|---|-----|-----|-----|-------|"
            "---------|")
        for c in sorted(group_cells, key=lambda x: (x["model"] or "", x["source_file"])):
            cond_str = ", ".join(
                f"{k}={v}" for k, v in c["condition"].items()
                if k not in ("n", "n_events", "source",
                              "prefix_caching", "serving",
                              "vllm_version", "torch_version",
                              "gpu_model", "model_type",
                              "arrivals", "duration_s",
                              "max_events_per_rate", "drain_timeout_s",
                              "seed", "cold_start_s", "threshold",
                              "stratification", "min_tokens",
                              "_subkey")
                and v is not None
            )
            if not cond_str:
                cond_str = c["metric"]
            n_str = str(c["n"]) if c["n"] is not None else "-"
            raw_tag = "*" if c["has_raw_data"] else ""
            lines.append(
                f"| {c['model'] or '?'}{raw_tag} | {cond_str} | {n_str} | "
                f"{fmt_ms(c['p50_ms'])} | {fmt_ms(c['p95_ms'])} | "
                f"{fmt_ms(c['p99_ms'])} | {fmt_ms(c['p99_9_ms'])} | "
                f"{fmt_ratio(c['max_over_median'])} |")
        lines.append("")

    lines.append("*\\* = computed from raw_latencies_ms array*")
    lines.append("")

    path.write_text("\n".join(lines))


def main():
    all_cells = []
    files_scanned = 0
    files_with_latency = 0

    for json_file in sorted(RESULTS_DIR.rglob("*.json")):
        if json_file.name.startswith("issue76_"):
            continue
        if json_file.name.endswith(".provenance.json"):
            continue
        files_scanned += 1
        try:
            data = json.loads(json_file.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        rel = str(json_file.relative_to(RESULTS_DIR))
        cells = scan(data, None, {}, rel, None)
        if cells:
            files_with_latency += 1
            all_cells.extend(cells)

    # Deduplicate by (source_file, model, condition, metric)
    seen = set()
    unique = []
    for c in all_cells:
        key = (
            c["source_file"],
            c["model"],
            json.dumps(c["condition"], sort_keys=True),
            c["metric"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # Sort for stable output
    unique.sort(key=lambda c: (c["source_file"], c["model"] or "", c["metric"]))

    # Write JSON
    OUTPUT_JSON.write_text(json.dumps(unique, indent=2) + "\n")

    # Write markdown
    write_markdown(unique, OUTPUT_MD)

    # Print summary to stdout
    raw_count = sum(1 for c in unique if c["has_raw_data"])
    print(f"Files scanned: {files_scanned}")
    print(f"Files with latency data: {files_with_latency}")
    print(f"Total latency cells: {len(unique)}")
    print(f"  With raw data: {raw_count}")
    print(f"  Summary-only:  {len(unique) - raw_count}")
    print()

    # Key findings
    print("=== Key findings ===")
    print()

    # DG K=8
    k8 = [c for c in unique if "ksweep_K8" in c["source_file"]
          and c["metric"] == "latency"]
    if k8:
        c = k8[0]
        print(f"DG K=8 (issue44_dg_ksweep_K8.json):")
        print(f"  p50={c['p50_ms']}, p95={c['p95_ms']}, "
              f"p99={c['p99_ms']}, p99.9={c['p99_9_ms']}")
        print(f"  max/median={c['max_over_median']}, "
              f"has_raw={c['has_raw_data']}")
        print()

    # AR-3B baseline
    ar3b = [c for c in unique
            if "issue47_baseline" in c["source_file"]
            and c["metric"] == "latency"]
    if ar3b:
        c = ar3b[0]
        print(f"AR-3B baseline (issue47_baseline.json):")
        print(f"  p50={c['p50_ms']}, p95={c['p95_ms']}, "
              f"p99={c['p99_ms']}, p99.9={c['p99_9_ms']}")
        print(f"  max/median={c['max_over_median']}, "
              f"has_raw={c['has_raw_data']}")
        print()

    # AR-1.5B unforced
    ar1b = [c for c in unique
            if "issue49_ar1b" in c["source_file"]]
    if ar1b:
        c = ar1b[0]
        print(f"AR-1.5B unforced (issue49_ar1b.json):")
        print(f"  p50={c['p50_ms']}, p95={c['p95_ms']}, "
              f"p99={c['p99_ms']}")
        print(f"  has_raw={c['has_raw_data']}")
        print()

    # AR-1.5B forced L=1024
    ar1b_l1024 = [c for c in unique
                  if "issue_evidence_ar1b_L1024" in c["source_file"]
                  and c["metric"] == "latency"]
    if ar1b_l1024:
        c = ar1b_l1024[0]
        print(f"AR-1.5B forced L=1024 (issue_evidence_ar1b_L1024.json):")
        print(f"  p50={c['p50_ms']}, p95={c['p95_ms']}, "
              f"p99={c['p99_ms']}, p99.9={c['p99_9_ms']}")
        print(f"  has_raw={c['has_raw_data']}")
        print()

    # Widest tails (p99/p50 ratio)
    print("=== Widest tails (p99/p50 ratio) ===")
    tail_cells = [c for c in unique
                  if c["p99_over_p50"] is not None
                  and c["p50_ms"] is not None
                  and c["p50_ms"] > 1.0
                  and c["metric"] not in ("throughput_sweep",)
                  and c["n"] is not None
                  and c["n"] >= 50]
    tail_cells.sort(key=lambda c: c["p99_over_p50"], reverse=True)
    for c in tail_cells[:10]:
        print(f"  {c['p99_over_p50']:.2f}x  {c['source_file']}  "
              f"p50={c['p50_ms']:.1f} p99={c['p99_ms']:.1f}  "
              f"({c['model']})")
    print()

    # Max/median ratio verification (3.21x)
    print("=== Max/median ratio (looking for 3.21x) ===")
    maxmed_cells = [c for c in unique
                    if c["max_over_median"] is not None
                    and c["p50_ms"] is not None
                    and c["p50_ms"] > 1.0
                    and c["metric"] not in ("throughput_sweep",)
                    and c["n"] is not None
                    and c["n"] >= 50]
    maxmed_cells.sort(key=lambda c: c["max_over_median"], reverse=True)
    for c in maxmed_cells[:10]:
        print(f"  {c['max_over_median']:.2f}x  {c['source_file']}  "
              f"p50={c['p50_ms']:.1f} max={c['max_ms']:.1f}  "
              f"({c['model']})")
    print()

    print(f"Output: {OUTPUT_JSON}")
    print(f"Output: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
