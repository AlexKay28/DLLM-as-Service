"""
Experiment runner: 5 scenarios comparing rule-based, Drain-only, and LLM baselines.

Usage:
    python run_experiments.py --logs raw_logs.txt --source hdfs --gold gold.jsonl
    python run_experiments.py --logs raw_logs.txt --source hdfs --gold gold.jsonl --scenario all
    python run_experiments.py --logs raw_logs.txt --source hdfs --gold gold.jsonl --scenario perturbation

Scenarios:
    1. clean       — All fields on clean logs (rules win structural, LLMs win semantic)
    2. drift       — Format drift over time (rules degrade, LLMs stable)
    3. perturbation — Adversarial perturbations (rules collapse, LLMs robust)
    4. cross_source — Cross-source transfer (rules need rewrite, LLMs transfer)
    5. semantic    — Semantic fields only (rules = 0 F1, LLMs > 0.8 F1)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Ensure repo root is importable when run as a script from any CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.schema import Record, FIELD_NAMES, FIELD_BY_NAME, FieldCategory
from metrics.quality_metrics import field_f1, schema_validity, consistency, anomaly_f1
from metrics.latency_metrics import latency_percentiles, throughput_at_sla, energy_per_event
from models.baselines.rule_based import extract as rule_based_extract
from models.baselines.drain_only import extract as drain_extract
from models.perturbations.perturb import (
    PerturbationType,
    PerturbationConfig,
    perturb_logs,
    perturb_mixed,
)


@dataclass
class ExperimentResult:
    scenario: str
    model_name: str
    field_f1: float = 0.0
    per_field_f1: dict = field(default_factory=dict)
    schema_valid_pct: float = 0.0
    consistency_pct: float = 0.0
    anomaly_f1: float = 0.0
    p50_ms: float = 0.0
    p99_ms: float = 0.0
    p999_ms: float = 0.0
    events_per_s: float = 0.0
    j_per_event: float = 0.0
    extra: dict = field(default_factory=dict)


def load_log_lines(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_records(path: str) -> list[Record]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(Record(fields=data))
    return records


def evaluate_model(pred_records: list[Record], gold_records: list[Record],
                   latencies_ms: list[float], model_name: str,
                   scenario: str, extra: dict | None = None) -> ExperimentResult:
    """Run all metrics on a model's predictions."""
    f1_res = field_f1(pred_records, gold_records)
    sv = schema_validity(pred_records)
    cons = consistency(pred_records)
    anom = anomaly_f1(pred_records, gold_records)
    lat = latency_percentiles(latencies_ms) if latencies_ms else {}

    return ExperimentResult(
        scenario=scenario,
        model_name=model_name,
        field_f1=f1_res["macro_field_f1"],
        per_field_f1={k: v for k, v in f1_res.items() if k.startswith("f1_")},
        schema_valid_pct=sv,
        consistency_pct=cons,
        anomaly_f1=anom["anomaly_f1"],
        p50_ms=lat.get("p50_ms", 0.0),
        p99_ms=lat.get("p99_ms", 0.0),
        p999_ms=lat.get("p99_9_ms", 0.0),
        events_per_s=1000.0 / lat.get("p50_ms", 1) if lat.get("p50_ms", 0) > 0 else 0,
        j_per_event=0.0,
        extra=extra or {},
    )


def run_scenario_clean(log_lines: list[str], gold_records: list[Record],
                        source: str) -> list[ExperimentResult]:
    """Scenario 1: Clean logs, all fields — rules win structural, LLMs win semantic."""
    print(f"\n{'='*60}")
    print(f"Scenario 1: Clean logs (source={source})")
    print(f"{'='*60}")
    results = []

    # Rule-based
    print("  Running rule-based extractor...")
    rb_res = rule_based_extract(log_lines, source=source)
    rb_eval = evaluate_model(rb_res.records, gold_records, rb_res.latencies_ms,
                              "rule-based", "clean",
                              {"fields_extracted": rb_res.fields_extracted,
                               "fields_skipped": rb_res.fields_skipped})
    results.append(rb_eval)
    _print_result(rb_eval)

    # Drain-only
    print("  Running Drain-only extractor...")
    dr_res = drain_extract(log_lines, source=source)
    dr_eval = evaluate_model(dr_res.records, gold_records, dr_res.latencies_ms,
                             "drain-only", "clean",
                             {"drain_available": dr_res.drain_available})
    results.append(dr_eval)
    _print_result(dr_eval)

    # LLM placeholder (requires model weights)
    llm_eval = ExperimentResult(
        scenario="clean", model_name="flowfill-mdm (placeholder)",
        field_f1=0.907, schema_valid_pct=99.4, consistency_pct=97.8,
        anomaly_f1=0.865, p50_ms=23, p99_ms=31, p999_ms=36,
        events_per_s=4350, j_per_event=0.7,
        extra={"note": "Paper values — model weights not available"})
    results.append(llm_eval)
    _print_result(llm_eval)

    return results


def run_scenario_drift(log_lines: list[str], gold_records: list[Record],
                       source: str, n_bins: int = 5) -> list[ExperimentResult]:
    """Scenario 2: Format drift over time — rules degrade, LLMs stable."""
    print(f"\n{'='*60}")
    print(f"Scenario 2: Format drift over time (source={source}, bins={n_bins})")
    print(f"{'='*60}")

    results = []
    bin_size = len(log_lines) // n_bins

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size
        bin_lines = log_lines[start:end]
        bin_gold = gold_records[start:end]

        # Apply increasing perturbation intensity per bin
        intensity = i / n_bins
        config = PerturbationConfig(
            perturbation_type=PerturbationType.WHITESPACE,
            intensity=intensity,
            seed=42 + i,
        )
        perturbed_lines = perturb_logs(bin_lines, config)

        rb_res = rule_based_extract(perturbed_lines, source=source)
        rb_eval = evaluate_model(rb_res.records, bin_gold, rb_res.latencies_ms,
                                 "rule-based", f"drift_bin{i}",
                                 {"perturbation_intensity": intensity})
        results.append(rb_eval)
        print(f"  Bin {i} (intensity={intensity:.1f}): rule-based F1={rb_eval.field_f1:.3f}")

    return results


def run_scenario_perturbation(log_lines: list[str], gold_records: list[Record],
                              source: str) -> list[ExperimentResult]:
    """Scenario 3: Adversarial perturbations — rules collapse, LLMs robust."""
    print(f"\n{'='*60}")
    print(f"Scenario 3: Adversarial perturbations (source={source})")
    print(f"{'='*60}")

    results = []
    perturbation_types = list(PerturbationType)
    intensities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for ptype in perturbation_types:
        for intensity in intensities:
            config = PerturbationConfig(perturbation_type=ptype, intensity=intensity, seed=42)
            perturbed_lines = perturb_logs(log_lines, config)

            rb_res = rule_based_extract(perturbed_lines, source=source)
            rb_eval = evaluate_model(rb_res.records, gold_records, rb_res.latencies_ms,
                                     "rule-based", f"perturb_{ptype.value}",
                                     {"perturbation": ptype.value, "intensity": intensity})
            results.append(rb_eval)

        last = results[-1]
        print(f"  {ptype.value:25s} (intensity=1.0): rule-based F1={last.field_f1:.3f}")

    # Mixed perturbation
    print("\n  Mixed perturbations:")
    for intensity in intensities:
        perturbed_lines, applied = perturb_mixed(log_lines, intensity=intensity, seed=42)

        rb_res = rule_based_extract(perturbed_lines, source=source)
        rb_eval = evaluate_model(rb_res.records, gold_records, rb_res.latencies_ms,
                                 "rule-based", f"perturb_mixed_{intensity}",
                                 {"intensity": intensity, "mixed": True})
        results.append(rb_eval)
        print(f"    mixed (intensity={intensity:.1f}): rule-based F1={rb_eval.field_f1:.3f}")

    return results


def run_scenario_cross_source(log_lines: list[str], gold_records: list[Record],
                               source: str) -> list[ExperimentResult]:
    """Scenario 4: Cross-source transfer — rules need rewrite, LLMs transfer."""
    print(f"\n{'='*60}")
    print(f"Scenario 4: Cross-source transfer (trained on {source})")
    print(f"{'='*60}")

    results = []
    other_sources = [s for s in ["hdfs", "bgl", "thunderbird", "spirit"] if s != source]

    for other in other_sources:
        # Apply other source's patterns to current logs (simulates cross-source)
        rb_res = rule_based_extract(log_lines, source=other)
        rb_eval = evaluate_model(rb_res.records, gold_records, rb_res.latencies_ms,
                                 f"rule-based (trained on {other})",
                                 f"cross_source_{other}",
                                 {"trained_on": other, "tested_on": source})
        results.append(rb_eval)
        print(f"  Rules from {other:12s} -> test on {source}: F1={rb_eval.field_f1:.3f}")

    return results


def run_scenario_semantic(gold_records: list[Record]) -> list[ExperimentResult]:
    """Scenario 5: Semantic fields only — rules = 0 F1, LLMs > 0.8 F1."""
    print(f"\n{'='*60}")
    print(f"Scenario 5: Semantic fields only")
    print(f"{'='*60}")

    semantic_fields = [f for f in FIELD_NAMES
                      if FIELD_BY_NAME[f].category == FieldCategory.SEMANTIC]
    anomaly_fields = [f for f in FIELD_NAMES
                      if FIELD_BY_NAME[f].category == FieldCategory.ANOMALY]

    results = []

    # Rule-based: cannot extract semantic fields
    rule_eval = ExperimentResult(
        scenario="semantic", model_name="rule-based",
        field_f1=0.0,
        per_field_f1={f"rule_based_{f}": 0.0 for f in semantic_fields},
        extra={"semantic_fields": semantic_fields,
               "note": "Rules cannot produce semantic fields by definition"})
    results.append(rule_eval)
    _print_result(rule_eval)

    # Drain-only: cannot extract semantic fields
    drain_eval = ExperimentResult(
        scenario="semantic", model_name="drain-only",
        field_f1=0.0,
        extra={"note": "Drain produces templates, not semantic classification"})
    results.append(drain_eval)
    _print_result(drain_eval)

    # LLM placeholder
    llm_eval = ExperimentResult(
        scenario="semantic", model_name="flowfill-mdm (placeholder)",
        field_f1=0.85,
        per_field_f1={"f1_event_type": 0.88, "f1_affected_entity": 0.82, "f1_remediation_hint": 0.85},
        extra={"note": "Paper values — semantic fields are where LLMs add value"})
    results.append(llm_eval)
    _print_result(llm_eval)

    return results


def _print_result(r: ExperimentResult):
    print(f"  {r.model_name:30s} | F1={r.field_f1:.3f} | Valid={r.schema_valid_pct:.1f}% | "
          f"Cons={r.consistency_pct:.1f}% | Anom={r.anomaly_f1:.3f} | "
          f"p50={r.p50_ms:.1f}ms | p99.9={r.p999_ms:.1f}ms | "
          f"ev/s={r.events_per_s:.0f} | J={r.j_per_event:.1f}")


def main():
    parser = argparse.ArgumentParser(description="FlowFill baseline experiments")
    parser.add_argument("--logs", required=True, help="Raw log file (one line per event)")
    parser.add_argument("--gold", required=True, help="Gold records (JSONL)")
    parser.add_argument("--source", default="hdfs", choices=["hdfs", "bgl", "thunderbird", "spirit"])
    parser.add_argument("--scenario", default="all",
                        choices=["all", "clean", "drift", "perturbation", "cross_source", "semantic"])
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()

    log_lines = load_log_lines(args.logs)
    gold_records = load_records(args.gold)
    print(f"Loaded {len(log_lines)} log lines, {len(gold_records)} gold records")

    all_results: list[ExperimentResult] = []

    if args.scenario in ("all", "clean"):
        all_results.extend(run_scenario_clean(log_lines, gold_records, args.source))

    if args.scenario in ("all", "drift"):
        all_results.extend(run_scenario_drift(log_lines, gold_records, args.source))

    if args.scenario in ("all", "perturbation"):
        all_results.extend(run_scenario_perturbation(log_lines, gold_records, args.source))

    if args.scenario in ("all", "cross_source"):
        all_results.extend(run_scenario_cross_source(log_lines, gold_records, args.source))

    if args.scenario in ("all", "semantic"):
        all_results.extend(run_scenario_semantic(gold_records))

    print(f"\n{'='*60}")
    print(f"Total experiments: {len(all_results)}")
    print(f"{'='*60}")

    if args.output:
        output_data = [r.__dict__ for r in all_results]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
