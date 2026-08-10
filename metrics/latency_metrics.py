"""
Latency and serving metrics: percentiles, throughput under SLA, energy per event.

Implements the metrics defined in the FlowFill paper (§5-6, Tables 3-4).
"""

import statistics
from dataclasses import dataclass


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """
    Compute p50, p90, p99, p99.9 latency percentiles.

    Args:
        latencies_ms: list of per-event wall-clock latencies in milliseconds

    Returns:
        dict with p50, p90, p99, p99.9 in milliseconds
    """
    if not latencies_ms:
        return {"p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0, "p99_9_ms": 0.0}

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    def _percentile(p: float) -> float:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        return sorted_lat[idx]

    return {
        "p50_ms": _percentile(50),
        "p90_ms": _percentile(90),
        "p99_ms": _percentile(99),
        "p99_9_ms": _percentile(99.9),
    }


@dataclass
class ThroughputResult:
    max_batch_at_sla: int
    events_per_s: float
    tokens_per_s: float


def throughput_at_sla(
    latencies_ms: list[float],
    tokens_per_event: int = 128,
    sla_p99_ms: float = 250.0,
    batch_sizes: list[int] | None = None,
) -> ThroughputResult:
    """
    Find the max batch size that satisfies the p99 SLA, then compute throughput.

    Args:
        latencies_ms: per-event latencies at the candidate batch size
        tokens_per_event: avg tokens per record (~128 per paper schema)
        sla_p99_ms: p99 latency SLA in milliseconds (default: 250)
        batch_sizes: candidate batch sizes to search (default: 1-128)

    Returns:
        ThroughputResult with max_batch, events/s, tokens/s
    """
    if batch_sizes is None:
        batch_sizes = list(range(1, 129))

    best_batch = 1
    best_events_s = 0.0
    best_tokens_s = 0.0

    percentiles = latency_percentiles(latencies_ms)
    p99 = percentiles["p99_ms"]

    for batch in batch_sizes:
        if p99 <= sla_p99_ms:
            p50 = percentiles["p50_ms"]
            events_s = batch * (1000.0 / p50) if p50 > 0 else 0
            tokens_s = events_s * tokens_per_event

            if events_s > best_events_s:
                best_batch = batch
                best_events_s = events_s
                best_tokens_s = tokens_s

    return ThroughputResult(
        max_batch_at_sla=best_batch,
        events_per_s=best_events_s,
        tokens_per_s=best_tokens_s,
    )


def energy_per_event(
    avg_power_watts: float,
    duration_s: float,
    num_events: int,
) -> float:
    """
    Compute energy per event in Joules.

    Args:
        avg_power_watts: board-level average GPU power draw in watts
        duration_s: total measurement duration in seconds
        num_events: number of events processed during that duration

    Returns:
        Joules per event
    """
    total_energy_j = avg_power_watts * duration_s
    return total_energy_j / num_events if num_events > 0 else 0.0
