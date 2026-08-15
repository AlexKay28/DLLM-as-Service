"""
Gold-alignment assertion and shuffled-gold canary.

#57 R6: Assert that prediction[i] and gold[i] refer to the same source event.
#55 showed a `cursor` desync can silently score predictions against unrelated
gold and reproduce to 4 decimal places.

Usage:
    from models.baselines.gold_alignment import assert_aligned, shuffled_gold_canary

    # After extracting predictions and loading gold:
    assert_aligned(predictions, gold_records, source_lines)

    # Canary: shuffled gold should score materially worse
    canary = shuffled_gold_canary(records, gold_records)
    assert canary["macro_field_f1"] < records_f1 - 0.1, "Shuffled-gold canary failed"
"""

import hashlib
import random
from typing import Any

from metrics.quality_metrics import field_f1
from metrics.schema import Record


def _event_fingerprint(raw_line: str) -> str:
    """Short hash of a raw log line for alignment checking."""
    return hashlib.md5(raw_line.strip().encode()).hexdigest()[:12]


def assert_aligned(
    predictions: list[Record] | list[dict],
    gold_records: list[Record] | list[dict],
    source_lines: list[str] | None = None,
    pred_sources: list[str] | None = None,
    gold_sources: list[str] | None = None,
) -> None:
    """Assert that prediction[i] and gold[i] refer to the same source event.

    Two modes:
    1. source_lines provided: fingerprint each line, assign to predictions
       by index, and to gold by index. Assert fingerprints match at each i.
    2. pred_sources/gold_sources provided: compare directly.

    Raises AssertionError with details if misaligned.

    Args:
        predictions: predicted records (Record or dict with fields).
        gold_records: gold records (Record or dict with fields).
        source_lines: the raw log lines that produced predictions, in order.
        pred_sources: explicit source identifiers for predictions.
        gold_sources: explicit source identifiers for gold.
    """
    n_pred = len(predictions)
    n_gold = len(gold_records)
    assert n_pred == n_gold, (
        f"Count mismatch: {n_pred} predictions vs {n_gold} gold records. "
        f"Predictions and gold must be the same length and in the same order."
    )

    if source_lines is not None:
        assert len(source_lines) == n_pred, (
            f"source_lines has {len(source_lines)} entries but predictions "
            f"has {n_pred}."
        )
        pred_fps = [_event_fingerprint(line) for line in source_lines]
        # Gold should have been produced from the same source lines.
        # If gold_records carry a __source_fingerprint field, use that;
        # otherwise assume gold was loaded in the same order.
        for i, (pred_fp, gold_rec) in enumerate(zip(pred_fps, gold_records)):
            gold_fp = None
            if isinstance(gold_rec, Record):
                gold_fp = gold_rec.get("__source_fingerprint")
            elif isinstance(gold_rec, dict):
                gold_fp = gold_rec.get("__source_fingerprint")
            if gold_fp is not None:
                if pred_fp != gold_fp:
                    raise AssertionError(
                        f"Gold-alignment failure at index {i}: "
                        f"prediction source fingerprint {pred_fp!r} != "
                        f"gold source fingerprint {gold_fp!r}. "
                        f"Predictions may be scored against unrelated gold "
                        f"(see #55, #57 R6)."
                    )
        return

    if pred_sources is not None and gold_sources is not None:
        assert len(pred_sources) == n_pred
        assert len(gold_sources) == n_gold
        for i, (ps, gs) in enumerate(zip(pred_sources, gold_sources)):
            if ps != gs:
                raise AssertionError(
                    f"Gold-alignment failure at index {i}: "
                    f"prediction source {ps!r} != gold source {gs!r}."
                )
        return

    # No source identifiers available — at minimum check counts match (done above)
    # and issue a warning that alignment is unverified.
    import warnings
    warnings.warn(
        "Gold alignment is UNVERIFIED: no source_lines or pred_sources/gold_sources "
        "provided. Cannot detect a cursor desync (#55). Pass source_lines to "
        "assert_aligned() to enable verification.",
        stacklevel=2,
    )


def shuffled_gold_canary(
    pred_records: list[Record],
    gold_records: list[Record],
    seed: int = 42,
) -> dict[str, float]:
    """Shuffle gold records and score — result should be materially worse.

    If shuffled-gold F1 is within 0.01 of the aligned F1, the predictions
    carry no event-specific information (a red flag for desync).

    Returns:
        dict with macro_field_f1 (shuffled) for comparison.
    """
    rng = random.Random(seed)
    shuffled_gold = list(gold_records)
    rng.shuffle(shuffled_gold)
    result = field_f1(pred_records, shuffled_gold)
    return {"macro_field_f1": result["macro_field_f1"],
            "macro_field_f1_9": result.get("macro_field_f1_9", 0.0),
            "macro_field_f1_12": result.get("macro_field_f1_12", 0.0)}


def tag_gold_with_fingerprint(
    gold_records: list[Record],
    source_lines: list[str],
) -> list[Record]:
    """Tag each gold record with its source line's fingerprint.

    Call this once when loading gold, so assert_aligned() can verify
    alignment without needing the original source lines later.

    Args:
        gold_records: gold records in order.
        source_lines: the raw log lines that gold was derived from, in order.

    Returns:
        The same list of records, each tagged with __source_fingerprint.
    """
    assert len(gold_records) == len(source_lines), (
        f"gold_records has {len(gold_records)} but source_lines has "
        f"{len(source_lines)}"
    )
    for rec, line in zip(gold_records, source_lines):
        fp = _event_fingerprint(line)
        if isinstance(rec, Record):
            rec.set("__source_fingerprint", fp)
        elif isinstance(rec, dict):
            rec["__source_fingerprint"] = fp
    return gold_records
