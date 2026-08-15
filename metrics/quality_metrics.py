"""
Quality metrics: Field-F1, Schema-validity, Consistency, Anomaly-F1.

Implements the metrics defined in the FlowFill paper (§5, Table 2).
"""

from typing import Optional

from metrics.schema import FIELD_NAMES, FIELD_BY_NAME, Record, FieldType

# 9-field subset: 7 structural + 2 anomaly, per #57 R4.
# Drops event_type, affected_entity, remediation_hint (#53 gold problem).
NINE_FIELD_NAMES = [
    "timestamp_norm", "host", "component", "severity",
    "message_template", "error_code", "metric_value",
    "is_anomaly", "anomaly_score",
]


def _tokenize(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    return str(value).strip().split()


def field_f1_single(pred: Optional[str], gold: Optional[str]) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for a single field's token sequence."""
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return (1.0, 1.0, 1.0)

    pred_set: dict[str, int] = {}
    for t in pred_tokens:
        pred_set[t] = pred_set.get(t, 0) + 1

    gold_set: dict[str, int] = {}
    for t in gold_tokens:
        gold_set[t] = gold_set.get(t, 0) + 1

    tp = 0
    for t, count in pred_set.items():
        tp += min(count, gold_set.get(t, 0))

    fp = len(pred_tokens) - tp
    fn = len(gold_tokens) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return (precision, recall, f1)


def field_f1(pred_records: list[Record], gold_records: list[Record],
              fields_to_eval: Optional[list[str]] = None) -> dict[str, float]:
    """
    Compute macro-averaged Field-F1 across slots.

    Args:
        fields_to_eval: if set, restrict the macro average to these fields
            only (per-field F1 is still computed for all 12). Default None
            evaluates all fields (backward-compatible).

    Returns:
        dict with 'field_f1', per-field F1 scores, and 'macro_f1'
    """
    assert len(pred_records) == len(gold_records), "Record counts must match"

    eval_fields = fields_to_eval if fields_to_eval else FIELD_NAMES
    per_field_f1: dict[str, list[float]] = {name: [] for name in FIELD_NAMES}

    for pred_rec, gold_rec in zip(pred_records, gold_records):
        for field_name in FIELD_NAMES:
            pred_val = pred_rec.get(field_name)
            gold_val = gold_rec.get(field_name)
            _, _, f1 = field_f1_single(pred_val, gold_val)
            per_field_f1[field_name].append(f1)

    result: dict[str, float] = {}
    for field_name in FIELD_NAMES:
        scores = per_field_f1[field_name]
        result[f"f1_{field_name}"] = sum(scores) / len(scores) if scores else 0.0

    result["macro_field_f1"] = sum(result[f"f1_{n}"] for n in eval_fields) / len(eval_fields)

    # 9-field macro (7 structural + 2 anomaly) — #57 R4
    nine_scores = [result[f"f1_{n}"] for n in NINE_FIELD_NAMES if f"f1_{n}" in result]
    result["macro_field_f1_9"] = sum(nine_scores) / len(nine_scores) if nine_scores else 0.0

    # 12-field macro (all fields) — for continuity with published tables
    all_scores = [result[f"f1_{n}"] for n in FIELD_NAMES if f"f1_{n}" in result]
    result["macro_field_f1_12"] = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Backward compat: macro_field_f1 is the 12-field value (all fields)
    # unless fields_to_eval was explicitly set.
    if not fields_to_eval:
        result["macro_field_f1"] = result["macro_field_f1_12"]
    return result


def schema_validity(pred_records: list[Record],
                     skip_missing: Optional[list[str]] = None) -> float:
    """
    Compute Schema-valid %: percentage of records that are parseable,
    have all required keys, and correct types.

    Args:
        skip_missing: if set, fields in this list are not flagged as invalid
            when missing (for imputation where known fields are intentionally
            absent from the prediction). Default None checks all fields.
    """
    skip_set = set(skip_missing) if skip_missing else set()
    valid_count = 0

    for rec in pred_records:
        is_valid = True

        for field_name in FIELD_NAMES:
            field = FIELD_BY_NAME[field_name]

            if field.required and not rec.has(field_name):
                if field_name in skip_set:
                    continue
                is_valid = False
                break

            value = rec.get(field_name)
            if value is None:
                continue

            try:
                if field.type == FieldType.FLOAT:
                    float(value)
                elif field.type == FieldType.BOOL:
                    if isinstance(value, str):
                        if value.lower() not in ("true", "false", "0", "1"):
                            is_valid = False
                            break
                elif field.type == FieldType.ENUM:
                    if field.enum_values and str(value) not in field.enum_values:
                        is_valid = False
                        break
            except (ValueError, TypeError):
                is_valid = False
                break

        if is_valid:
            valid_count += 1

    return valid_count / len(pred_records) * 100 if pred_records else 0.0


def _check_consistency(rec: Record) -> bool:
    """
    Check inter-field constraints (3 rules over 4 fields, all involving severity).

    Constraints:
      R1: severity ∈ {ERROR,FATAL,CRITICAL} ⇒ is_anomaly == true
      R2: severity ∈ {INFO,DEBUG} ⇒ anomaly_score ≤ 0.8;
          severity ∈ {FATAL,CRITICAL} ⇒ anomaly_score ≥ 0.5
      R3: error_code non-empty ⇒ severity != "INFO"

    Note: the docstring previously advertised a "component ↔ error_code"
    constraint. It does not exist — `component` was read into a local but
    never used (dead variable, issue #54). Removed.
    """
    severity = rec.get("severity")
    is_anomaly = rec.get("is_anomaly")
    anomaly_score = rec.get("anomaly_score")
    error_code = rec.get("error_code")

    # Constraint 1: severity ↔ is_anomaly
    if severity and is_anomaly is not None:
        anomaly_bool = is_anomaly if isinstance(is_anomaly, bool) else str(is_anomaly).lower() == "true"
        if severity in ("ERROR", "FATAL", "CRITICAL") and not anomaly_bool:
            return False
        if severity in ("INFO", "DEBUG") and anomaly_bool and anomaly_score is None:
            pass  # Not a hard violation — INFO + anomaly is rare but possible

    # Constraint 2: anomaly_score ↔ severity
    if severity and anomaly_score is not None:
        try:
            score = float(anomaly_score)
            if severity in ("INFO", "DEBUG") and score > 0.8:
                return False
            if severity in ("FATAL", "CRITICAL") and score < 0.5:
                return False
        except (ValueError, TypeError):
            pass

    # Constraint 3: error_code present implies non-INFO severity
    if error_code and severity == "INFO":
        return False

    return True


def consistency(pred_records: list[Record]) -> float:
    """Compute Consistency %: percentage of records satisfying inter-field constraints."""
    if not pred_records:
        return 0.0
    consistent_count = sum(1 for rec in pred_records if _check_consistency(rec))
    return consistent_count / len(pred_records) * 100


def anomaly_f1(pred_records: list[Record], gold_records: list[Record]) -> dict[str, float]:
    """
    Compute Anomaly-F1: binary F1 on the is_anomaly slot.
    """
    assert len(pred_records) == len(gold_records), "Record counts must match"

    tp = fp = fn = 0

    for pred_rec, gold_rec in zip(pred_records, gold_records):
        def _to_bool(v) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.lower() == "true"
            return bool(v)

        pred_anomaly = _to_bool(pred_rec.get("is_anomaly"))
        gold_anomaly = _to_bool(gold_rec.get("is_anomaly"))

        if pred_anomaly and gold_anomaly:
            tp += 1
        elif pred_anomaly and not gold_anomaly:
            fp += 1
        elif not pred_anomaly and gold_anomaly:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"anomaly_precision": precision, "anomaly_recall": recall, "anomaly_f1": f1}
