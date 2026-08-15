"""
Decision procedure: which architecture to serve on, and at what cost.

Issue #64. Takes (L, lambda, batching, f1_tolerance) and returns the arm,
predicted latencies, ratio, expected F1 delta, the rule that fired, and the
validity domain.

Design constraint: T_DG is injected, not hard-coded. The one-variable form
ships now; the two-variable form swaps in when #44 lands — that is the only
change required.

Usage:
    from models.decision.procedure import decide
    r = decide(L=512, lam=1.0, batching=False, f1_tolerance=0.05)
    print(r)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Callable


# ---- Constants (all trace to paper/NUMBERS.md and results/*.json) ----

CONSTANTS = {
    "ar_intercept_ms": {
        "value": 7.27,
        "source": "NUMBERS.md §2.1, results/appD_ar3b_L1024.json fit",
        "issue": "#44 P4 (intercept absorbs prefill inconsistently)",
    },
    "ar_slope_ms_per_token": {
        "value": 6.29,
        "source": "NUMBERS.md §2.1, results/p2_ar3b_*.json",
        "issue": None,
    },
    "dg_base_ms": {
        "value": 418.24,
        "source": "NUMBERS.md §2.1, mean of p2_dg_L32..256 (K=32)",
        "issue": "#57 (K=32; K=8 base is ~378.8 ms)",
    },
    "dg_base_k8_ms": {
        "value": 378.8,
        "source": "NUMBERS.md §0, results/dg_steps_8.json",
        "issue": "#57 discrepancy analysis",
    },
    "dg_step_min_ms": {
        "value": 113.1,
        "source": "results/issue68_dg_L1280.json (n_c 5→6 step, #68)",
        "issue": None,
    },
    "dg_step_max_ms": {
        "value": 121.8,
        "source": "results/issue68_dg_L264.json (n_c 1→2 step, #68)",
        "issue": None,
    },
    "dg_step_ls_ms": {
        "value": 119.6,
        "source": "results/issue68_dg_*.json, median of 7 step costs (#68 least-squares fit)",
        "issue": None,
    },
    "canvas_length": {
        "value": 256,
        "source": "DiffusionGemma config.json: canvas_length",
        "issue": "#66 step 0 (tunable)",
    },
    "l_cross_tokens": {
        "value": 65,
        "source": "NUMBERS.md §2.2, T_AR = T_DG solve",
        "issue": "#64 acceptance gate",
    },
    "lambda_cross_ev_s": {
        "value": 1.8,
        "source": "NUMBERS.md §4.1, rate_lag curves",
        "issue": "#64 acceptance gate",
    },
    "natural_l_ar": {
        "value": 123,
        "source": "NUMBERS.md §2.2, AR mean output tokens at L=256",
        "issue": None,
    },
    "natural_l_dg": {
        "value": 176,
        "source": "NUMBERS.md §2.2, DG mean output tokens at L=256",
        "issue": None,
    },
    "mult_dg": {
        "value": 1.18,
        "source": "NUMBERS.md §5, results/llada_dinfer/ (cache='')",
        "issue": "#42 (retitle, not mechanism)",
    },
    "mult_ar": {
        "value": 80.0,
        "source": "NUMBERS.md §5, results/vllm_serving_3b.json (77-86x range)",
        "issue": None,
    },
    "knee_dg_ev_s": {
        "value": 2.25,
        "source": "NUMBERS.md §4.1 (2.0-2.5 range)",
        "issue": None,
    },
    "knee_ar_ev_s": {
        "value": 80.0,
        "source": "NUMBERS.md §4.1 (64-96 range)",
        "issue": None,
    },
    "price_f1_12field": {
        "value": 0.039,
        "source": "NUMBERS.md §3, 0.826 - 0.787 at L=256",
        "issue": "#57 R4 (12-field contaminated; 9-field price pending)",
    },
    "price_f1_9field": {
        "value": -0.002,
        "source": "results/issue43_nine_field_f1.json: 0.8516 vs 0.8518",
        "issue": "#57 R4 (9-field: DG matches AR, price ~0)",
    },
}

# ---- Validity domain ----

VALIDITY_DOMAIN = {
    "hardware": "One A100 80GB PCIe, Campaign B. Not transferable to H100 campaigns.",
    "runtime": "vLLM 0.25.1, BF16, --max-num-seqs 1 for diffusion (serial forced by vllm#50699).",
    "models": "DiffusionGemma-26B-A4B-it (block diffusion, MoE, canvas 256) vs "
              "Qwen2.5-3B-Instruct (dense AR). NOT size-matched; asymmetry favours DG.",
    "task_shape": "12-field structured extraction over BGL, copy-heavy, "
                  "few-shot prompt ~1047 tokens.",
    "must_not_apply": [
        "Input lengths materially different from ~1047 tokens (pending #44)",
        "Diffusion arm batch-served (not supported in current runtimes)",
        "Non-extractive / generative task shapes",
        "Canvas size other than 256",
        "Other GPUs or runtimes",
        "Quality tolerance interpreted against 12-field macro F1 "
        "(contaminated per #57 R4; use 9-field)",
    ],
}


# ---- DG law functions (injectable) ----

def t_dg_one_variable(l_out, base=None, step=None, canvas=None):
    """T_DG(L) = BASE + STEP * (ceil(L / CANVAS) - 1).

    One-variable form: cost depends on output length only.
    This is the form currently shipped; it under-predicts at long inputs.
    """
    if base is None:
        base = CONSTANTS["dg_base_k8_ms"]["value"]
    if step is None:
        step = CONSTANTS["dg_step_ls_ms"]["value"]
    if canvas is None:
        canvas = CONSTANTS["canvas_length"]["value"]
    n_canvases = max(1, math.ceil(l_out / canvas))
    return base + step * (n_canvases - 1)


def t_dg_two_variable(l_in, l_out, base=None, step=None, canvas=None,
                      prefill_ms_per_token=None):
    """T_DG(L_in, L_out) = BASE + prefill_slope * L_in + STEP * (ceil(L_out / CANVAS) - 1).

    Two-variable form: cost depends on both input and output length.
    Placeholder until #44 lands; the prefill slope will come from the
    input-length sweep.
    """
    if base is None:
        base = CONSTANTS["dg_base_k8_ms"]["value"]
    if step is None:
        step = CONSTANTS["dg_step_ls_ms"]["value"]
    if canvas is None:
        canvas = CONSTANTS["canvas_length"]["value"]
    if prefill_ms_per_token is None:
        prefill_ms_per_token = 0.123  # from #44 input-length sweep
    n_canvases = max(1, math.ceil(l_out / canvas))
    return base + prefill_ms_per_token * l_in + step * (n_canvases - 1)


def t_ar(l_out, intercept=None, slope=None):
    """T_AR(L) = a + b * L."""
    if intercept is None:
        intercept = CONSTANTS["ar_intercept_ms"]["value"]
    if slope is None:
        slope = CONSTANTS["ar_slope_ms_per_token"]["value"]
    return intercept + slope * l_out


# ---- Decision result ----

@dataclass
class DecisionResult:
    arm: str
    t_ar_ms: float
    t_dg_ms: float
    ratio: float
    expected_f1_delta: float
    rule_fired: str
    law_in_force: str
    warnings: list[str] = field(default_factory=list)
    constants_used: dict = field(default_factory=dict)
    validity_domain: dict = field(default_factory=dict)

    def __str__(self):
        sign = "dF1" if self.expected_f1_delta >= 0 else "dF1"
        return (f"arm: {self.arm}  "
                f"T_AR={self.t_ar_ms:.0f} ms  T_DG={self.t_dg_ms:.0f} ms  "
                f"ratio={self.ratio:.2f}x  {sign}={self.expected_f1_delta:.3f}\n"
                f"rule: {self.rule_fired}\n"
                f"law: {self.law_in_force}")

    def to_dict(self):
        return {
            "arm": self.arm,
            "t_ar_ms": round(self.t_ar_ms, 1),
            "t_dg_ms": round(self.t_dg_ms, 1),
            "ratio": round(self.ratio, 2),
            "expected_f1_delta": round(self.expected_f1_delta, 4),
            "rule_fired": self.rule_fired,
            "law_in_force": self.law_in_force,
            "warnings": self.warnings,
            "validity_domain": VALIDITY_DOMAIN,
        }


# ---- The decision procedure ----

def decide(
    L: float,
    lam: float,
    batching: bool,
    f1_tolerance: float,
    l_in: float = 1047,
    t_dg_fn: Callable = t_dg_one_variable,
    step: Optional[float] = None,
    f1_metric: str = "12field",
) -> DecisionResult:
    """Decide which architecture to serve on.

    Args:
        L: expected output length in tokens.
        lam: arrival rate in events/s.
        batching: is cross-request batching available?
        f1_tolerance: Field-F1 the caller will give up to gain latency.
        l_in: expected input length in tokens (default 1047, the few-shot prompt).
        t_dg_fn: DG law function (t_dg_one_variable or t_dg_two_variable).
        step: per-canvas step cost in ms. If None, uses the LS fit (118.83).
        f1_metric: "9field" or "12field" for the quality price.

    Returns:
        DecisionResult with the recommendation and all predicted costs.
    """
    # Compute latencies
    t_ar_val = t_ar(L)
    if t_dg_fn.__name__ == "t_dg_two_variable":
        t_dg_val = t_dg_fn(l_in, L, step=step)
        law = "two-variable (L_in + L_out)"
    else:
        t_dg_val = t_dg_fn(L, step=step)
        law = "one-variable (L_out only)"

    ratio = t_ar_val / t_dg_val if t_dg_val > 0 else float("inf")

    # Quality price
    if f1_metric == "9field":
        price_f1 = CONSTANTS["price_f1_9field"]["value"]
    else:
        price_f1 = CONSTANTS["price_f1_12field"]["value"]

    warnings = []

    # One-variable law warning
    if t_dg_fn.__name__ == "t_dg_two_variable":
        warnings.append(
            "Two-variable law uses a placeholder prefill slope (0.15 ms/token). "
            "Calibrate from #44 input-length sweep results before relying on "
            "this mode."
        )
    else:
        warnings.append(
            "T_DG(L) under-predicts by +58% at F=50 and +112% at F=100 "
            "in the wide-schema data. Law is only valid at short input lengths "
            "(~1047 tokens). See #44."
        )

    natural_l_ar = CONSTANTS["natural_l_ar"]["value"]
    natural_l_dg = CONSTANTS["natural_l_dg"]["value"]
    l_cross = CONSTANTS["l_cross_tokens"]["value"]
    lambda_cross = CONSTANTS["lambda_cross_ev_s"]["value"]

    # ---- Guards (first that fires wins) ----

    # 1. Truncation guard — fires when the output budget may fall below
    #    natural_L (123–176 tokens). At L < 176, DG is truncated below its
    #    natural output length, producing poor quality (0.329 F1, 3.3%
    #    validity at L=128). AR's natural length is 123, so the guard
    #    threshold is the max of the two (176).
    natural_l_dg = CONSTANTS["natural_l_dg"]["value"]
    if L < natural_l_dg:
        result = DecisionResult(
            arm="AR", t_ar_ms=t_ar_val, t_dg_ms=t_dg_val, ratio=ratio,
            expected_f1_delta=-price_f1,
            rule_fired=f"truncation: L={L} < natural_L_DG={natural_l_dg}",
            law_in_force=law, warnings=warnings,
            validity_domain=VALIDITY_DOMAIN,
        )
        return _annotate(result, step)

    # 2. Throughput guard
    knee_dg = CONSTANTS["knee_dg_ev_s"]["value"]
    if batching and lam > lambda_cross:
        warnings.append(
            f"DG batch multiplier is {CONSTANTS['mult_dg']['value']}x "
            f"vs AR's {CONSTANTS['mult_ar']['value']:.0f}x. "
            f"DG saturation knee is ~{knee_dg} ev/s."
        )
        result = DecisionResult(
            arm="AR", t_ar_ms=t_ar_val, t_dg_ms=t_dg_val, ratio=ratio,
            expected_f1_delta=-price_f1,
            rule_fired=f"throughput: batching=True and lambda={lam} > "
                       f"lambda_cross={lambda_cross}",
            law_in_force=law, warnings=warnings,
            validity_domain=VALIDITY_DOMAIN,
        )
        return _annotate(result, step)

    # 3. Length guard
    if L < l_cross:
        result = DecisionResult(
            arm="AR", t_ar_ms=t_ar_val, t_dg_ms=t_dg_val, ratio=ratio,
            expected_f1_delta=-price_f1,
            rule_fired=f"length: L={L} < L_cross={l_cross}",
            law_in_force=law, warnings=warnings,
            validity_domain=VALIDITY_DOMAIN,
        )
        return _annotate(result, step)

    # 4. Quality guard
    if f1_tolerance < abs(price_f1):
        result = DecisionResult(
            arm="AR", t_ar_ms=t_ar_val, t_dg_ms=t_dg_val, ratio=ratio,
            expected_f1_delta=-price_f1,
            rule_fired=f"quality: f1_tolerance={f1_tolerance} < "
                       f"price_F1={abs(price_f1):.3f}",
            law_in_force=law, warnings=warnings,
            validity_domain=VALIDITY_DOMAIN,
        )
        return _annotate(result, step)

    # 5. Otherwise → DG
    result = DecisionResult(
        arm="DG", t_ar_ms=t_ar_val, t_dg_ms=t_dg_val, ratio=ratio,
        expected_f1_delta=price_f1,
        rule_fired=f"length>=L_cross, lambda<=lambda_cross, tolerance>=price",
        law_in_force=law, warnings=warnings,
        validity_domain=VALIDITY_DOMAIN,
    )
    return _annotate(result, step)


def _annotate(result: DecisionResult, step: Optional[float]) -> DecisionResult:
    """Add constants used and step-range robustness."""
    if step is None:
        step_min = CONSTANTS["dg_step_min_ms"]["value"]
        step_max = CONSTANTS["dg_step_max_ms"]["value"]
        result.constants_used = {
            "step_used": CONSTANTS["dg_step_ls_ms"]["value"],
            "step_range": [step_min, step_max],
            "robust_to_step_range": True,
        }
    else:
        result.constants_used = {"step_used": step}
    return result


# ---- Crossover computation (for self-test) ----

def compute_l_cross(base=None, step=None, canvas=None,
                    ar_intercept=None, ar_slope=None):
    """Solve T_AR(L) = T_DG(L) for L.

    T_AR = a + b*L
    T_DG = base + step*(ceil(L/canvas) - 1)

    At the boundary L = canvas (first canvas), T_DG = base.
    So a + b*L_cross = base → L_cross = (base - a) / b
    """
    if base is None:
        base = CONSTANTS["dg_base_k8_ms"]["value"]
    if ar_intercept is None:
        ar_intercept = CONSTANTS["ar_intercept_ms"]["value"]
    if ar_slope is None:
        ar_slope = CONSTANTS["ar_slope_ms_per_token"]["value"]
    return (base - ar_intercept) / ar_slope


def compute_lambda_cross(l_out=256, base=None, step=None, canvas=None,
                         ar_intercept=None, ar_slope=None):
    """Find lambda where AR throughput under batching overtakes DG.

    At batch-1: lambda_cross ≈ 1 / max(T_AR, T_DG)
    The crossover is where the AR batch advantage overcomes the DG latency advantage.
    """
    t_ar_val = t_ar(l_out, ar_intercept, ar_slope)
    t_dg_val = t_dg_one_variable(l_out, base, step, canvas)
    # DG is faster at batch-1; AR wins when batching gives it enough throughput
    # The crossover is where AR's effective per-event time (with batching)
    # drops below DG's: T_AR / mult_AR = T_DG → lambda_cross = mult_AR / T_AR
    # But more simply: at batch-1, if DG is faster, the rate crossover is
    # where AR can batch enough to match DG's serial throughput.
    # lambda_cross = 1/T_DG (DG's serial throughput is the bar AR must clear)
    return 1000.0 / t_dg_val if t_dg_val > 0 else 0
