# Baseline-Pricing Checklist

**How to check your architecture comparison for the five defects this repo
found in its own harness.** Apply these before publishing; each one takes
under five minutes and catches a failure that moved a headline number by
more than the effect it was reporting.

All five were found by **re-reading committed artifacts**, not by running new
experiments. Four of five biased the same direction — against the
autoregressive baseline. That is not a coincidence; it is the finding.

---

## Summary table

| # | Check | Symptom | Diagnostic | Bias direction | Magnitude |
|---|---|---|---|---|---|
| 1 | Decoding config | Baseline drops fields it should copy | Read `generation_config.json` | Against AR | 0.779 → 0.844 F1 |
| 2 | Serving parity | One arm runs a research loop | Name the serving stack | Against AR | 1841.6 → 413.3 ms (4.5×) |
| 3 | Parser policy | High validity, collapsing per-field F1 | Count multi-object outputs | Against AR | 0.599 → 0.826 F1 |
| 4 | Metric defaults | Exact 0% or 100% on a non-trivial cell | Feed synthetic falsy values | Against AR | 0% → 100% completeness |
| 5 | Provenance | Artifact names a model the code can't load | Grep the runner's write path | Neutral (correctness) | 3 files mislabelled |

---

## 1. Decoding configuration

- **Symptom:** the baseline drops fields it should copy verbatim from the
  input; the effect is stronger at smaller model sizes for no architectural
  reason.
- **Diagnostic:** print the effective sampling params actually used, not the
  ones you passed. Read the checkpoint's `generation_config.json`. HuggingFace
  `generate()` honours `repetition_penalty` **even under `do_sample=False`**.
  vLLM defaults it to 1.0; HF does not.
- **Fix:** set every decoding parameter explicitly on both arms; assert the
  effective values into the result artifact.
- **Magnitude:** Qwen2.5's default `repetition_penalty` (**1.1** at 1.5B,
  **1.05** at 3B/7B) emptied the `host` field on **80/100** events. AR-1.5B
  Field-F1 **0.779 → 0.844**. LLaDA was unaffected — no penalty in its config
  and a hand-written denoising loop — so the handicap was **one-sided**.
- **Traces to:** `results/vllm_serving_1.5b.json` (p50=413.3 ms, F1=0.844),
  `results/vllm_serving.md:18` (HF: p50=1841.6 ms, F1=0.779), #12.

## 2. Serving parity

- **Symptom:** a latency comparison where one arm runs a research loop and the
  other runs a production server, reported as an architecture result.
- **Diagnostic:** name the serving stack for each arm in one sentence. If the
  two sentences are not the same shape, you are measuring the stack.
- **Fix:** put both arms on a production-grade server, or state loudly that you
  are comparing implementations rather than architectures. Where no production
  server exists for one arm (as with masked diffusion at the time), say so and
  treat the number as a lower bound.
- **Magnitude:** naive HF loop → vLLM moved AR-1.5B p50 **1841.6 → 413.3 ms**
  (**4.46×**) and AR-7B **2122.0 → 1161.1 ms** (1.83×). It reversed the
  direction of the headline speed claim: properly served AR-1.5B is **3.36×
  faster** than LLaDA-8B, against a claimed 7.3× the other way.
- **Traces to:** `results/vllm_serving_1.5b.json` (vLLM: 413.3 ms),
  `results/vllm_serving.md:18` (HF: 1841.6 ms),
  `results/vllm_serving_3b.json`, `results/vllm_serving_7b.json`, #12, #22.

## 3. Parser policy under over-generation

- **Symptom:** schema validity stays high while per-field accuracy collapses.
  That combination is a parser signature, not a model signature — a model that
  is genuinely degrading loses validity too.
- **Diagnostic:** count how many outputs contain more than one JSON object
  (`ramble %`). Re-score the *same saved generations* under three parser
  policies: last-object-wins, first-object-wins, strict whole-output. **This
  requires that raw predictions were saved** — if they were not, the defect is
  undetectable and the check cannot be run.
- **Fix:** first-object-wins as the primary policy, strict as a reported
  secondary column, and write the policy into the result JSON rather than
  assuming it.
- **Magnitude:** at L=512 with `min_tokens = 512` forced, **96%** of AR
  outputs contained multiple JSON objects. The old parser took
  `text[text.find("{") : text.rfind("}")+1]` (first brace to **last** brace),
  and the regex fallback assigned `result[key]` in scan order, so the final,
  most-drifted copy of each field was scored. Re-scoring identical generations
  with first-object-wins moved AR-3B L=512 12-field F1 from **0.599 → 0.826**
  (**+0.23 F1**). On the 9-field metric: **0.600 → 0.852**. DiffusionGemma
  rambles 1–3% and is parser-invariant, so the defect hit one arm only.
- **Traces to:** `results/issue43/ar3b_L512_preds.jsonl` (raw generations),
  `results/issue43_nine_field_f1.json` (re-scored),
  `models/baselines/ar_model.py:194-230` (fixed first-object-wins parser),
  #43 (closed: "AR's result is a parser artifact, DG's is real"),
  `models/experiments/rescore_parsers.py`.

## 4. Metric behaviour on valid defaults

- **Symptom:** a completeness or coverage metric reads exactly **0%** in a
  cell where the model plainly produced output and finished inside the
  deadline. An exact 0.0 or 1.0 is a metric bug until proven otherwise.
- **Diagnostic:** feed the metric a hand-written perfect record containing
  legitimate falsy values (`""`, `0.0`, `False`) and confirm it scores 1.0.
  Separately, feed it `{}` and confirm it does **not** score well — in this
  repo `{}` scores 0.3167 on the 12-field macro, and two of three semantic
  fields score 1.000 for an empty prediction.
- **Fix:** distinguish *absent key* from *present key holding a falsy value*.
  Test the metric on synthetic inputs before it scores a model.
- **Magnitude:** the completeness metric compared `rec.get(f)` against the
  defaults `""` / `0.0` / `False`, which are structurally valid, not missing.
  AR-3B record completeness **0% → 100%** at D=1500 ms (and 0% → 97.5% at
  D=1000). The corrected numbers **refuted** the finding they had supported
  (#25's P-F7: diffusion does *not* beat AR on completeness).
- **Traces to:** `results/bounded_deadline_ar3b.json`,
  `results/bounded_deadline_report.md`, #38 (fixed in commit `f771683`),
  `metrics/quality_metrics.py` (completeness check).

## 5. Provenance

- **Symptom:** an artifact's `model` field names a model that branch of the
  code cannot have loaded.
- **Diagnostic:** for every result file, grep the runner for where each
  recorded field is *written*, and check it against where the model is
  *constructed*. A CLI default that survives into an artifact is the common
  form.
- **Fix:** emit provenance from the object that actually ran — model id,
  revision, runtime and version, GPU, dtype, and every decoding knob — via a
  shared helper no runner can bypass. Fail the build when a cited artifact
  lacks the block.
- **Magnitude:** not a latency or F1 shift — a **correctness-of-record**
  defect, which is worse. `models/experiments/run_perturbation.py:146`
  hardcodes `GSAI-ML/LLaDA-8B-Instruct` in the diffusion branch, while the
  argparse default `--model` is `Qwen/Qwen2.5-3B-Instruct` and the result dict
  at `:177` wrote that default into the artifact. Three result files therefore
  record `"model": "Qwen/Qwen2.5-3B-Instruct"` for runs of LLaDA-8B:
  `perturbation_dm_hf.json`, `bounded_deadline_dm_hf.json`,
  `refinement_dm_hf.json`. Correction sidecars written as `*.provenance.json`.
- **Traces to:** `results/perturbation_dm_hf.provenance.json`,
  `results/bounded_deadline_dm_hf.provenance.json`,
  `results/refinement_dm_hf.provenance.json`,
  `models/baselines/provenance.py` (shared helper), #57 R1.

---

## The asymmetry

Four of the five defects biased against the autoregressive baseline. The
mechanisms are different in each case, but they share a pattern: **the defect
lives in the infrastructure the AR model runs on, not in the model itself**,
and the experimenter was not rooting for the AR model.

| Defect | Mechanism of asymmetry |
|---|---|
| Decoding config | A default in the AR checkpoint's `generation_config.json`. LLaDA's hand-written denoising loop has no such default. |
| Serving parity | A production server exists for AR (vLLM) and not for diffusion. The AR number was inflated by a naive loop; the diffusion number was already on a production-grade engine. |
| Parser policy | The AR model rambles when forced to over-generate (`min_tokens` > natural length). DiffusionGemma stops at its canvas boundary. |
| Metric defaults | The AR model stops early (produces falsy defaults). The metric treated valid falsy values as missing. DiffusionGemma always fills all fields. |
| Provenance | Neutral — a code bug that mislabels the DM arm. Does not bias a comparison but makes audit impossible. |

A comparison harness written by someone hoping the new architecture wins
will fail asymmetrically. The failure is not random; it clusters on the arm
the experimenter was not rooting for, because that arm inherits the defaults,
the naive serving path, and the production-grade infrastructure of the
baseline it is trying to beat.

---

## How to use this checklist

1. **Before running any experiment**, answer the five questions below.
2. **After each run**, verify the five magnitudes against the artifacts.
3. **Before publishing**, confirm that every cited number traces to an
   artifact with a provenance block.

### Pre-experiment questions

1. **Decoding:** Have you set `repetition_penalty`, `temperature`,
   `top_p`, and `top_k` explicitly on both arms? Have you read
   `generation_config.json`?
2. **Serving:** Can you describe each arm's serving stack in one sentence
   of the same shape? If not, which arm is on a research loop?
3. **Parser:** Are you saving raw model output (not just parsed records)?
   Have you counted multi-object outputs?
4. **Metrics:** Have you fed your metrics synthetic perfect and empty
   records? Does `{}` score poorly?
5. **Provenance:** Does your result JSON record the model actually loaded,
   from the object that loaded it?

### Post-run verification commands

```bash
# 1. Check effective decoding params
python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); print('repetition_penalty:', getattr(c, 'repetition_penalty', 'not set'))"

# 2. Check serving stack is recorded
python3 -c "import json; d=json.load(open('results/your_result.json')); print(d.get('provenance',{}).get('backend','MISSING'))"

# 3. Count multi-object outputs (requires saved raw predictions)
python3 -c "
import json
n_multi = 0; n_total = 0
for line in open('results/your_preds.jsonl'):
    d = json.loads(line); raw = d.get('raw', '')
    n_total += 1
    if raw.count('{') > 1: n_multi += 1
print(f'ramble: {n_multi}/{n_total} ({100*n_multi/n_total:.0f}%)')
"

# 4. Test metric on synthetic inputs
python3 -c "
import sys; sys.path.insert(0, '.')
from metrics.quality_metrics import field_f1
from metrics.schema import Record
# Perfect record with falsy values
perfect = Record(fields={'timestamp_norm':'123','host':'','component':'X','severity':'INFO','event_type':'','message_template':'t','error_code':'','affected_entity':'','metric_value':0.0,'is_anomaly':False,'anomaly_score':0.0,'remediation_hint':'none'})
gold = Record(fields={'timestamp_norm':'123','host':'','component':'X','severity':'INFO','event_type':'','message_template':'t','error_code':'','affected_entity':'','metric_value':0.0,'is_anomaly':False,'anomaly_score':0.0,'remediation_hint':'none'})
print('perfect:', field_f1([perfect], [gold])['macro_field_f1_9'])
# Empty record
empty = Record(fields={})
print('empty:', field_f1([empty], [gold])['macro_field_f1_9'])
"

# 5. Verify provenance
python3 paper/verify_numbers.py
```
