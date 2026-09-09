"""Re-score saved primary-pair outputs and count generation-budget completions.

The reference slice is frozen from the local prepared test split; no inference
or new parsing is performed. Run --write to update version-local audit outputs,
or --check to verify them without changes.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from metrics.quality_metrics import NINE_FIELD_NAMES, field_f1, field_f1_single, schema_validity
from metrics.schema import Record


def rows(path, n=None):
    with path.open() as stream:
        return [json.loads(line) for line in itertools.islice(stream, n)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    ref = HERE/'data/bgl_reference_300.jsonl'
    gold = [Record(fields=x) for x in rows(ref)]
    assert len(gold) == 300
    ar_paths = [REPO/f'results/anchors/ar30b_L256_s{s}_preds.jsonl' for s in (42, 137, 314)]
    dg_path = REPO/'results/anchors/dg_L256_r1_preds.jsonl'
    ar = rows(ar_paths[0]); dg = rows(dg_path)
    assert len(ar) == 300 and len(dg) == 2000
    paired = {'ar30b': ar, 'dg': dg[:300]}
    quality = {}
    scores = {}
    for name, items in paired.items():
        recs = [Record(fields=x['record']) for x in items]
        q = field_f1(recs, gold)
        scores[name] = [sum(field_f1_single(p.get(k), g.get(k))[2]
                            for k in NINE_FIELD_NAMES)/9 for p, g in zip(recs, gold)]
        quality[name] = {'n': len(items), 'field_f1_9': q['macro_field_f1_9'],
                         'schema_valid_pct': schema_validity(recs),
                         'median_latency_ms': float(np.median([x['latency_ms'] for x in items])),
                         'per_field_f1': {k: q['f1_'+k] for k in NINE_FIELD_NAMES}}
    # Reproducing the archived AR score also checks the frozen reference order.
    archived = json.loads((REPO/'results/anchors/ar30b_L256_s42.json').read_text())
    assert abs(quality['ar30b']['field_f1_9']-archived['field_f1_9']) < .00005
    delta = np.array(scores['dg'])-np.array(scores['ar30b'])
    rng = np.random.default_rng(42)
    boot = rng.choice(delta, size=(10000, len(delta))).mean(axis=1)
    budgets = {}
    pooled = {'ar30b': sum([rows(p) for p in ar_paths], []), 'dg': dg}
    for name, items in pooled.items():
        valid = [schema_validity([Record(fields=x['record'])]) == 100 for x in items]
        budgets[name] = {'n': len(items), 'thresholds_ms': {}}
        for limit in (400, 500, 750, 1000):
            completed = [x['latency_ms'] <= limit for x in items]
            n = sum(completed); nv = sum(a and b for a, b in zip(completed, valid))
            budgets[name]['thresholds_ms'][str(limit)] = {
                'completed': n, 'completed_pct': n/len(items)*100,
                'valid_completed': nv, 'valid_completed_pct': nv/len(items)*100}
    return {
        'scope': 'Post-hoc analysis of saved outputs; generation time only, no new inference.',
        'pairing': 'First 300 test records; AR seed 42 and DG repeat 1 retain input order in their runners.',
        'reference': 'Parser-derived nine-field evaluation; semantic action fields excluded.',
        'sources_sha256': {str(p.relative_to(REPO)): digest(p) for p in [ref, *ar_paths, dg_path,
            REPO/'metrics/quality_metrics.py', REPO/'metrics/schema.py']},
        'quality': quality,
        'paired_difference_dg_minus_ar': {
            'mean': float(delta.mean()), 'bootstrap_ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
            'resamples': 10000, 'seed': 42,
            'interpretation': 'Event-paired percentile bootstrap, conditional on this sample; no equivalence margin selected.'},
        'generation_budgets': budgets,
    }


def outputs(result):
    q = result['quality']; budgets = result['generation_budgets']
    quality_rows = []
    for label, key, fmt in [('Nine-field F1', 'field_f1_9', '.3f'),
                            ('Schema-valid records (\\%)', 'schema_valid_pct', '.1f'),
                            ('Paired-sample median (ms)', 'median_latency_ms', '.1f')]:
        quality_rows.append(f"{label} & {q['ar30b'][key]:{fmt}} & {q['dg'][key]:{fmt}} \\\\")
    budget_rows = []
    for limit in ('500', '1000'):
        a = budgets['ar30b']['thresholds_ms'][limit]; d = budgets['dg']['thresholds_ms'][limit]
        budget_rows.append(f"{limit} & {a['completed_pct']:.2f} & {d['completed_pct']:.2f} & "
                           f"{a['valid_completed_pct']:.2f} & {d['valid_completed_pct']:.2f} \\\\")
    return {'primary_pair_audit.json': json.dumps(result, indent=2)+'\n',
            'primary_quality_rows.tex': '\\begin{tabular}{@{}lrr@{}}\n\\toprule\nMeasure & AR-30B & DG \\\\\n\\midrule\n'+'\n'.join(quality_rows)+'\n\\bottomrule\n\\end{tabular}\n',
            'generation_budget_rows.tex': '\\begin{tabular}{@{}rrrrr@{}}\n\\toprule\n& \\multicolumn{2}{c}{Completed} & \\multicolumn{2}{c}{Valid and completed} \\\\\nBudget (ms) & AR-30B & DG & AR-30B & DG \\\\\n\\midrule\n'+'\n'.join(budget_rows)+'\n\\bottomrule\n\\end{tabular}\n'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-reference', action='store_true')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true'); mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.freeze_reference:
        if not args.write: parser.error('--freeze-reference requires --write')
        dest = HERE/'data/bgl_reference_300.jsonl'
        dest.parent.mkdir(exist_ok=True)
        # Never overwrite a previously frozen reference slice.
        content = ''.join(json.dumps(x, sort_keys=True)+'\n' for x in
                          rows(REPO/'datasets/splits/test/bgl.jsonl', 300))
        if dest.exists():
            assert dest.read_text() == content, 'Reference slice differs from the local split'
        else:
            dest.write_text(content)
    result = audit()
    for name, content in outputs(result).items():
        path = HERE/'data'/name
        if args.write: path.write_text(content)
        else: assert path.read_text() == content, f'Stale audit output: {name}'
    print(json.dumps({'quality': {k: v['field_f1_9'] for k,v in result['quality'].items()},
                      'difference': result['paired_difference_dg_minus_ar'],
                      'generation_budgets': result['generation_budgets']}, indent=2))


if __name__ == '__main__':
    main()
