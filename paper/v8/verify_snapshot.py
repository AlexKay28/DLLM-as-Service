#!/usr/bin/env python3
"""Audit selected v8 claims and all local TeX dependencies; does not run models.

Checks primary macros against their artifacts, recomputes descriptive timing
statistics from saved events, validates energy arithmetic, and checks figure,
input, label and citation references. This is not a full scientific validation.
"""
from pathlib import Path
import json
import math
import re
import statistics

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
CHECKS=[]
def check(condition,message):
    if not condition: raise AssertionError(message)
    CHECKS.append(message)
def read(path): return json.loads((REPO/path).read_text())
def close(a,b,tol=.051):return abs(a-b)<=tol
def percentile(values,p):
    v=sorted(values);pos=(len(v)-1)*p/100;lo=math.floor(pos);hi=math.ceil(pos)
    return v[lo]+(v[hi]-v[lo])*(pos-lo)
def lats(path):return [json.loads(x)['latency_ms'] for x in (REPO/path).read_text().splitlines() if x.strip()]

def main():
    macro_text=(HERE/'assets/macros.tex').read_text()
    macros=dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^{}]*)\}',macro_text))
    for row in read('paper/v8/numeric_bindings.json'):
        obj=read(row['artifact'])
        for key in row['path']:obj=obj[key]
        expected=f"{obj:.{row['decimals']}f}"
        check(macros[row['macro']]==expected,f"macro {row['macro']} matches {row['artifact']}")
    fit=read('results/issue111_qwen3_serial_sweep.json')['cost_law']
    for name,key in [('arBase','intercept_ms'),('arSlope','slope_ms_per_token'),('dgBase','dg_base_ms'),('dgStep','dg_step_ms')]:
        check(close(float(macros[name]),fit[key]),f'{name} matches saved cost-model constant')
    crossover=(float(macros['dgBase'])-float(macros['arBase']))/float(macros['arSlope'])
    check(round(crossover)==int(macros['crossoverL']),'conditional crossover rounds to 48 tokens')
    stats=read('results/anchors/ar30b_stats.json')
    seeds=[lats(f'results/anchors/ar30b_L256_s{s}_preds.jsonl') for s in [42,137,314]]
    ar=sum(seeds,[]);dg=lats('results/anchors/dg_L256_r1_preds.jsonl')
    for key,values in [('ar30b_bgl',ar),('dg_bgl',dg)]:
        d=stats[key];check(len(values)==d['n_events'],f'{key} event count')
        for field,p in [('p50_ms',50),('p99_ms',99),('p99_9_ms',99.9)]:
            check(close(percentile(values,p),d[field]),f'{key} {field} recomputed from saved events')
        check(close(max(values),d['max_ms']),f'{key} maximum')
        check(close(statistics.stdev(values)/statistics.mean(values)*100,d['cv_pct'],.011),f'{key} CV')
        check(close(max(values)/statistics.median(values),d['max_median_ratio'],.011),f'{key} max/median')
    seed_medians=[statistics.median(x) for x in seeds]
    check(close(statistics.stdev(seed_medians)/statistics.mean(seed_medians)*100,stats['ar30b_bgl']['seed_p50_cv_pct'],.011),'seed-median CV recomputed')
    check(round(statistics.median(ar)/statistics.median(dg),1)==2.4,'primary median ratio rounds to 2.4')
    check(stats['dg_bgl']['p99_ms']<stats['ar30b_bgl']['p99_ms'],'DG absolute p99 is lower than AR-30B')
    test=stats['permutation_test']
    check(test['n_permutations']==100000 and test['p_value']==0,'saved permutation summary records zero exceedances in 100,000 draws')
    # No new randomization test is claimed: v8 describes this saved result
    # using the finite-Monte-Carlo correction rather than asserting p=0.
    energy=read('results/issue111_ar30b_energy.json')
    p=energy['power']
    calculated=(p['avg_power_w']-p['idle_power_w'])*p['duration_s']/energy['n_events']
    check(close(calculated,energy['energy_per_event_j'],.02),'AR-30B idle-subtracted energy agrees with power × duration / count')
    check(close(float(macros['arEnergy']),166.5),'AR energy is corrected to 166.5 J/event')
    sweep=read('results/issue111_qwen3_serial_sweep.json')
    expected=[(32,265,32),(128,907,121),(256,945,126),(512,945,126),(1024,945,126)]
    for budget,time,tokens in expected:
        d=sweep[f'L{budget}'];check(round(d['per_event_ms'])==time and round(d['mean_output_tokens'])==tokens,f'budget-table AR row {budget} uses mean service cost and actual tokens')
    for arm,score,valid,median in [('dg_K8',.287,31.1,280),('ar30b',.720,93.9,800),('ar3b',.573,99.4,560)]:
        d=read(f'results/issue115_humaneval/{arm}.json')
        check(round(d['pass_at_1'],3)==score and d['code_validity_pct']==valid and round(d['latency_ms']['p50'])==median,f'HumanEval baseline row {arm}')
    chain=read('results/issue117_chain/dg_K8.json')
    check(chain['pass_at_1_round1']>read('results/issue115_humaneval/dg_K8.json')['pass_at_1']+.5,'HumanEval protocol discrepancy remains explicitly unresolved')
    # Follow actual manuscript inputs, excluding unused shared style assets.
    paths=[HERE/'main.tex'];seen=set();texts=[]
    while paths:
        path=paths.pop()
        if path in seen:continue
        check(path.exists(),f'TeX input exists: {path.relative_to(HERE)}');seen.add(path)
        text=re.sub(r'(?<!\\)%[^\n]*','',path.read_text());texts.append(text)
        for target in re.findall(r'\\input\{([^}]+)\}',text):
            p=HERE/target
            if not p.suffix:p=p.with_suffix('.tex')
            paths.append(p)
        for target in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):check((HERE/target).exists(),f'figure exists: {target}')
    alltext='\n'.join(texts)
    labels=re.findall(r'\\label\{([^}]+)\}',alltext)
    check(len(labels)==len(set(labels)),'no duplicate manuscript labels')
    for ref in re.findall(r'\\(?:ref|pageref)\{([^}]+)\}',alltext):check(ref in labels,f'reference resolves: {ref}')
    bib=(HERE/'assets/references.bib').read_text();keys=set(re.findall(r'@\w+\{([^,]+),',bib))
    for group in re.findall(r'\\cite\{([^}]+)\}',alltext):
        for key in group.split(','):check(key.strip() in keys,f'citation resolves: {key.strip()}')
    check('2104.14018' not in bib,'unrelated Hoefler arXiv identifier removed')
    check('No LLM was used' not in alltext,'editing disclosure updated')
    print(f'PASS: {len(CHECKS)} selected numeric and document-integrity checks.')
    print('Scope: no GPU reruns, full bibliography audit, statistical reanalysis, or protocol certification.')

if __name__=='__main__':main()
