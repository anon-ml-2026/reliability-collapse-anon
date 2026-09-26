# -*- coding: utf-8 -*-
"""Paper A, Appendix B -- model-level significance tests (Table `tab:app-tests`).

Exact two-sided Wilcoxon signed-rank (enumerated over the 2^n sign patterns,
zeros dropped) and paired t-tests across the eight models, for D1/D2/D3 and
both condition contrasts, on primary-judge (DeepSeek) per-model means.

Usage:  python model_level_tests.py   (writes results/model_level_tests.json)
"""
import json, itertools
from collections import defaultdict
from pathlib import Path
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[2]
scores = json.load(open(ROOT / 'data' / 'scores' / 'r3_judge_deepseek.json', encoding='utf-8'))

tables = {}
for dim in ('D1', 'D2', 'D3'):
    tab = defaultdict(lambda: defaultdict(list))
    for r in scores:
        if r.get('BVI_condition') in ('PARSE_ERROR', 'API_ERROR') or r.get('domain') == 'CTL':
            continue
        tab[r['model_id']][r['role_line']].append(r[dim])
    tables[dim] = tab
models = sorted(tables['D2'])
avg = lambda L: sum(L) / len(L)


def exact_wilcoxon(diffs):
    d = [x for x in diffs if x != 0]
    n = len(d)
    ranks = st.rankdata([abs(x) for x in d])
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    hit = 0
    for signs in itertools.product([0, 1], repeat=n):
        w = sum(r for r, s in zip(ranks, signs) if s)
        if w >= w_plus or w <= sum(ranks) - w_plus:
            hit += 1
    return hit / 2 ** n, n


out = {}
for dim in ('D1', 'D2', 'D3'):
    for a, b, tag in (('no-role', 'boundary-role', 'no_minus_bnd'),
                      ('with-role', 'boundary-role', 'role_minus_bnd')):
        diffs = [avg(tables[dim][m][a]) - avg(tables[dim][m][b]) for m in models]
        p_w, n_eff = exact_wilcoxon(diffs)
        t_stat, p_t = st.ttest_rel([avg(tables[dim][m][a]) for m in models],
                                   [avg(tables[dim][m][b]) for m in models])
        out[f'{dim}_{tag}'] = {
            'mean_diff': round(avg(diffs), 4),
            'wilcoxon_exact_p': round(p_w, 4), 'wilcoxon_n_eff': n_eff,
            'paired_t': round(float(t_stat), 3), 'paired_t_p': round(float(p_t), 4)}

dest = ROOT / 'results' / 'model_level_tests.json'
dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', dest)
for k, v in out.items():
    print(k, v)
