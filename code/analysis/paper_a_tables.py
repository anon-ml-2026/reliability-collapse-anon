# -*- coding: utf-8 -*-
"""Paper A -- re-derives the main tables and bootstrap CIs.

Covers: Table 1 (D1 distribution), Table 2 (D2 by model x condition with
bootstrap 95% CIs, both contrasts, primary judge), Table 4 (D2 by domain and
response range), Table 5 (D3), and the appendix contrast table (role->bnd
drop with CIs under all three R3 judges). Bootstrap procedure matches the
published figure script: python random.Random(42), 10,000 resamples,
percentiles s[250] / s[9749].

Usage:  python paper_a_tables.py   (writes results/paper_a_tables.json)
"""
import json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HIGHER = {'deepseek-v4-pro', 'qwen3.6-plus', 'glm-5.1'}
JUDGES = {'deepseek': 'r3_judge_deepseek.json', 'qwen': 'r3_judge_qwen.json',
          'claude': 'r3_judge_claude.json'}


def load_valid(path):
    rows = json.load(open(path, encoding='utf-8'))
    return [r for r in rows
            if isinstance(r.get('D2'), int) and 0 <= r['D2'] <= 2
            and isinstance(r.get('D1'), int) and isinstance(r.get('D3'), int)
            and r.get('domain') != 'CTL']


def boot_ci(a, b, seed=42, B=10000):
    rng = random.Random(seed)
    na, nb = len(a), len(b)
    s = []
    for _ in range(B):
        ma = sum(a[rng.randrange(na)] for _ in range(na)) / na
        mb = sum(b[rng.randrange(nb)] for _ in range(nb)) / nb
        s.append(ma - mb)
    s.sort()
    return s[int(0.025 * B)], s[int(0.975 * B) - 1]


ds = load_valid(ROOT / 'data' / 'scores' / 'r3_judge_deepseek.json')
out = {}

# --- Table 1: D1 distribution by condition ---
t1 = {}
for cond in ('no-role', 'with-role', 'boundary-role'):
    v = [r['D1'] for r in ds if r['role_line'] == cond]
    t1[cond] = {'n': len(v),
                'D1=0': round(sum(1 for x in v if x == 0) / len(v), 4),
                'D1=1': round(sum(1 for x in v if x == 1) / len(v), 4),
                'D1=2': round(sum(1 for x in v if x == 2) / len(v), 4),
                'mean': round(sum(v) / len(v), 3)}
out['table1_d1_distribution'] = t1

# --- Table 2: D2 by model, both contrasts, CIs (primary judge) ---
by_model = defaultdict(lambda: defaultdict(list))
for r in ds:
    by_model[r['model_id']][r['role_line']].append(r['D2'])
t2 = {}
for m in sorted(by_model):
    g = by_model[m]
    avg = {c: sum(v) / len(v) for c, v in g.items()}
    d_nb = avg['no-role'] - avg['boundary-role']
    d_rb = avg['with-role'] - avg['boundary-role']
    lo1, hi1 = boot_ci(g['no-role'], g['boundary-role'])
    lo2, hi2 = boot_ci(g['with-role'], g['boundary-role'])
    t2[m] = {'means': {c: round(v, 3) for c, v in avg.items()},
             'delta_no_bnd': round(d_nb, 3), 'ci_no_bnd': [round(lo1, 2), round(hi1, 2)],
             'delta_role_bnd': round(d_rb, 3), 'ci_role_bnd': [round(lo2, 2), round(hi2, 2)],
             'higher_response_group': m in HIGHER}
out['table2_d2_by_model'] = t2

# --- Table 4: D2 by domain x group x condition ---
by_cell = defaultdict(lambda: defaultdict(list))
for r in ds:
    grp = 'higher' if r['model_id'] in HIGHER else 'lower'
    by_cell[(r['domain'], grp)][r['role_line']].append(r['D2'])
t4 = {}
for (dom, grp), g in sorted(by_cell.items()):
    means = {c: round(sum(v) / len(v), 3) for c, v in g.items()}
    t4[f'{dom}_{grp}'] = {'means': means,
                          'swing': round(max(means.values()) - min(means.values()), 3)}
out['table4_d2_by_domain'] = t4

# --- Table 5: D3 by model ---
t5 = {}
for m in sorted(by_model):
    g = defaultdict(list)
    for r in ds:
        if r['model_id'] == m:
            g[r['role_line']].append(r['D3'])
    t5[m] = {'no_role': round(sum(g['no-role']) / len(g['no-role']), 3),
             'boundary': round(sum(g['boundary-role']) / len(g['boundary-role']), 3),
             'delta': round(sum(g['no-role']) / len(g['no-role'])
                            - sum(g['boundary-role']) / len(g['boundary-role']), 3)}
out['table5_d3_by_model'] = t5

# --- Appendix contrast table: role->bnd drop with CIs, three judges ---
t7 = {}
for tag, fname in JUDGES.items():
    rows = load_valid(ROOT / 'data' / 'scores' / fname)
    jm = defaultdict(lambda: defaultdict(list))
    for r in rows:
        jm[r['model_id']][r['role_line']].append(r['D2'])
    t7[tag] = {}
    for m in sorted(jm):
        g = jm[m]
        if 'with-role' not in g or 'boundary-role' not in g:
            continue
        d = sum(g['with-role']) / len(g['with-role']) - sum(g['boundary-role']) / len(g['boundary-role'])
        lo, hi = boot_ci(g['with-role'], g['boundary-role'])
        t7[tag][m] = {'drop': round(d, 3), 'ci': [round(lo, 2), round(hi, 2)]}
out['appendix_role_to_bnd_three_judges'] = t7

dest = ROOT / 'results' / 'paper_a_tables.json'
dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', dest)
print('spot: GLM ci_no_bnd =', t2['glm-5.1']['ci_no_bnd'], '| qwen judge glm drop ci =',
      t7['qwen']['glm-5.1']['ci'])
