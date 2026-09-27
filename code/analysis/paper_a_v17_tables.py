# Re-derives the tables added in the V17 revision of Paper A, from the frozen
# release data. Writes JSON to results/paper_a_v17_tables.json.
#
#   1. Judge-side exclusions by model x identity condition (Appendix B,
#      Table "Judge-side exclusions"; 113/3240 = 3.5% non-CTL).
#   2. REG-only (REG-H + REG-S) per-model D2 contrasts with bootstrap 95% CIs
#      (Appendix B, REG-only contrasts table; 2,086 valid trials).
#   3. Three-judge role-referenced and no-role-referenced contrasts used by
#      Figure 1 (cross-check against Table "D2 drop ... under three judges").
#
# All numbers were verified against the manuscript on 2026-09-27.
# Usage: python paper_a_v17_tables.py
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../GITHUB
SCORES = ROOT / 'data' / 'scores'
META = ROOT / 'data' / 'responses' / 'r3_responses.jsonl'
OUT = ROOT / 'results' / 'paper_a_v17_tables.json'

INV = ('PARSE_ERROR', 'API_ERROR')
CONDS = ['no-role', 'with-role', 'boundary-role']
MODELS = ['Claude-Haiku', 'Claude-Opus', 'DeepSeek-V4', 'GLM-5.1',
          'GPT-4o', 'GPT-4o-mini', 'MiMo-V2.5-Pro', 'Qwen3.6-Plus']

meta = {}
with open(META, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        meta[d['trial_id']] = (d.get('role_line'), d.get('model_label'))

ds = json.load(open(SCORES / 'r3_judge_deepseek.json', encoding='utf-8'))

# 1. exclusions (invalid/total per model x condition, non-CTL)
tab = defaultdict(lambda: [0, 0])
for r in ds:
    if r['domain'] == 'CTL':
        continue
    c, m = meta.get(r['trial_id'], (r.get('role_line'), r.get('model_label')))
    tab[(c, m)][0] += 1
    if r['BVI_condition'] in INV:
        tab[(c, m)][1] += 1
exclusions = {
    'per_model': {m: {c: f"{tab[(c, m)][1]}/{tab[(c, m)][0]}" for c in CONDS}
                  for m in MODELS},
    'condition_totals': {c: f"{sum(tab[(c, m)][1] for m in MODELS)}/"
                            f"{sum(tab[(c, m)][0] for m in MODELS)}"
                         for c in CONDS},
}


def boot_ci(a, b, seed=42, B=10000):
    rng = random.Random(seed)
    na, nb = len(a), len(b)
    s = []
    for _ in range(B):
        ma = sum(a[rng.randrange(na)] for _ in range(na)) / na
        mb = sum(b[rng.randrange(nb)] for _ in range(nb)) / nb
        s.append(ma - mb)
    s.sort()
    return round(s[int(0.025 * B)], 3), round(s[int(0.975 * B)], 3)


# 2. REG-only D2 contrasts (DeepSeek judge)
reg = [r for r in ds if r['domain'] in ('REG-H', 'REG-S')
       and r['BVI_condition'] not in INV
       and meta.get(r['trial_id'], (None,))[0] in CONDS]
reg_only = {}
for m in MODELS:
    g = {c: [r['D2'] for r in reg
             if meta[r['trial_id']][1] == m and meta[r['trial_id']][0] == c]
         for c in CONDS}
    means = {c: round(sum(g[c]) / len(g[c]), 3) for c in CONDS}
    reg_only[m] = {
        'means': means, 'n': sum(len(g[c]) for c in CONDS),
        'd_no_bnd': round(means['no-role'] - means['boundary-role'], 3),
        'ci_no_bnd': boot_ci(g['no-role'], g['boundary-role']),
        'd_role_bnd': round(means['with-role'] - means['boundary-role'], 3),
        'ci_role_bnd': boot_ci(g['with-role'], g['boundary-role']),
    }
pooled = {c: round(sum(r['D2'] for r in reg if meta[r['trial_id']][0] == c)
                   / sum(1 for r in reg if meta[r['trial_id']][0] == c), 3)
          for c in CONDS}

# 3. three-judge contrasts (Figure 1 values)
judges = [('DeepSeek', 'r3_judge_deepseek.json'),
          ('Qwen', 'r3_judge_qwen.json'),
          ('Claude-Haiku', 'r3_judge_claude.json')]
three_judge = {}
for jname, jf in judges:
    recs = json.load(open(SCORES / jf, encoding='utf-8'))
    for m in MODELS:
        g = {c: [r['D2'] for r in recs
                 if r['domain'] != 'CTL' and r['BVI_condition'] not in INV
                 and meta.get(r['trial_id'], (None, None))[0] == c
                 and meta.get(r['trial_id'], (None, None))[1] == m]
             for c in CONDS}
        three_judge[f'{jname}|{m}'] = {
            'd_no_bnd': round(sum(g['no-role']) / len(g['no-role'])
                              - sum(g['boundary-role']) / len(g['boundary-role']), 3),
            'd_role_bnd': round(sum(g['with-role']) / len(g['with-role'])
                                - sum(g['boundary-role']) / len(g['boundary-role']), 3),
        }

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({'exclusions': exclusions, 'reg_only': reg_only,
                           'reg_only_pooled': pooled, 'three_judge': three_judge},
                          ensure_ascii=False, indent=1), encoding='utf-8')
print('written', OUT)
