# -*- coding: utf-8 -*-
"""Reproduces the headline table of the post-hoc control pilot from scores/."""
import json, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ['no-role', 'with-role', 'boundary-role', 'neutral-clause', 'role-boundary']
LABELS = {'deepseek': 'DeepSeek-V4', 'qwen': 'Qwen3.6-Plus',
          'glm': 'GLM-5.1', 'mimo': 'MiMo-V2.5-Pro'}

for tag in ('deepseek', 'qwen', 'glm', 'mimo'):
    recs = json.load(open(os.path.join(HERE, 'scores', f'{tag}.json'), encoding='utf-8'))
    per = defaultdict(list)
    bad = 0
    for r in recs:
        cond = r.get('role_line') or r['trial_id'].split('|')[2]
        if r.get('BVI_condition') in ('PARSE_ERROR', 'API_ERROR'):
            bad += 1
            continue
        per[cond].append(r['D2'])
    row = ' '.join(f'{sum(per[c]) / len(per[c]):.2f}(n={len(per[c])})' if per.get(c) else '--'
                   for c in ORDER)
    print(f'{LABELS[tag]:14s} {row}  parse/api errors: {bad}')
