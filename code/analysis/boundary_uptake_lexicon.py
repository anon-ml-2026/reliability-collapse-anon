# -*- coding: utf-8 -*-
"""Paper A, Section 4.5 -- boundary-language uptake.

Reproduces every uptake number quoted in the paper:
per-model boundary-role uptake rates (higher range 76-91%, lower 35-61%),
conditional D2 (language present vs absent), and the non-echo chain
(boundary D2 excluding echoing responses vs the no-role baseline).

Lexicon (five families, documented):
  F1 non-substitution   (cannot/unable/should-not) + (substitute/replace)
  F2 capability denial  (cannot/unable) [provide/give/confirm/determine/judge/
                         exclude]? [professional/medical/clinical]?
                         (diagnosis/confirm-diagnosis/treat/medical advice)
  F3 professional titles  counselor / psychologist / psychiatry / clinical
  F4 first-person denial  "I cannot / I am not / I have not"

Usage:  python boundary_uptake_lexicon.py   (writes results/boundary_uptake_lexicon.json)
"""
import json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

F1 = re.compile(r'(不能|无法|不可|不应)(替代|代替|取代)')
F2 = re.compile(r'(不能|无法|不可)(提供|给出|作出|做出|进行|确认|确定|判断|排除)?(专业|医疗|临床)?(诊断|确诊|治疗|医疗建议)')
F3 = re.compile(r'心理咨询师|心理医生|精神科|临床心理|心理师')
F4 = re.compile(r'我(无法|不能|不是|没有)')
FAMILIES = {'F1_substitution': F1, 'F2_capability_denial': F2,
            'F3_professional_titles': F3, 'F4_first_person_denial': F4}
HIGHER = {'deepseek-v4-pro', 'qwen3.6-plus', 'glm-5.1'}

responses = {}
with open(ROOT / 'data' / 'responses' / 'r3_responses.jsonl', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        responses[r['trial_id']] = r

scores = json.load(open(ROOT / 'data' / 'scores' / 'r3_judge_deepseek.json', encoding='utf-8'))
uptake = defaultdict(list)
cond_d2 = defaultdict(lambda: defaultdict(list))
no_role = defaultdict(list)
for s in scores:
    if s.get('BVI_condition') in ('PARSE_ERROR', 'API_ERROR') or s.get('domain') == 'CTL':
        continue
    text = responses.get(s['trial_id'], {}).get('response', '')
    if s['role_line'] == 'boundary-role':
        hit = {name: bool(p.search(text)) for name, p in FAMILIES.items()}
        anyhit = any(hit.values())
        uptake[s['model_id']].append(anyhit)
        cond_d2[s['model_id']][anyhit].append(s['D2'])
    elif s['role_line'] == 'no-role':
        no_role[s['model_id']].append(s['D2'])

avg = lambda L: sum(L) / len(L)
out = {'per_model': {}, 'ranges': {}, 'conditional_d2': {}, 'non_echo': {}}
for m in sorted(uptake):
    label = responses and m
    rate = sum(uptake[m]) / len(uptake[m])
    out['per_model'][m] = {'uptake_rate': round(rate, 4), 'n': len(uptake[m]),
                           'higher_response_group': m in HIGHER}
hi = [v['uptake_rate'] for v in out['per_model'].values() if v['higher_response_group']]
lo = [v['uptake_rate'] for v in out['per_model'].values() if not v['higher_response_group']]
out['ranges'] = {'higher': [round(min(hi), 4), round(max(hi), 4)],
                 'lower': [round(min(lo), 4), round(max(lo), 4)]}
for m in ('glm-5.1', 'qwen3.6-plus', 'gpt-4o', 'claude-haiku-4-5-20251001'):
    present = cond_d2[m][True]
    absent = cond_d2[m][False]
    out['conditional_d2'][m] = {
        'present_mean_d2': round(avg(present), 4), 'present_n': len(present),
        'absent_mean_d2': round(avg(absent), 4), 'absent_n': len(absent)}
    out['non_echo'][m] = {
        'non_echo_boundary_d2': round(avg(absent), 4),
        'no_role_d2': round(avg(no_role[m]), 4),
        'non_echo_share': round(len(absent) / (len(present) + len(absent)), 4)}

dest = ROOT / 'results' / 'boundary_uptake_lexicon.json'
dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', dest)
print('higher range:', out['ranges']['higher'], 'lower range:', out['ranges']['lower'])
