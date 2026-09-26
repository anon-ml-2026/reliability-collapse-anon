# Remake of the Paper A judge-sensitivity forest figure from the frozen release.
# Regenerates d2_judge_sensitivity.pdf/png from GITHUB/data/scores/*.json.
# Contrast: delta D2 = no-role minus boundary-role (positive = boundary lowers D2),
# matching the sign convention of paperA_revised.tex Tables 2/4.
# Usage: python make_fig_judge_sensitivity.py [output_dir]
import json, os, random, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'scores')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "paper", "paperA", "figures")

MODELS = [
    ('deepseek-v4-pro', 'DeepSeek-V4'),
    ('qwen3.6-plus', 'Qwen3.6-Plus'),
    ('glm-5.1', 'GLM-5.1'),
    ('claude-haiku-4-5-20251001', 'Claude-Haiku'),
    ('mimo-v2.5-pro', 'MiMo-V2.5-Pro'),
    ('gpt-4o', 'GPT-4o'),
    ('gpt-4o-mini', 'GPT-4o-mini'),
    ('claude-opus-4-7', 'Claude-Opus'),
]
JUDGES = [
    ('DeepSeek-V4 (primary)', 'r3_judge_deepseek.json', '#1f77b4'),
    ('Qwen3.6-Plus', 'r3_judge_qwen.json', '#ff7f0e'),
    ('Claude-Haiku', 'r3_judge_claude.json', '#7f7f7f'),
]


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


def series(path):
    recs = json.load(open(path, encoding='utf-8'))
    pts = {}
    for mid, ml in MODELS:
        g = {r['role_line']: [] for r in recs if r.get('model_id') == mid
             and r.get('domain') != 'CTL' and isinstance(r.get('D2'), int) and 0 <= r['D2'] <= 2}
        for r in recs:
            if r.get('model_id') == mid and r.get('domain') != 'CTL' and isinstance(r.get('D2'), int) and 0 <= r['D2'] <= 2:
                g[r['role_line']].append(r['D2'])
        if g.get('no-role') and g.get('boundary-role'):
            pts[ml] = (sum(g['no-role']) / len(g['no-role'])
                       - sum(g['boundary-role']) / len(g['boundary-role']),
                       *boot_ci(g['no-role'], g['boundary-role']))
    return pts


fig, ax = plt.subplots(figsize=(6.4, 3.6))
y_pos, y_labels = [], []
for i, (mid, ml) in enumerate(reversed(MODELS)):
    base = i
    for j, (jname, fname, color) in enumerate(JUDGES):
        pt, lo, hi = series(os.path.join(SCORES, fname))[ml]
        y = base + (j - 1) * 0.22
        ax.errorbar([pt], [y], xerr=[[pt - lo], [hi - pt]], fmt='o', color=color, capsize=3,
                    markersize=4.5, linewidth=1.4, label=jname if i == len(MODELS) - 1 else None)
    y_pos.append(base)
    y_labels.append(ml)

ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel(r'$\Delta$ D2 (no-role $-$ boundary-role)')
ax.legend(loc='lower right', frameon=False, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'd2_judge_sensitivity.pdf'))
fig.savefig(os.path.join(OUT, 'd2_judge_sensitivity.png'), dpi=150)
print('written to', OUT)
