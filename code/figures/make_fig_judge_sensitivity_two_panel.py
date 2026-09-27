# Release version of the Paper A Figure 1 script (two-panel judge sensitivity).
# Regenerates figures/d2_judge_sensitivity_two_panel.pdf from data/scores/*.json.
# Panel (a): with-role minus boundary-role (the judge-stable contrast; primary).
# Panel (b): no-role minus boundary-role (the judge-sensitive contrast).
# Layout constants match the camera-ready figure: 7.0 x 2.7 in, fonts x0.92,
# single legend (two rows) inside panel (b)'s upper right.
# Usage: python make_fig_judge_sensitivity_two_panel.py [output_dir]
import json, os, random, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]          # .../GITHUB
SCORES = ROOT / 'data' / 'scores'
META = ROOT / 'data' / 'responses' / 'r3_responses.jsonl'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results' / 'figures'

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
    ('DeepSeek (primary)', 'r3_judge_deepseek.json', 'o', '#1f77b4'),
    ('Qwen', 'r3_judge_qwen.json', 's', '#ff7f0e'),
    ('Claude-Haiku', 'r3_judge_claude.json', '^', '#7f7f7f'),
]
INV = ('PARSE_ERROR', 'API_ERROR')
HEIGHT, SCALE = 2.7, 0.92
XL, TI, YL, TK, LG = (8.5 * SCALE, 10.5 * SCALE, 10 * SCALE, 9 * SCALE, 8 * SCALE)


def boot_ci(a, b, seed=42, B=10000):
    rng = random.Random(seed)
    na, nb = len(a), len(b)
    s = []
    for _ in range(B):
        ma = sum(a[rng.randrange(na)] for _ in range(na)) / na
        mb = sum(b[rng.randrange(nb)] for _ in range(nb)) / nb
        s.append(ma - mb)
    s.sort()
    return s[int(0.025 * B)], s[int(0.975 * B)]


meta = {}
with open(META, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        meta[d['trial_id']] = (d.get('role_line'), d.get('model_id'))

d2 = {}
for _, jf, _, _ in JUDGES:
    for r in json.load(open(SCORES / jf, encoding='utf-8')):
        if r['domain'] == 'CTL' or r['BVI_condition'] in INV:
            continue
        c, m = meta.get(r['trial_id'], (None, None))
        d2.setdefault((jf, m), {}).setdefault(c, []).append(r['D2'])

fig, axes = plt.subplots(1, 2, figsize=(7.0, HEIGHT), sharey=True)
panels = [
    ('(a) with-role $-$ boundary-role', 'with-role', 'boundary-role'),
    ('(b) no-role $-$ boundary-role', 'no-role', 'boundary-role'),
]
for ax, (title, ca, cb) in zip(axes, panels):
    ax.axvspan(-0.5, 2.5, color='#f2f2f2', zorder=0)
    for j, (jname, jf, marker, color) in enumerate(JUDGES):
        xs, ys, los, his = [], [], [], []
        for i, (mid, _) in enumerate(MODELS):
            g = d2[(jf, mid)]
            lo, hi = boot_ci(g[ca], g[cb])
            xs.append(i + (j - 1) * 0.22)
            ys.append(sum(g[ca]) / len(g[ca]) - sum(g[cb]) / len(g[cb]))
            los.append(lo)
            his.append(hi)
        ax.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, los)],
                                  [h - y for y, h in zip(ys, his)]],
                    fmt=marker, color=color, ms=5, lw=1.1, capsize=2.5,
                    elinewidth=1.0, label=jname)
    ax.axhline(0, color='black', lw=0.7, zorder=1)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([lbl for _, lbl in MODELS], rotation=32, ha='right', fontsize=XL)
    ax.set_title(title, fontsize=TI)
    ax.set_ylabel('$\\Delta$D2', fontsize=YL)
    ax.tick_params(axis='y', labelsize=TK)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(-0.15, 1.05)
# two-row legend inside panel (b)'s upper right; stays clear of all whiskers
axes[1].legend(fontsize=LG, loc='upper right', ncols=2, framealpha=0.95,
               handletextpad=0.35, columnspacing=0.9, borderpad=0.3)
axes[1].set_ylabel('')
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
fig.savefig(OUT / 'd2_judge_sensitivity_two_panel.pdf')
fig.savefig(OUT / 'd2_judge_sensitivity_two_panel.png', dpi=200)
print('written', OUT)
