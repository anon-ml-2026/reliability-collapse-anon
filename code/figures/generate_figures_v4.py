"""
Generate V9 main-text figures (binary positive flag, PSI>=0.5)
fig1: reliability gradient (cross-judge binary Po x2 judge pairs + 6-model test-retest Po + human median ref)
fig2: binary positive-flag rate by domain x model (R3)
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[2]  # repository root
OUT_DIR = BASE / "paper/latex/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_ORDER = ['CTL', 'REG-S', 'REG-H', 'UNREG']
DOMAIN_COLORS = {'CTL': '#66c2a5', 'REG-S': '#fc8d62', 'REG-H': '#8da0cb', 'UNREG': '#e78ac3'}

with open(BASE / 'data/scores/r3_judge_deepseek.json', 'r', encoding='utf-8') as f:
    ds_all = json.load(f)
ds_valid = [r for r in ds_all if r.get('D1', -1) >= 0 and not r.get('parse_error', False)]
ds_map = {r['trial_id']: r for r in ds_valid}

with open(BASE / 'data/scores/r3_judge_claude.json', 'r', encoding='utf-8') as f:
    cl_all = json.load(f)
cl_map = {r['trial_id']: r for r in cl_all if r.get('D1', -1) >= 0 and not r.get('parse_error', False)}
common_ids = sorted(set(ds_map) & set(cl_map))

with open(BASE / 'data/scores/r4_judge_deepseek.json', 'r', encoding='utf-8') as f:
    r4_all = json.load(f)
r4_valid = [r for r in r4_all if r.get('D1', -1) >= 0 and not r.get('parse_error', False)]

with open(BASE / 'data/scores/r4_judge_kimi_run1.json', 'r', encoding='utf-8') as f:
    ki_all = json.load(f)
ki_map = {r['trial_id']: r for r in ki_all if r.get('D1', -1) >= 0 and not r.get('parse_error', False)}
r4_map = {r['trial_id']: r for r in r4_valid}
r4_common = sorted(set(r4_map) & set(ki_map))

MODEL_NAMES = {
    'gpt-4o': 'GPT-4o', 'gpt-4o-mini': 'GPT-4o-mini',
    'claude-opus-4-7': 'Claude-Opus', 'claude-haiku-4-5-20251001': 'Claude-Haiku',
    'deepseek-v4-pro': 'DeepSeek-V4', 'qwen3.6-plus': 'Qwen3.6-Plus',
    'glm-5.1': 'GLM-5.1', 'mimo-v2.5-pro': 'MiMo-V2.5'
}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 7.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

def flag(r):
    return 1 if r['BVI'] >= 0.5 else 0

def mkey(r):
    return '|'.join([r['probe_id'], r.get('framing', ''), r.get('role_line', ''), r['model_id']])

# ═══════════════════════════════════════════════
# Figure 1 (V5): Reliability gradient, binary flag
# ═══════════════════════════════════════════════
print("Generating V5 Figure 1 (reliability gradient, binary)...")

# Layer 1: R3 cross-judge binary Po by domain (DeepSeek vs Claude)
cj_po = []
for dom in DOMAIN_ORDER:
    ids = [t for t in common_ids if ds_map[t]['domain'] == dom]
    po = sum(1 for t in ids if flag(ds_map[t]) == flag(cl_map[t])) / len(ids) * 100
    cj_po.append(po)
print("  R3 cross-judge (DS-Claude) Po:", [f"{v:.1f}" for v in cj_po], " n=", len(common_ids))

# Layer 2: R4 cross-judge binary Po by domain (DeepSeek vs Kimi)
kj_po = []
for dom in DOMAIN_ORDER:
    ids = [t for t in r4_common if r4_map[t]['domain'] == dom]
    po = sum(1 for t in ids if flag(r4_map[t]) == flag(ki_map[t])) / len(ids) * 100
    kj_po.append(po)
print("  R4 cross-judge (DS-Kimi) Po:", [f"{v:.1f}" for v in kj_po], " n=", len(r4_common))

# Layer 3: test-retest binary Po by domain (DeepSeek judge, 6 models)
r4_lk = {mkey(r): r for r in r4_valid}
pairs = [(a, r4_lk[mkey(a)]) for a in ds_valid if mkey(a) in r4_lk]
tr_po = []
for dom in DOMAIN_ORDER:
    sub = [(a, b) for a, b in pairs if a['domain'] == dom]
    po = sum(1 for a, b in sub if flag(a) == flag(b)) / len(sub) * 100
    tr_po.append(po)
print("  test-retest (6-model) Po:", [f"{v:.1f}" for v in tr_po], " n=", len(pairs))

human_median = 61.2  # three-level inter-human median Po, five raters (A1-A5), Anchor Set (CTL exemption applied to humans, symmetric with AI judges)

fig1, ax1 = plt.subplots(figsize=(6.75, 3.4))
x_pos = np.arange(len(DOMAIN_ORDER))

ax1.plot(x_pos, cj_po, 'o-', color='#e78ac3', linewidth=2, markersize=8,
         label='Cross-Judge R3: DeepSeek–Claude (n=4,027)')
ax1.plot(x_pos, kj_po, '^-', color='#66c2a5', linewidth=2, markersize=8,
         label='Cross-Judge R4: DeepSeek–Kimi (n=3,214)')
ax1.plot(x_pos, tr_po, 's--', color='#8da0cb', linewidth=2, markersize=8,
         label='Test-Retest R3–R4, six models (n=3,148)')
ax1.axhline(y=human_median, color='#fc8d62', linewidth=1.5, linestyle=':',
            label=f'Inter-Human Median Po, Anchor Set ({human_median:.0f}%, 3-level)')

# annotate non-CTL points (CTL = 100% ceiling, annotated once)
for i in range(1, len(DOMAIN_ORDER)):
    ax1.annotate(f'{cj_po[i]:.1f}', (x_pos[i], cj_po[i]), textcoords="offset points",
                 xytext=(0, -17), ha='center', fontsize=7, color='#e78ac3', fontweight='bold')
    ax1.annotate(f'{kj_po[i]:.1f}', (x_pos[i], kj_po[i]), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=7, color='#66c2a5', fontweight='bold')
    ax1.annotate(f'{tr_po[i]:.1f}', (x_pos[i], tr_po[i]), textcoords="offset points",
                 xytext=(15, -3), ha='left', fontsize=7, color='#8da0cb', fontweight='bold')
ax1.annotate('100%', (x_pos[0], 100), textcoords="offset points", xytext=(0, 12),
             ha='center', fontsize=7, color='gray', fontweight='bold')

ax1.set_ylabel('Percent Agreement (Po)')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(['CTL\n(Control)', 'REG-S\n(Medical)', 'REG-H\n(Psychological)', 'UNREG\n(Grey Zone)'])
ax1.set_ylim(40, 108)
ax1.set_xlim(-0.35, 3.5)
ax1.legend(loc='lower left', framealpha=0.9, fontsize=7.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', alpha=0.3, linewidth=0.5)

ax1.set_title('Reliability collapses with regulatory specificity.',
              fontsize=9, loc='left', pad=8, fontweight='bold')
ax1.text(0, -0.30, 'CTL (100%) is a sanity-check ceiling with PSI=0 by construction. Both cross-judge pairs decline REG-S → REG-H → UNREG;\n'
         'the six-model test-retest layer separates REG-S from the two psychosocial-content domains (REG-H/UNREG tie, p = .55).',
         transform=ax1.transAxes, fontsize=7.5, color='gray')

fig1.savefig(OUT_DIR / 'fig1_reliability_gradient.pdf')
fig1.savefig(OUT_DIR / 'fig1_reliability_gradient.png', dpi=200)
plt.close(fig1)
print("  -> paper/latex/figures/fig1_reliability_gradient.pdf/.png")

# ═══════════════════════════════════════════════
# Figure 2 (V9, unchanged data): Binary positive-flag rate by domain × model
# ═══════════════════════════════════════════════
print("Generating Figure 2 (binary positive flag by domain × model, R3 — unchanged)...")

models_all = sorted(set(r['model_id'] for r in ds_valid))
model_flag = {}
for mid in models_all:
    model_flag[mid] = {}
    for dom in DOMAIN_ORDER:
        sub = [r for r in ds_valid if r['model_id'] == mid and r['domain'] == dom]
        model_flag[mid][dom] = sum(flag(r) for r in sub) / len(sub) * 100 if sub else 0

models_sorted = sorted(models_all, key=lambda m: model_flag[m]['UNREG'], reverse=True)
model_labels = [MODEL_NAMES.get(m, m) for m in models_sorted]

fig2, ax2 = plt.subplots(figsize=(6.75, 3.5))
x = np.arange(len(models_sorted))
width = 0.2
for i, dom in enumerate(DOMAIN_ORDER):
    vals = [model_flag[m][dom] for m in models_sorted]
    bars = ax2.bar(x + i*width, vals, width, label=dom, color=DOMAIN_COLORS[dom],
                   edgecolor='white', linewidth=0.3)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.0f}',
                     ha='center', va='bottom', fontsize=6)

ax2.set_ylabel('Positive-Flag Rate (%, PSI ≥ 0.5)')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(model_labels, rotation=15, ha='right')
ax2.legend(loc='upper right', ncol=2, framealpha=0.9)
ax2.set_ylim(0, 70)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.set_title('Positive-flag rates (PSI ≥ 0.5) by regulatory domain across eight models.',
              fontsize=9, loc='left', pad=8, fontweight='bold')
ax2.text(0, -0.20, 'The domain gradient holds unanimously across models (DeepSeek judge, R3, binary positive flag). '
         'Domain rates: CTL 0%, REG-S 5.6%, REG-H 20.8%, UNREG 39.8%.',
         transform=ax2.transAxes, fontsize=7.5, color='gray')

fig2.savefig(OUT_DIR / 'fig2_flag_by_domain_model.pdf')
fig2.savefig(OUT_DIR / 'fig2_flag_by_domain_model.png', dpi=200)
plt.close(fig2)
print("  -> paper/latex/figures/fig2_flag_by_domain_model.pdf/.png")

print("\nDone! 2 V9 figures saved.")
