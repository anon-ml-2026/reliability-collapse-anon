"""
Generate 3 appendix figures for the NAACL paper (sanitized PSI terminology)
Following naacl-figure skill guidelines: PDF vector, colorblind-friendly, >=8pt fonts
"""
import json, statistics
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE = Path(__file__).resolve().parents[2]  # repository root
OUT_DIR = BASE / "paper/latex/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Color palette: ColorBrewer Set2 (colorblind-friendly) ──
C = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
     '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
DOMAIN_COLORS = {'CTL': '#66c2a5', 'REG-S': '#fc8d62', 'REG-H': '#8da0cb', 'UNREG': '#e78ac3'}
DOMAIN_ORDER = ['CTL', 'REG-S', 'REG-H', 'UNREG']

# ── Load data ──
with open(BASE / 'data/scores/r3_judge_deepseek.json', 'r', encoding='utf-8') as f:
    ds_all = json.load(f)
ds_valid = [r for r in ds_all if r.get('D1', -1) >= 0]

with open(BASE / 'results/cross_judge_deepseek_claude.json', 'r', encoding='utf-8') as f:
    cj = json.load(f)

with open(BASE / 'results/human_anchor_analysis.json', 'r', encoding='utf-8') as f:
    hv = json.load(f)

# Model display names
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

models_all = sorted(set(r['model_id'] for r in ds_valid))

# ═══════════════════════════════════════════════
# Figure A: Three-Level Reliability Gradient
# ═══════════════════════════════════════════════
print("Generating three-level gradient figure...")
fig2, ax2 = plt.subplots(figsize=(6.75, 3.2))

# Line 1: AI-AI cross-judge Po by domain
cj_domains = cj['domains']
ai_ai_po = [cj_domains[d]['agree_pct'] for d in DOMAIN_ORDER]

# Line 2: Test-retest Po by domain (three-level, six models)
tr_po = [100.0, 93.4, 77.1, 72.3]

# Line 3: Inter-human median Po (five raters A1-A5; CTL exemption applied to humans)
human_median = 61.2

x_pos = np.arange(len(DOMAIN_ORDER))

ax2.plot(x_pos, ai_ai_po, 'o-', color='#e78ac3', linewidth=2, markersize=8, label='AI-AI Cross-Judge (DS vs. Claude)')
ax2.plot(x_pos, tr_po, 's--', color='#8da0cb', linewidth=2, markersize=8, label='Test-Retest (R3 vs. R4, same judge)')
ax2.axhline(y=human_median, color='#fc8d62', linewidth=1.5, linestyle=':', label=f'Inter-Human Median Po ({human_median:.0f}%)')

for i, (v1, v2) in enumerate(zip(ai_ai_po, tr_po)):
    ax2.annotate(f'{v1:.1f}%', (x_pos[i], v1), textcoords="offset points", xytext=(0, -16),
                ha='center', fontsize=7, color='#e78ac3', fontweight='bold')
    ax2.annotate(f'{v2:.1f}%', (x_pos[i], v2), textcoords="offset points", xytext=(0, 11),
                ha='center', fontsize=7, color='#8da0cb', fontweight='bold')

ax2.set_ylabel('Percent Agreement (Po)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['CTL\n(Control)', 'REG-S\n(Medical)', 'REG-H\n(Psychological)', 'UNREG\n(Grey Zone)'])
ax2.set_ylim(40, 108)
ax2.legend(loc='lower left', framealpha=0.9, fontsize=7.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.3, linewidth=0.5)

ax2.set_title('Three-level reliability gradient — agreement rates across three reliability layers.',
              fontsize=9, loc='left', pad=8, fontweight='bold')
ax2.text(0, -0.18, 'Both AI-judge layers decline from CTL to UNREG; cross-judge drops steepest (100%→51.1%). '
         'The inter-human median (61.2%, five raters, Anchor Set) lies between the two AI layers in UNREG.',
         transform=ax2.transAxes, fontsize=7.5, color='gray')

fig2.savefig(OUT_DIR / 'fig2_threelevel_gradient.pdf')
fig2.savefig(OUT_DIR / 'fig2_threelevel_gradient.png', dpi=200)
plt.close(fig2)
print("  -> paper/latex/figures/fig2_threelevel_gradient.pdf/.png")

# ═══════════════════════════════════════════════
# Figure B: UNREG Confusion Matrices (side-by-side)
# ═══════════════════════════════════════════════
print("Generating confusion matrices figure...")
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(6.75, 3.2))

# Panel A: AI-AI cross-judge confusion in UNREG
ds_map = {r['trial_id']: r for r in ds_valid}
with open(BASE / 'data/scores/r3_judge_claude.json', 'r', encoding='utf-8') as f:
    cl_all = json.load(f)
cl_map = {r['trial_id']: r for r in cl_all if r.get('D1', -1) >= 0 and not r.get('parse_error', False)}
common_ids = sorted(set(ds_map) & set(cl_map))
cm_ai = np.zeros((3, 3))
bvi_idx = {0: 0, 0.5: 1, 1: 2}
for tid in common_ids:
    if ds_map[tid]['domain'] == 'UNREG':
        a = ds_map[tid]['BVI']
        b = cl_map[tid]['BVI']
        cm_ai[bvi_idx[a], bvi_idx[b]] += 1
cm_ai_norm = cm_ai / cm_ai.sum()
n_unreg = int(cm_ai.sum())
print(f"  UNREG cross-judge n={n_unreg}")
labels = ['PSI=0', 'PSI=0.5\n(interm.)', 'PSI=1']

im_a = ax3a.imshow(cm_ai_norm, cmap='YlOrRd', vmin=0, vmax=0.7)
ax3a.set_title(f'(a) AI-AI Cross-Judge (n={n_unreg})', fontsize=8.5, fontweight='bold')
ax3a.set_xticks([0, 1, 2]); ax3a.set_yticks([0, 1, 2])
ax3a.set_xticklabels(labels, fontsize=7); ax3a.set_yticklabels(labels, fontsize=7)
ax3a.set_xlabel('Claude-Haiku', fontsize=8)
ax3a.set_ylabel('DeepSeek-V4', fontsize=8)

for i in range(3):
    for j in range(3):
        color = 'white' if cm_ai_norm[i, j] > 0.35 else 'black'
        ax3a.text(j, i, f'{int(cm_ai[i,j])}\n({cm_ai_norm[i,j]*100:.1f}%)',
                 ha='center', va='center', fontsize=6.5, color=color, fontweight='bold')

# Panel B: R3 vs Human Majority (computed from human_vs_r3 data)
maj_details = hv.get("majority_details", [])
cm_hm = np.zeros((3, 3))
for m in maj_details:
    r3_bvi = m['r3_bvi']
    maj_bvi = m['majority_bvi']
    cm_hm[bvi_idx[r3_bvi], bvi_idx[maj_bvi]] += 1

n_human = int(cm_hm.sum())
po_human = sum(cm_hm[i,i] for i in range(3)) / n_human * 100 if n_human > 0 else 0
cm_hm_norm = cm_hm / cm_hm.sum() if cm_hm.sum() > 0 else cm_hm

im_b = ax3b.imshow(cm_hm_norm, cmap='YlOrRd', vmin=0, vmax=0.7)
ax3b.set_title(f'(b) R3 vs. Human Majority (n={n_human})', fontsize=8.5, fontweight='bold')
ax3b.set_xticks([0, 1, 2]); ax3b.set_yticks([0, 1, 2])
ax3b.set_xticklabels(labels, fontsize=7); ax3b.set_yticklabels([])
ax3b.set_xlabel('Human Majority', fontsize=8)

for i in range(3):
    for j in range(3):
        if cm_hm[i, j] > 0:
            color = 'white' if cm_hm_norm[i, j] > 0.35 else 'black'
            ax3b.text(j, i, f'{int(cm_hm[i,j])}\n({cm_hm_norm[i,j]*100:.1f}%)',
                     ha='center', va='center', fontsize=6.5, color=color, fontweight='bold')

plt.colorbar(im_b, ax=[ax3a, ax3b], shrink=0.8, label='Proportion')
fig3.suptitle('Confusion matrices for three-level PSI scores.',
              fontsize=9, fontweight='bold', x=0, ha='left')

fig3.savefig(OUT_DIR / 'fig3_confusion_matrices.pdf')
fig3.savefig(OUT_DIR / 'fig3_confusion_matrices.png', dpi=200)
plt.close(fig3)
print("  -> paper/latex/figures/fig3_confusion_matrices.pdf/.png")

# ═══════════════════════════════════════════════
# Figure C: Marker-Code Profiles Heatmap
# ═══════════════════════════════════════════════
print("Generating marker-code profiles figure...")
fig4, ax4 = plt.subplots(figsize=(6.75, 3.5))

# Use Claude judge marker-code data to compute PSI=1 rate by model × marker code
with open(BASE / 'data/scores/r3_judge_claude.json', 'r', encoding='utf-8') as f:
    cl_all = json.load(f)
cl_valid = [r for r in cl_all if r.get('D1', -1) >= 0 and not r.get('parse_error', False)]

bm_codes = ['B1', 'B2', 'B3', 'B4', 'M1', 'M2', 'M3', 'M4']
bm_labels = ['B1\nPsy Dx', 'B2\nPsy Rx', 'B3\nPsy Eff', 'B4\nRole Sub',
             'M1\nMed Dx', 'M2\nMed Rx', 'M3\nMed Eff', 'M4\nNo Ref']

model_flag_unreg = {}
for mid in models_all:
    sub = [r for r in ds_valid if r['model_id'] == mid and r['domain'] == 'UNREG']
    model_flag_unreg[mid] = sum(1 for r in sub if r['BVI'] == 1) / len(sub) * 100 if sub else 0
models_sorted = sorted(models_all, key=lambda m: model_flag_unreg[m], reverse=True)

heatmap_data = np.zeros((len(models_sorted), len(bm_codes)))
for i, mid in enumerate(models_sorted):
    for j, bm in enumerate(bm_codes):
        sub = [r for r in cl_valid if r['model_id'] == mid and r.get('BM', {}).get(bm) == 'YES']
        if sub:
            heatmap_data[i, j] = sum(1 for r in sub if r['BVI'] == 1) / len(sub) * 100

# Cap at 60 for color scaling
im4 = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=60)
ax4.set_xticks(range(len(bm_codes)))
ax4.set_yticks(range(len(models_sorted)))
ax4.set_xticklabels(bm_labels, fontsize=7)
ax4.set_yticklabels([MODEL_NAMES.get(m, m) for m in models_sorted], fontsize=7.5)

for i in range(len(models_sorted)):
    for j in range(len(bm_codes)):
        val = heatmap_data[i, j]
        if val > 0:
            color = 'white' if val > 30 else 'black'
            ax4.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6.5, color=color, fontweight='bold')

plt.colorbar(im4, ax=ax4, shrink=0.85, label='PSI=1 Rate (%)')
ax4.set_title('Marker-code profiles: PSI=1 rate by model × marker code (Claude-Haiku judge).',
              fontsize=9, loc='left', pad=8, fontweight='bold')
ax4.text(0, -0.15, 'GPT-4o is M-dominant (higher medical-marker rates); the other models are B-dominant to varying degrees. '
         'White cells indicate marker codes rarely triggered by that model.',
         transform=ax4.transAxes, fontsize=7.5, color='gray')

fig4.savefig(OUT_DIR / 'fig4_marker_profiles.pdf')
fig4.savefig(OUT_DIR / 'fig4_marker_profiles.png', dpi=200)
plt.close(fig4)
print("  -> paper/latex/figures/fig4_marker_profiles.pdf/.png")

print(f"\nDone! 3 appendix figures saved to {OUT_DIR}/")
