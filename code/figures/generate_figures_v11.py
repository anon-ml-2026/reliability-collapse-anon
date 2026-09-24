"""
V11 新图（风格同 _generate_figures_v4.py）
fig5_multi_config: 8 个裁判配置 × 3 域二值 Po（正文 §4.1.1）
fig_ablation: UNREG 锚定 ablation + REG-S 安慰剂（附录 M）
数据来源: results/test_retest_qwen_and_r4_cross.json,
         results/cross_judge_qwen_claude.json, results/ablation_unreg_anchor.json,
         results/kimi_t1_noise_ceiling.json（数值已核对，直接硬编码避免重算分歧）
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[2]  # repository root
OUT_DIR = BASE / "paper/latex/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_COLORS = {'REG-S': '#fc8d62', 'REG-H': '#8da0cb', 'UNREG': '#e78ac3'}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ═══════════════════════════════════════════════
# fig5_multi_config: 8 configurations x 3 domains
# ═══════════════════════════════════════════════
print("fig5_multi_config ...")

CONFIGS = [
    # (label, REG-S, REG-H, UNREG, kind)  kind: R3 cross / R4 cross / TR
    ("DS$\\times$CL\n(R3)",        88.2,  76.5,  63.8,  'r3'),
    ("QW$\\times$CL\n(R3)",        86.84, 83.30, 66.12, 'r3'),
    ("DS$\\times$QW\n(R3)",        96.67, 84.43, 84.99, 'r3'),
    ("DS$\\times$Kimi\n(R4)",      97.0,  87.9,  81.9,  'r4'),
    ("QW$\\times$Kimi\n(R4)",      99.25, 88.68, 92.26, 'r4'),
    ("QW$\\times$DS\n(R4)",        97.27, 80.27, 86.69, 'r4'),
    ("DS\ntest–retest",            93.66, 80.39, 81.59, 'tr'),
    ("QW\ntest–retest",            97.65, 95.29, 88.71, 'tr'),
]

fig, ax = plt.subplots(figsize=(7.0, 2.9))
x = np.arange(len(CONFIGS))
offsets = {'REG-S': -0.26, 'REG-H': 0.0, 'UNREG': 0.26}
markers = {'REG-S': 'o', 'REG-H': 's', 'UNREG': 'D'}
for dom in ['REG-S', 'REG-H', 'UNREG']:
    ys = [c[{'REG-S': 1, 'REG-H': 2, 'UNREG': 3}[dom]] for c in CONFIGS]
    ax.scatter(x + offsets[dom], ys, marker=markers[dom], s=42,
               color=DOMAIN_COLORS[dom], label=dom, zorder=3,
               edgecolors='white', linewidths=0.6)
    for xi, yi in zip(x + offsets[dom], ys):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 5.5), ha='center', fontsize=5.8,
                    color=DOMAIN_COLORS[dom])

# group separators + labels
for gx in [2.5, 5.5]:
    ax.axvline(gx, color='#cccccc', lw=0.8, zorder=1)
ax.text(1.0, 57.5, 'cross-judge (R3)', ha='center', fontsize=7.5, color='#555555')
ax.text(4.0, 57.5, 'cross-judge (R4)', ha='center', fontsize=7.5, color='#555555')
ax.text(6.5, 57.5, 'test–retest', ha='center', fontsize=7.5, color='#555555')

ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in CONFIGS])
ax.set_ylim(55, 104)
ax.set_ylabel('binary agreement Po (%)')
ax.legend(loc='upper left', frameon=False, ncol=1, handlelength=1.0,
          borderpad=0.2, labelspacing=0.25)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#eeeeee', lw=0.6, zorder=0)

fig.savefig(OUT_DIR / 'fig5_multi_config.pdf')
fig.savefig(OUT_DIR / 'fig5_multi_config.png')
plt.close(fig)
print("  -> fig5_multi_config.pdf/.png")

# ═══════════════════════════════════════════════
# fig_ablation: UNREG anchored rubric ablation + REG-S placebo
# ═══════════════════════════════════════════════
print("fig_ablation ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7),
                               gridspec_kw={'width_ratios': [1.55, 1]})

C_ORIG = '#bbbbbb'
C_ANCH = '#1b9e77'

# (a) UNREG: binary + three-level, original vs anchored
labels = ['binary\n(PSI$\\geq$0.5)', 'three-level']
orig = [85.07, 78.03]
anch = [86.71, 82.56]
xx = np.arange(2)
w = 0.34
b1 = ax1.bar(xx - w/2, orig, w, color=C_ORIG, label='original rubric')
b2 = ax1.bar(xx + w/2, anch, w, color=C_ANCH, label='UNREG-anchored')
for b in list(b1) + list(b2):
    ax1.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width()/2, b.get_height()),
                 textcoords="offset points", xytext=(0, 2.5), ha='center', fontsize=7)
ax1.axhline(87.9, color='#d95f02', ls='--', lw=1.0)
ax1.axhline(90.9, color='#7570b3', ls='--', lw=1.0)
ax1.text(-0.42, 88.25, 'codification threshold 87.9', fontsize=6.3,
         color='#d95f02', ha='left')
ax1.text(-0.42, 91.25, 'instrument-fit threshold 90.9', fontsize=6.3,
         color='#7570b3', ha='left')
ax1.set_xticks(xx)
ax1.set_xticklabels(labels)
ax1.set_ylim(70, 96)
ax1.set_ylabel('DS$\\times$QW agreement Po (%)')
ax1.set_title('UNREG ($n$ = 1,038 paired)', fontsize=8.5)
ax1.legend(loc='lower right', frameon=False, fontsize=6.8)
ax1.spines[['top', 'right']].set_visible(False)

# (b) REG-S placebo
orig_p = [97.92]
anch_p = [97.57]
b1 = ax2.bar([-w/2], orig_p, w, color=C_ORIG, label='original rubric')
b2 = ax2.bar([w/2], anch_p, w, color=C_ANCH, label='UNREG-anchored')
for b in list(b1) + list(b2):
    ax2.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width()/2, b.get_height()),
                 textcoords="offset points", xytext=(0, 2.5), ha='center', fontsize=7)
ax2.axhline(97.92 - 2.0, color='#999999', ls=':', lw=0.9)
ax2.axhline(97.92 + 2.0, color='#999999', ls=':', lw=0.9)
ax2.text(0, 95.0, 'placebo band $\\pm$2 pp', fontsize=6.3, color='#777777',
         ha='center')
ax2.set_xticks([])
ax2.set_xlim(-0.6, 0.6)
ax2.set_ylim(94, 101)
ax2.set_title('REG-S placebo ($n$ = 288)', fontsize=8.5)
ax2.spines[['top', 'right']].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_ablation.pdf')
fig.savefig(OUT_DIR / 'fig_ablation.png')
plt.close(fig)
print("  -> fig_ablation.pdf/.png")
