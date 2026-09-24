"""Calibration figure for the synthetic-audit validation (paper B, appendix).

Panel A: P(survives) vs true margin Delta for the S1 (single-judge) and
S2 (intersection) rules, with the real-data margins marked.  Panel B:
false-survival under a true null when DeepSeek (one-sided) or both judges
(shared) over-flag UNREG by delta.  Numbers are the locked values from
synthetic_calibration.py (300 simulated datasets x 2,000-draw bootstrap).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "V16" / "figures"

deltas = [0, 2.5, 5, 7.5, 10, 12.5, 15, 20, 25]
s1 = [0.013, 0.133, 0.490, 0.847, 0.973, 1.000, 1.000, 1.000, 1.000]
s2 = [0.010, 0.130, 0.390, 0.760, 0.917, 0.990, 0.997, 1.000, 1.000]

bd = [5, 10, 15, 19]
b_s1 = [0.200, 0.917, 1.000, 1.000]
b_s2 = [0.010, 0.110, 0.313, 0.460]
c_s2 = [0.183, 0.827, 0.993, 1.000]

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), dpi=300)

ax = axes[0]
ax.axhline(0.8, color="#999999", lw=0.7, ls="--", zorder=1)
ax.plot(deltas, s1, "o-", ms=3.4, lw=1.3, color="#1f6fb4", label="S1 single-judge")
ax.plot(deltas, s2, "s-", ms=3.2, lw=1.3, color="#d1681e", label="S2 intersection")
for x, lab, col in ((11.2, "S2 margin", "#d1681e"), (19.0, "S1 margin", "#1f6fb4")):
    ax.axvline(x, color=col, lw=0.8, ls=":", zorder=1)
    ax.annotate(lab, (x, 0.47), fontsize=6.0, rotation=90, ha="right", color=col)
ax.set_xlabel("true UNREG-highest margin (pp)", fontsize=7.5)
ax.set_ylabel("P(verdict = survives)", fontsize=7.5)
ax.set_ylim(-0.03, 1.05)
ax.tick_params(labelsize=6.8)
ax.legend(fontsize=6.2, loc="lower right", frameon=False)
ax.set_title("(a) power curve, unbiased judges", fontsize=7.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = axes[1]
xpos = np.arange(4)
wdt = 0.27
ax.bar(xpos - wdt, b_s1, wdt, color="#1f6fb4", label="S1, one-sided")
ax.bar(xpos, b_s2, wdt, color="#d1681e", label="S2, one-sided")
ax.bar(xpos + wdt, c_s2, wdt, color="#8a5db4", label="S2, shared")
ax.set_xticks(xpos)
ax.set_xticklabels([str(d) for d in bd], fontsize=6.8)
ax.set_xlabel("injected UNREG over-flag bias $\\delta$ (pp)", fontsize=7.5)
ax.set_ylabel("P(false survives)", fontsize=7.5)
ax.set_ylim(0, 1.05)
ax.tick_params(labelsize=6.8)
ax.legend(fontsize=5.9, loc="upper left", frameon=False)
ax.set_title("(b) false survival under a true null", fontsize=7.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout(pad=0.5)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_synthetic_calibration.{ext}", bbox_inches="tight")
print(f"written {OUT / 'fig_synthetic_calibration.pdf'}")
