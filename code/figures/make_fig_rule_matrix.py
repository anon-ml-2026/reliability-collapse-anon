"""Specification-curve figure for the revision-v8 judge-rule x threshold matrix.

Reads code/analysis/revision_v8_numbers.json (produced by
claim_audit_revision_v8.py) and draws the eight specification margins with
their 95% probe-clustered intervals, sorted by point estimate.  Candidate
specifications are drawn in blue, strict in orange; open markers denote
intervals that include zero.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
NUMBERS = HERE.parent / "analysis" / "revision_v8_numbers.json"
OUT = HERE.parents[2] / "V14" / "figures"

RULE_NAME = {"DS": "DeepSeek only", "CL": "Claude only",
             "AND": "both ($\\cap$)", "OR": "either ($\\cup$)"}

data = json.loads(NUMBERS.read_text(encoding="utf-8"))["matrix"]
rows = []
for key, v in data.items():
    thr, rule = key.split("|")
    h, s, u, m, lo, hi, models = v
    rows.append((thr, rule, m, lo, hi, models))
rows.sort(key=lambda r: r[2])

fig, ax = plt.subplots(figsize=(3.35, 2.3), dpi=300)
for i, (thr, rule, m, lo, hi, models) in enumerate(rows):
    color = "#1f6fb4" if thr == "cand" else "#d1681e"
    excl = lo > 0 or hi < 0
    ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=4.2,
                color=color, mfc=color if excl else "white", mec=color,
                ecolor=color, elinewidth=1.1, capsize=2.2, capthick=1.1, zorder=3)
    ax.annotate(f"{models}/8", (i, hi + 1.6), ha="center", fontsize=5.6,
                color="#555555")

ax.axhline(0, color="#333333", lw=0.8, zorder=1)
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([RULE_NAME[r[1]] for r in rows], fontsize=6.2, rotation=32,
                   ha="right")
ax.set_ylabel("UNREG-highest margin (pp)", fontsize=7.5)
ax.tick_params(axis="y", labelsize=6.8)
ax.set_ylim(-20, 50)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

handles = [
    plt.Line2D([], [], color="#1f6fb4", marker="o", ms=4, ls="-",
               mfc="#1f6fb4", label="candidate ($\\geq$0.5)"),
    plt.Line2D([], [], color="#d1681e", marker="o", ms=4, ls="-",
               mfc="#d1681e", label="strict (=1)"),
    plt.Line2D([], [], color="#666666", marker="o", ms=4, ls="",
               mfc="white", mec="#666666", label="CI includes 0"),
]
ax.legend(handles=handles, fontsize=6.0, loc="upper left", frameon=False)
fig.tight_layout(pad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_rule_matrix.{ext}", bbox_inches="tight")
print(f"written to {OUT / 'fig_rule_matrix.pdf'}")
