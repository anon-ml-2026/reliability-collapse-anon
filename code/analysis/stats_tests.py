# Statistical tests and bootstrap CIs for draft v3 revision (sections 4.1, 4.2, 4.6).
# 4a: bootstrap 95% CIs for Cohen's kappa (human-human, AI-human, AI-AI UNREG)
# 4b: two-proportion z tests (UNREG vs REG-H; role-line pooled) +
#     per-model exact Wilcoxon signed-rank tests (role-line, domain)
# Usage: python code/analysis/stats_tests.py
import json, math, os, random, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(42)

# ---------- helpers ----------
def kappa_from_pairs(pairs):
    # pairs: list of (a, b) with values in {0.0, 0.5, 1.0}
    n = len(pairs)
    cats = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    po = sum(1 for a, b in pairs if a == b) / n
    pa = {c: sum(1 for a, _ in pairs if a == c) / n for c in cats}
    pb = {c: sum(1 for _, b in pairs if b == c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")

def pairs_from_confusion(cm):
    # keys like "A:0->B:0.5" or "0->0" or "R3:1->Human:0.5"
    pairs = []
    for k, v in cm.items():
        left, right = k.split("->")
        a = left.split(":")[-1]
        b = right.split(":")[-1]
        pairs.extend([(float(a), float(b))] * v)
    return pairs

def bootstrap_kappa_ci(pairs, reps=10000):
    n = len(pairs)
    ks = []
    for _ in range(reps):
        sample = [pairs[random.randrange(n)] for _ in range(n)]
        k = kappa_from_pairs(sample)
        if not math.isnan(k):
            ks.append(k)
    ks.sort()
    return ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks))]

def z_test_two_prop(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    # two-sided p via normal cdf
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, pval

def wilcoxon_exact(diffs):
    # exact signed-rank test, two-sided, zero diffs dropped
    d = [x for x in diffs if x != 0]
    n = len(d)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(d[order[j + 1]]) - abs(d[order[i]])) < 1e-12:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    Wplus = sum(r for r, x in zip(ranks, d) if x > 0)
    total = sum(ranks)
    # exact distribution via sign enumeration
    counts = {}
    for signs in itertools.product([0, 1], repeat=n):
        w = sum(r for r, s in zip(ranks, signs) if s)
        counts[w] = counts.get(w, 0) + 1
    extreme = sum(c for w, c in counts.items() if w <= min(Wplus, total - Wplus) or w >= max(Wplus, total - Wplus))
    return Wplus, n, extreme / 2 ** n

# ---------- load ----------
r3 = json.load(open(os.path.join(HERE, "..", "..", "data", "scores", "r3_judge_deepseek.json"), encoding="utf-8"))
valid = [r for r in r3 if not r.get("parse_error") and r.get("BVI") is not None]
hv = json.load(open(os.path.join(HERE, "..", "..", "results", "human_anchor_analysis.json"), encoding="utf-8"))
cj = json.load(open(os.path.join(HERE, "..", "..", "results", "cross_judge_deepseek_claude.json"), encoding="utf-8"))

print("=" * 60)
print("4b-1. UNREG vs REG-H BVI=1 rate (DeepSeek judge, R3)")
def dom_counts(rs, dom):
    sub = [r for r in rs if r["domain"] == dom]
    return sum(1 for r in sub if r["BVI"] == 1), len(sub)
xu, nu = dom_counts(valid, "UNREG")
xh, nh = dom_counts(valid, "REG-H")
z, p = z_test_two_prop(xu, nu, xh, nh)
print(f"UNREG {xu}/{nu} ({100*xu/nu:.1f}%) vs REG-H {xh}/{nh} ({100*xh/nh:.1f}%)")
print(f"z = {z:.2f}, p = {p:.2e}")
# per-model Wilcoxon: UNREG rate vs REG-H rate
models = sorted({r["model_id"] for r in valid})
diffs = []
for m in models:
    sub = [r for r in valid if r["model_id"] == m]
    ru = dom_counts(sub, "UNREG"); rh = dom_counts(sub, "REG-H")
    diffs.append(ru[0] / ru[1] - rh[0] / rh[1])
W, n, p = wilcoxon_exact(diffs)
print(f"per-model Wilcoxon (UNREG - REG-H): W+ = {W:.1f}, n = {n}, exact p = {p:.4f}")

print("=" * 60)
print("4b-2. Role-line effect (non-CTL, DeepSeek judge, R3)")
def role_counts(rs, rl):
    sub = [r for r in rs if r["domain"] != "CTL" and r["role_line"] == rl]
    return sum(1 for r in sub if r["BVI"] == 1), len(sub)
xn, nn = role_counts(valid, "no-role")
xw, nw = role_counts(valid, "with-role")
xb, nb = role_counts(valid, "boundary-role")
for name, (x1, n1, x2, n2) in {
    "no-role vs boundary-role": (xn, nn, xb, nb),
    "with-role vs no-role": (xw, nw, xn, nn),
}.items():
    z, p = z_test_two_prop(x1, n1, x2, n2)
    print(f"{name}: {100*x1/n1:.1f}% vs {100*x2/n2:.1f}%  z = {z:.2f}, p = {p:.4f}")
# per-model Wilcoxon
for name, rl1, rl2 in [("no-role - boundary-role", "no-role", "boundary-role"),
                        ("with-role - no-role", "with-role", "no-role")]:
    diffs = []
    for m in models:
        sub = [r for r in valid if r["model_id"] == m]
        r1 = role_counts(sub, rl1); r2 = role_counts(sub, rl2)
        diffs.append(r1[0] / r1[1] - r2[0] / r2[1])
    W, n, p = wilcoxon_exact(diffs)
    print(f"per-model Wilcoxon ({name}): W+ = {W:.1f}, n = {n}, exact p = {p:.4f}")

print("=" * 60)
print("4a. Bootstrap 95% CI for Cohen's kappa (10000 reps)")
def report(label, entry):
    pairs = pairs_from_confusion(entry["confusion_matrix"])
    k = entry["cohen_kappa"]
    lo, hi = bootstrap_kappa_ci(pairs)
    print(f"{label:24s} n={len(pairs):4d}  k={k:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

print("-- inter-human (Table 3) --")
for pair, entry in hv["inter_human"].items():
    report(pair, entry)
print("-- R3 vs human (Table 4) --")
for ann, entry in hv["r3_vs_human"].items():
    report(f"R3 vs {ann}", entry)
report("R3 vs majority", hv["r3_vs_human_majority"])
report("AI-AI (anchor subset)", cj["anchor_subset"])
print("-- AI-AI cross-judge UNREG (Table 2) --")
u = cj["domains"]["UNREG"]
pairs = pairs_from_confusion(u["confusion_matrix"])
lo, hi = bootstrap_kappa_ci(pairs)
print(f"{'AI-AI UNREG':24s} n={len(pairs):4d}  k={u['cohen_kappa']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
