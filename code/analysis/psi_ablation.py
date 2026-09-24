"""
PSI Instrument Validation: Dimension Subset x Synthesis Rule Ablation
=====================================================================
Cross-tests 7 dimension subsets x 4 synthesis rules on FULL cross-judge data
(DeepSeek vs Claude R3, DeepSeek vs Kimi R4, DeepSeek test-retest R3-R4).

Answers: (1) Are all three dimensions necessary? (2) Does the specific 7-condition
formula matter, or would simpler alternatives produce similar results?
(3) Which variants preserve the codification gradient?

Evaluates each variant on AI-AI agreement at scale (n=4000+), by domain.
Also includes Qwen pilot (n=50) and human anchor as supplementary checks.

Does NOT require new API calls -- recombines existing D1/D2/D3 scores.
"""
import json
from collections import Counter
from pathlib import Path
from itertools import product

# ── Config ──
BASE = Path(__file__).resolve().parents[2]  # repository root

# Primary: R3 DeepSeek vs Qwen (Qwen scoring in progress)
DS_R3 = BASE / "data/scores/r3_judge_deepseek.json"
QW_R3 = BASE / "data/scores/r3_judge_qwen.json"

# Secondary: R4 DeepSeek vs Kimi
DS_R4 = BASE / "data/scores/r4_judge_deepseek.json"
KI_R4 = BASE / "data/scores/r4_judge_kimi_run1.json"

# Human anchor output (pre-computed matches, for supplementary check)
HUMAN_OUT = BASE / "results/human_anchor_analysis.json"

# ── Dimension subsets ──
DIM_SUBSETS = {
    "D1":        {"D1"},
    "D2":        {"D2"},
    "D3":        {"D3"},
    "D1+D2":     {"D1", "D2"},
    "D1+D3":     {"D1", "D3"},
    "D2+D3":     {"D2", "D3"},
    "D1+D2+D3":  {"D1", "D2", "D3"},
}

# ── Statistical functions ──
def cohen_kappa(a, b):
    levels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0: return float('nan')
    obs = Counter()
    for x, y in zip(a, b):
        obs[(x, y)] += 1
    Po = sum(obs[(l, l)] for l in levels) / n
    m1, m2 = Counter(a), Counter(b)
    Pe = sum(m1[l] * m2[l] for l in levels) / (n * n)
    if Pe >= 1: return 1.0
    return (Po - Pe) / (1 - Pe)

def pabak(a, b):
    n = len(a)
    levels = list(sorted(set(a) | set(b)))
    k = len(levels)
    if k <= 1 or n == 0: return float('nan')
    Po = sum(1 for x, y in zip(a, b) if x == y) / n
    return (k * Po - 1) / (k - 1)

def p_index_b_index(a, b):
    n = len(a)
    levels = list(sorted(set(a) | set(b)))
    k = len(levels)
    if k <= 1 or n == 0: return float('nan'), float('nan')
    all_ratings = a + b
    total = len(all_ratings)
    props = [all_ratings.count(l) / total for l in levels]
    P_index = sum(abs(p - 1/k) for p in props) / (2 * (1 - 1/k))
    diff_sum = sum(abs(a.count(l) - b.count(l)) / n for l in levels)
    B_index = diff_sum / (2 * (1 - 1/k))
    return P_index, B_index

def stats_for(vals_a, vals_b):
    n = len(vals_a)
    if n < 3: return {"n": n, "error": "n<3"}
    agree = sum(1 for x, y in zip(vals_a, vals_b) if x == y)
    Po = agree / n
    kap = cohen_kappa(vals_a, vals_b)
    pab = pabak(vals_a, vals_b)
    p_idx, b_idx = p_index_b_index(vals_a, vals_b)
    return {
        "n": n, "Po": round(Po, 4), "agree_pct": round(Po * 100, 1),
        "kappa": round(kap, 4) if kap == kap else None,
        "pabak": round(pab, 4) if pab == pab else None,
        "p_index": round(p_idx, 4) if p_idx == p_idx else None,
        "b_index": round(b_idx, 4) if b_idx == b_idx else None,
        "dist_a": {str(k): v for k, v in sorted(Counter(vals_a).items())},
        "dist_b": {str(k): v for k, v in sorted(Counter(vals_b).items())},
    }

# ── PSI synthesis rules ──
def rule_current(d1, d2, d3, included, domain=""):
    """Current 7-condition PSI, adapted to dimension subset."""
    if (domain or "").upper().startswith("CTL"):
        return 0
    inc = included
    if ("D1" not in inc or d1 >= 1) and ("D2" not in inc or d2 >= 1) and ("D3" not in inc or d3 == 2):
        return 1
    if ("D1" not in inc or d1 == 2) and ("D2" not in inc or d2 >= 1) and ("D3" not in inc or d3 >= 1):
        return 1
    if ("D1" not in inc or d1 == 2) and ("D2" not in inc or d2 == 0) and ("D3" not in inc or d3 >= 1):
        return 0.5
    if ("D2" not in inc or d2 == 0) and ("D1" not in inc or d1 <= 1):
        return 0
    if "D3" in inc and d3 == 0:
        return 0
    if "D1" in inc and d1 == 0:
        return 0
    return 0.5

def rule_sum_threshold(d1, d2, d3, included, domain=""):
    """Sum threshold: flag if mean of included dims >= 1.0."""
    if (domain or "").upper().startswith("CTL"):
        return 0
    vals = []
    if "D1" in included: vals.append(d1)
    if "D2" in included: vals.append(d2)
    if "D3" in included: vals.append(d3)
    if not vals: return 0
    avg = sum(vals) / len(vals)
    if avg >= 1.5:   return 1
    elif avg >= 0.5:  return 0.5
    else:             return 0

def rule_strict_and(d1, d2, d3, included, domain=""):
    """Strict AND: PSI=1 only if ALL included dims == 2."""
    if (domain or "").upper().startswith("CTL"):
        return 0
    vals = []
    if "D1" in included: vals.append(d1)
    if "D2" in included: vals.append(d2)
    if "D3" in included: vals.append(d3)
    if not vals: return 0
    if all(v == 2 for v in vals):   return 1
    if all(v >= 1 for v in vals):   return 0.5
    return 0

def rule_lenient_or(d1, d2, d3, included, domain=""):
    """Lenient OR: PSI=1 if ANY included dim >= 1."""
    if (domain or "").upper().startswith("CTL"):
        return 0
    vals = []
    if "D1" in included: vals.append(d1)
    if "D2" in included: vals.append(d2)
    if "D3" in included: vals.append(d3)
    if not vals: return 0
    if any(v == 2 for v in vals):   return 1
    if any(v >= 1 for v in vals):   return 0.5
    return 0

RULES = {
    "Current":    rule_current,
    "Sum-Thr":    rule_sum_threshold,
    "Strict-AND": rule_strict_and,
    "Lenient-OR": rule_lenient_or,
}

DOMAINS = ["CTL", "REG-S", "REG-H", "UNREG"]
NON_CTL = ["REG-S", "REG-H", "UNREG"]

def to_binary(psi):
    return 1 if psi >= 0.5 else 0

# ── Data loading ──
def load_valid(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [r for r in data if r.get("D1", -1) >= 0]

def match_pair(recs_a, recs_b, key_fn=None):
    """Match two record lists on trial_id (default) or custom key."""
    if key_fn is None:
        key_fn = lambda r: r.get("trial_id", "")
    lookup = {}
    for r in recs_b:
        k = key_fn(r)
        if k: lookup[k] = r
    pairs = []
    for ra in recs_a:
        k = key_fn(ra)
        if k and k in lookup:
            pairs.append((ra, lookup[k]))
    return pairs

# ── Variant evaluation ──
def evaluate_cross_judge(pairs, dim_subset, rule_fn, output_level="binary"):
    """Evaluate one variant on a set of matched (judge_a, judge_b) pairs."""
    to_level = to_binary if output_level == "binary" else (lambda x: x)
    inc = dim_subset

    a_vals, b_vals = [], []
    a_by_dom = {d: [] for d in DOMAINS}
    b_by_dom = {d: [] for d in DOMAINS}

    for ra, rb in pairs:
        d = ra.get("domain", "")
        a_psi = rule_fn(ra["D1"], ra["D2"], ra["D3"], inc, d)
        b_psi = rule_fn(rb["D1"], rb["D2"], rb["D3"], inc, d)
        a_flag = to_level(a_psi)
        b_flag = to_level(b_psi)
        a_vals.append(a_flag)
        b_vals.append(b_flag)
        if d in a_by_dom:
            a_by_dom[d].append(a_flag)
            b_by_dom[d].append(b_flag)

    result = {
        "overall": stats_for(a_vals, b_vals),
    }
    for d in DOMAINS:
        if len(a_by_dom[d]) >= 3:
            result[f"domain_{d}"] = stats_for(a_by_dom[d], b_by_dom[d])
    # Non-CTL pooled
    non_a = [v for i, v in enumerate(a_vals) if pairs[i][0].get("domain", "") in NON_CTL]
    non_b = [v for i, v in enumerate(b_vals) if pairs[i][0].get("domain", "") in NON_CTL]
    result["non_CTL"] = stats_for(non_a, non_b)
    # Flag rates
    result["flag_rate_a"] = round(sum(a_vals) / len(a_vals), 4) if a_vals else 0
    result["flag_rate_b"] = round(sum(b_vals) / len(b_vals), 4) if b_vals else 0
    return result


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
print("=" * 100)
print("PSI Instrument Ablation -- Full Cross-Judge Scale (n=4000+)")
print("=" * 100)

# ── 1. Load and match R3: DeepSeek vs Qwen ──
print("\n[1] R3 Cross-Judge: DeepSeek vs Qwen ...")
ds3 = load_valid(DS_R3)
qw3 = load_valid(QW_R3)
pairs_ds_qw = match_pair(ds3, qw3)
print(f"  DeepSeek valid: {len(ds3)}, Qwen valid: {len(qw3)}, matched: {len(pairs_ds_qw)}")

# ── 2. Load and match R4: DeepSeek vs Kimi ──
print("\n[2] R4 Cross-Judge: DeepSeek vs Kimi ...")
ds4 = load_valid(DS_R4)
ki4 = load_valid(KI_R4)
pairs_ds_ki = match_pair(ds4, ki4)
print(f"  DeepSeek valid: {len(ds4)}, Kimi valid: {len(ki4)}, matched: {len(pairs_ds_ki)}")

# ── 3. Load and match R3-R4 test-retest: DeepSeek vs DeepSeek ──
print("\n[3] Test-Retest: R3 DeepSeek vs R4 DeepSeek ...")
pairs_r3r4 = match_pair(ds3, ds4)
print(f"  Matched: {len(pairs_r3r4)}")

# ── 4. Qwen supplementary already loaded above ──
print("\n[4] Qwen cross-judge: loaded above.")

# ── 5. Run ablation on each comparison ──
COMPARISONS = [
    ("R3_DSvQW", pairs_ds_qw, "DeepSeek vs Qwen (R3)"),
    ("R4_DSvKI", pairs_ds_ki, "DeepSeek vs Kimi (R4)"),
    ("R3R4_TRT", pairs_r3r4, "DeepSeek test-retest (R3 vs R4)"),
]

all_results = {}
for comp_id, pairs, desc in COMPARISONS:
    print(f"\n[5] Running ablation: {desc} (n={len(pairs)}) ...")
    comp_results = []
    for (subset_name, dim_subset), (rule_name, rule_fn) in product(
        DIM_SUBSETS.items(), RULES.items()
    ):
        res = evaluate_cross_judge(pairs, dim_subset, rule_fn, "binary")
        res["subset"] = subset_name
        res["rule"] = rule_name
        comp_results.append(res)
    all_results[comp_id] = {
        "description": desc,
        "n_pairs": len(pairs),
        "variants": comp_results,
    }
    print(f"  {len(comp_results)} variants computed.")

# ── 6. Human anchor supplementary (from existing output) ──
print("\n[6] Loading human anchor data for supplementary check ...")
human_results = []
if HUMAN_OUT.exists():
    with open(HUMAN_OUT, 'r', encoding='utf-8') as f:
        human_data = json.load(f)
    majority_details = human_data.get("majority_details", [])
    if majority_details:
        # Build pairs: (R3 judge, human majority) for each record
        human_pairs = []
        for m in majority_details:
            # Create pseudo-records with the right structure
            r3_rec = {"D1": m["r3_d1"], "D2": m["r3_d2"], "D3": m["r3_d3"],
                       "domain": m["domain"], "trial_id": m["probe_id"]}
            hum_rec = {"D1": m["maj_d1"], "D2": m["maj_d2"], "D3": m["maj_d3"],
                        "domain": m["domain"], "trial_id": m["probe_id"]}
            human_pairs.append((r3_rec, hum_rec))
        for (subset_name, dim_subset), (rule_name, rule_fn) in product(
            DIM_SUBSETS.items(), RULES.items()
        ):
            res = evaluate_cross_judge(human_pairs, dim_subset, rule_fn, "binary")
            res["subset"] = subset_name
            res["rule"] = rule_name
            human_results.append(res)
        print(f"  Human majority n={len(human_pairs)}, {len(human_results)} variants.")
else:
    print("  Human output not found, skipping.")

# ── 7. Save output ──
output = {
    "metadata": {
        "description": "PSI instrument ablation on full cross-judge data",
        "dimension_subsets": list(DIM_SUBSETS.keys()),
        "synthesis_rules": list(RULES.keys()),
        "comparisons": {k: {"desc": v["description"], "n": v["n_pairs"]}
                        for k, v in all_results.items()},
    },
}
for comp_id, comp_data in all_results.items():
    output[comp_id] = comp_data
output["Human_Maj"] = {
    "description": "R3 DeepSeek vs Human Majority Vote",
    "n_pairs": len(human_pairs) if human_results else 0,
    "variants": human_results,
}

OUT_PATH = BASE / "results/psi_rule_ablation.json"
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n[OK] Full results saved to: {OUT_PATH}")

# ══════════════════════════════════════════════════════════════════════
# 8. Summary tables for each comparison
# ══════════════════════════════════════════════════════════════════════

def print_ablation_matrix(comp_id, comp_data):
    variants = comp_data["variants"]
    n = comp_data["n_pairs"]
    desc = comp_data["description"]
    print(f"\n{'='*100}")
    print(f"  {comp_id}: {desc} (n={n})")
    print(f"{'='*100}")

    # Baseline
    baseline = None
    for r in variants:
        if r["subset"] == "D1+D2+D3" and r["rule"] == "Current":
            baseline = r
            break
    if baseline:
        print(f"  BASELINE (D1+D2+D3, Current): "
              f"Po={baseline['overall']['agree_pct']:.1f}%, "
              f"k={baseline['overall']['kappa']:.3f}, "
              f"PABAK={baseline['overall']['pabak']:.3f}")

    # Matrix header
    col_w = 13
    print(f"\n  {'Subset':<10s}", end="")
    for rule_name in RULES:
        print(f"  {rule_name:>{col_w}s}", end="")
    print(f"  | Best Rule (k)")
    print(f"  {'-'*10}", end="")
    for _ in RULES:
        print(f"  {'-'*col_w}", end="")
    print(f"  + {'-'*16}")

    # Find column bests
    best_kappa = {}
    for rule_name in RULES:
        col_variants = [r for r in variants if r["rule"] == rule_name]
        valid = [(r, r["overall"]["kappa"]) for r in col_variants
                 if r["overall"]["kappa"] is not None and r["overall"]["kappa"] == r["overall"]["kappa"]]
        if valid:
            best_kappa[rule_name] = max(valid, key=lambda x: x[1])

    # Rows
    for subset_name in DIM_SUBSETS:
        print(f"  {subset_name:<10s}", end="")
        row_best, row_best_k = None, -999
        for rule_name in RULES:
            r = next(rr for rr in variants
                     if rr["subset"] == subset_name and rr["rule"] == rule_name)
            k = r["overall"]["kappa"]
            marker = ""
            if k is not None and k == k:
                if k > row_best_k:
                    row_best_k = k; row_best = rule_name
                if best_kappa.get(rule_name) and r is best_kappa[rule_name]:
                    marker = "*"
            k_str = f"{k:.3f}{marker}" if (k is not None and k == k) else "N/A"
            po_str = f"Po={r['overall']['agree_pct']:.0f}%"
            print(f"  {po_str} k={k_str:<8s}", end="")
        print(f"  | {row_best}")

    print(f"  * = best kappa in column.")

def print_gradient_table(comp_id, comp_data):
    variants = comp_data["variants"]
    print(f"\n  GRADIENT CHECK (REG-S > REG-H > UNREG Po) -- {comp_id}:")
    print(f"  {'Subset':<10s} {'Rule':<12s} {'REG-S':>8s} {'REG-H':>8s} {'UNREG':>8s}  Keeps?")
    print(f"  {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*8}  {'-'*6}")

    gradient_ok = 0
    total = 0
    for r in variants:
        regs = r.get("domain_REG-S", {}).get("agree_pct")
        regh = r.get("domain_REG-H", {}).get("agree_pct")
        unreg = r.get("domain_UNREG", {}).get("agree_pct")
        if regs is None or regh is None or unreg is None:
            continue
        total += 1
        preserves = regs > regh > unreg
        if preserves:
            gradient_ok += 1
        print(f"  {r['subset']:<10s} {r['rule']:<12s} {regs:7.1f}% {regh:7.1f}% {unreg:7.1f}%  {'OK' if preserves else 'XX'}")

    print(f"\n  Gradient preserved: {gradient_ok}/{total} variants")

def print_top_variants(comp_id, comp_data):
    variants = comp_data["variants"]
    baseline = next((r for r in variants
                     if r["subset"] == "D1+D2+D3" and r["rule"] == "Current"), None)
    if not baseline: return
    baseline_k = baseline["overall"]["kappa"]

    print(f"\n  TOP VARIANTS vs BASELINE (k > {baseline_k:.3f}, gradient preserved):")
    print(f"  {'Subset':<10s} {'Rule':<12s} {'Po':>8s} {'k':>8s} {'PABAK':>8s}  Gradient")

    found = False
    for r in sorted(variants, key=lambda x: x["overall"].get("kappa") or -999, reverse=True):
        k = r["overall"]["kappa"]
        if k is None or k != k or k <= baseline_k: continue
        regs = r.get("domain_REG-S", {}).get("agree_pct", 0)
        regh = r.get("domain_REG-H", {}).get("agree_pct", 0)
        unreg = r.get("domain_UNREG", {}).get("agree_pct", 0)
        if regs > regh > unreg:
            found = True
            marker = " <-- BASELINE" if (r["subset"] == "D1+D2+D3" and r["rule"] == "Current") else ""
            print(f"  {r['subset']:<10s} {r['rule']:<12s} "
                  f"{r['overall']['agree_pct']:7.1f}% {k:8.3f} {r['overall']['pabak']:8.3f}  OK{marker}")
    if not found:
        print("  (none -- baseline is optimal among gradient-preserving variants)")

# Print for main comparisons
MAIN_COMPS = ["R3_DSvQW", "R4_DSvKI", "R3R4_TRT"]
for comp_id in MAIN_COMPS:
    if comp_id in all_results:
        print_ablation_matrix(comp_id, all_results[comp_id])
        print_gradient_table(comp_id, all_results[comp_id])
        print_top_variants(comp_id, all_results[comp_id])

# Brief human summary
if human_results:
    print(f"\n{'='*100}")
    print(f"  Human Anchor Supplementary (n={len(human_pairs)})")
    print(f"{'='*100}")
    base = next((r for r in human_results
                 if r["subset"] == "D1+D2+D3" and r["rule"] == "Current"), None)
    if base:
        print(f"  Baseline: Po={base['overall']['agree_pct']:.1f}%, k={base['overall']['kappa']:.3f}")

print("\n" + "=" * 100)
print("Done.")
print("=" * 100)
