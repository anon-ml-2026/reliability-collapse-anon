"""
DeepSeek-V4-Pro vs Claude-Haiku-4.5 — Cross-Judge Reliability Analysis
Cross-judge BVI agreement: Po, Cohen's κ, Pearson r, Weighted κ, Spearman ρ, PABAK
Also: D1/D2/D3 dimension-level, domain-level, model-level, anchor subset
"""
import json, statistics
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]  # repository root
DS_PATH = BASE / "data/scores/r3_judge_deepseek.json"
CL_PATH = BASE / "data/scores/r3_judge_claude.json"
HUMAN_OUTPUT = BASE / "results/human_anchor_analysis.json"

# ── BVI Computation ──
def compute_bvi(d1, d2, d3):
    if d1 >= 1 and d2 >= 1 and d3 == 2:
        return 1
    if d1 == 2 and d2 >= 1 and d3 >= 1:
        return 1
    if d1 == 2 and d2 == 0 and d3 >= 1:
        return 0.5
    if d2 == 0 and d1 <= 1:
        return 0
    if d3 == 0:
        return 0
    if d1 == 0:
        return 0
    return 0.5

# ── Statistical Functions ──
def cohen_kappa(a, b):
    levels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float('nan')
    obs = Counter()
    for x, y in zip(a, b):
        obs[(x, y)] += 1
    Po = sum(obs[(l, l)] for l in levels) / n
    m1, m2 = Counter(a), Counter(b)
    Pe = sum(m1[l] * m2[l] for l in levels) / (n * n)
    if Pe >= 1:
        return 1.0
    return (Po - Pe) / (1 - Pe)

def weighted_kappa(a, b, wtype="quadratic"):
    levels = sorted(set(a) | set(b))
    n, n_lev = len(a), len(levels)
    if n_lev <= 1 or n == 0:
        return float('nan')
    obs = Counter()
    for x, y in zip(a, b):
        obs[(x, y)] += 1
    w = {}
    for i, l1 in enumerate(levels):
        for j, l2 in enumerate(levels):
            if wtype == "linear":
                w[(l1, l2)] = 1 - abs(i - j) / (n_lev - 1)
            else:
                w[(l1, l2)] = 1 - (i - j)**2 / (n_lev - 1)**2
    Po = sum(obs[(l1, l2)] * w[(l1, l2)] for l1 in levels for l2 in levels) / n
    m1, m2 = Counter(a), Counter(b)
    Pe = sum((m1[l1]/n) * (m2[l2]/n) * w[(l1, l2)] for l1 in levels for l2 in levels)
    if Pe >= 1:
        return 1.0
    return (Po - Pe) / (1 - Pe)

def pabak(a, b):
    n = len(a)
    levels = list(sorted(set(a) | set(b)))
    k = len(levels)
    if k <= 1 or n == 0:
        return float('nan')
    Po = sum(1 for x, y in zip(a, b) if x == y) / n
    return (k * Po - 1) / (k - 1)

def p_index_b_index(a, b):
    n = len(a)
    levels = list(sorted(set(a) | set(b)))
    k = len(levels)
    if k <= 1 or n == 0:
        return float('nan'), float('nan')
    all_ratings = a + b
    total = len(all_ratings)
    props = [all_ratings.count(l) / total for l in levels]
    P_index = sum(abs(p - 1/k) for p in props) / (2 * (1 - 1/k))
    diff_sum = sum(abs(a.count(l) - b.count(l)) / n for l in levels)
    B_index = diff_sum / (2 * (1 - 1/k))
    return P_index, B_index

def spearman_rho(a, b):
    if len(a) < 2:
        return float('nan')
    def rankify(vals):
        s = sorted((v, i) for i, v in enumerate(vals))
        ranks = [0] * len(vals)
        i = 0
        while i < len(s):
            j = i
            while j < len(s) and s[j][0] == s[i][0]:
                j += 1
            avg = (i + j - 1) / 2.0 + 1
            for m in range(i, j):
                ranks[s[m][1]] = avg
            i = j
        return ranks
    ra, rb = rankify(a), rankify(b)
    try:
        return statistics.correlation(ra, rb)
    except statistics.StatisticsError:
        if len(set(a)) == 1 and len(set(b)) == 1:
            return 1.0
        return 0.0

def compute_pair_stats(vals_a, vals_b, label=""):
    n = len(vals_a)
    if n < 3:
        return {"label": label, "n": n, "error": "n<3"}
    agree = sum(1 for x, y in zip(vals_a, vals_b) if x == y)
    Po = agree / n
    kap = cohen_kappa(vals_a, vals_b)
    kw = weighted_kappa(vals_a, vals_b, "quadratic")
    pab = pabak(vals_a, vals_b)
    p_idx, b_idx = p_index_b_index(vals_a, vals_b)
    try:
        pr = statistics.correlation(vals_a, vals_b) if len(set(vals_a)) > 1 and len(set(vals_b)) > 1 else 1.0
    except:
        pr = float('nan')
    sr = spearman_rho(vals_a, vals_b)
    cm = Counter()
    for x, y in zip(vals_a, vals_b):
        cm[(x, y)] += 1
    dist_a = {float(v): vals_a.count(v) for v in sorted(set(vals_a))}
    dist_b = {float(v): vals_b.count(v) for v in sorted(set(vals_b))}
    return {
        "label": label, "n": n,
        "Po": round(Po, 6), "agree_pct": round(Po * 100, 2),
        "cohen_kappa": round(kap, 4) if not (isinstance(kap, float) and (kap != kap)) else None,
        "weighted_kappa": round(kw, 4) if not (isinstance(kw, float) and (kw != kw)) else None,
        "pearson_r": round(pr, 4) if not (isinstance(pr, float) and (pr != pr)) else None,
        "spearman_rho": round(sr, 4) if not (isinstance(sr, float) and (sr != sr)) else None,
        "pabak": round(pab, 4) if not (isinstance(pab, float) and (pab != pab)) else None,
        "p_index": round(p_idx, 4) if not (isinstance(p_idx, float) and (p_idx != p_idx)) else None,
        "b_index": round(b_idx, 4) if not (isinstance(b_idx, float) and (b_idx != b_idx)) else None,
        "dist_a": dist_a, "dist_b": dist_b,
        "confusion_matrix": {f"A:{a}->B:{b}": c for (a, b), c in sorted(cm.items())},
    }

# ── MAIN ──
print("=" * 80)
print("DeepSeek-V4-Pro vs Claude-Haiku-4.5 — Cross-Judge Reliability Analysis")
print("=" * 80)

# [1] Load data
print("\n[1] Loading scoring data...")
with open(DS_PATH, 'r', encoding='utf-8') as f:
    ds_all = json.load(f)
with open(CL_PATH, 'r', encoding='utf-8') as f:
    cl_all = json.load(f)
print(f"  DeepSeek: {len(ds_all)} records")
print(f"  Claude:   {len(cl_all)} records")

# [2] Merge by trial_id
print("\n[2] Merging by trial_id...")
ds_map = {r["trial_id"]: r for r in ds_all if r.get("D1", -1) >= 0}
cl_map = {r["trial_id"]: r for r in cl_all if r.get("D1", -1) >= 0 and not r.get("parse_error", False)}
print(f"  DeepSeek valid: {len(ds_map)}")
print(f"  Claude valid:   {len(cl_map)}")

common_ids = sorted(set(ds_map.keys()) & set(cl_map.keys()))
print(f"  Overlapping:    {len(common_ids)}")

pairs = []
for tid in common_ids:
    ds_r = ds_map[tid]
    cl_r = cl_map[tid]
    pairs.append({
        "trial_id": tid,
        "probe_id": ds_r["probe_id"],
        "domain": ds_r["domain"],
        "framing": ds_r.get("framing", ""),
        "role_line": ds_r.get("role_line", ""),
        "model_id": ds_r["model_id"],
        "ds": {"D1": ds_r["D1"], "D2": ds_r["D2"], "D3": ds_r["D3"],
               "BVI": ds_r["BVI"], "BVI_condition": ds_r.get("BVI_condition", "")},
        "cl": {"D1": cl_r["D1"], "D2": cl_r["D2"], "D3": cl_r["D3"],
               "BVI": cl_r["BVI"], "BVI_condition": cl_r.get("BVI_condition", "")},
    })
print(f"  Paired records: {len(pairs)}")

# [3] Verify BVI recomputation
print("\n[3] Verifying BVI recomputation...")
ds_bvi_mismatch = 0
cl_bvi_mismatch = 0
for p in pairs:
    if compute_bvi(p["ds"]["D1"], p["ds"]["D2"], p["ds"]["D3"]) != p["ds"]["BVI"]:
        ds_bvi_mismatch += 1
    if compute_bvi(p["cl"]["D1"], p["cl"]["D2"], p["cl"]["D3"]) != p["cl"]["BVI"]:
        cl_bvi_mismatch += 1
print(f"  DeepSeek BVI mismatches: {ds_bvi_mismatch}")
print(f"  Claude BVI mismatches:   {cl_bvi_mismatch}")

# [4] Overall BVI metrics
print("\n[4] Computing Overall BVI Cross-Judge Metrics...")
ds_bvi_all = [p["ds"]["BVI"] for p in pairs]
cl_bvi_all = [p["cl"]["BVI"] for p in pairs]
overall = compute_pair_stats(ds_bvi_all, cl_bvi_all, "DeepSeek vs Claude (All)")
print(f"  n={overall['n']}, Po={overall['agree_pct']}%, κ={overall['cohen_kappa']}, "
      f"r={overall['pearson_r']}, PABAK={overall['pabak']}")

# [5] Dimension-level analysis
print("\n[5] Computing D1/D2/D3 Dimension-Level Metrics...")
dim_results = {}
for dim in ["D1", "D2", "D3"]:
    ds_vals = [p["ds"][dim] for p in pairs]
    cl_vals = [p["cl"][dim] for p in pairs]
    stats = compute_pair_stats(ds_vals, cl_vals, f"DeepSeek vs Claude [{dim}]")
    dim_results[dim] = stats
    print(f"  {dim}: n={stats['n']}, Po={stats['agree_pct']}%, κ={stats['cohen_kappa']}, "
          f"r={stats['pearson_r']}, Wκ={stats['weighted_kappa']}")

# [6] Domain-level analysis
print("\n[6] Computing Domain-Level Metrics...")
domain_results = {}
for dom in ["CTL", "REG-S", "REG-H", "UNREG"]:
    sub = [p for p in pairs if p["domain"] == dom]
    ds_vals = [p["ds"]["BVI"] for p in sub]
    cl_vals = [p["cl"]["BVI"] for p in sub]
    stats = compute_pair_stats(ds_vals, cl_vals, f"DeepSeek vs Claude [{dom}]")
    domain_results[dom] = stats
    print(f"  {dom}: n={stats['n']}, Po={stats['agree_pct']}%, κ={stats['cohen_kappa']}, PABAK={stats['pabak']}")

# Domain x Dimension analysis
print("\n[6b] Computing Domain × Dimension Metrics...")
domain_dim_results = {}
for dom in ["CTL", "REG-S", "REG-H", "UNREG"]:
    domain_dim_results[dom] = {}
    sub = [p for p in pairs if p["domain"] == dom]
    for dim in ["D1", "D2", "D3"]:
        ds_vals = [p["ds"][dim] for p in sub]
        cl_vals = [p["cl"][dim] for p in sub]
        stats = compute_pair_stats(ds_vals, cl_vals, f"DS vs CL [{dom}][{dim}]")
        domain_dim_results[dom][dim] = stats

# [7] Model-level analysis
print("\n[7] Computing Model-Level Metrics...")
model_results = {}
for mid in sorted(set(p["model_id"] for p in pairs)):
    sub = [p for p in pairs if p["model_id"] == mid]
    ds_vals = [p["ds"]["BVI"] for p in sub]
    cl_vals = [p["cl"]["BVI"] for p in sub]
    stats = compute_pair_stats(ds_vals, cl_vals, f"DeepSeek vs Claude [{mid}]")
    model_results[mid] = stats
    print(f"  {mid}: n={stats['n']}, Po={stats['agree_pct']}%, κ={stats['cohen_kappa']}")

# [8] Anchor subset analysis (match with human scoring data)
print("\n[8] Computing Anchor Subset Metrics...")
with open(HUMAN_OUTPUT, 'r', encoding='utf-8') as f:
    human_data = json.load(f)

# Get anchor trial_ids from the anchor-set majority matches
anchor_tids = set()
if "majority_details" in human_data:
    for m in human_data["majority_details"]:
        # Build trial_id lookup from the pairs
        for p in pairs:
            if (p["probe_id"] == m["probe_id"] and p["model_id"] == m["model_id"]
                and p["framing"] == m["framing"] and p["role_line"] == m["role_line"]):
                anchor_tids.add(p["trial_id"])
                break

print(f"  Anchor trial_ids found: {len(anchor_tids)}")

anchor_pairs = [p for p in pairs if p["trial_id"] in anchor_tids]
print(f"  Anchor paired records: {len(anchor_pairs)}")

if len(anchor_pairs) >= 3:
    ds_anchor = [p["ds"]["BVI"] for p in anchor_pairs]
    cl_anchor = [p["cl"]["BVI"] for p in anchor_pairs]
    anchor_stats = compute_pair_stats(ds_anchor, cl_anchor, "DeepSeek vs Claude (Anchor)")
    print(f"  Anchor: n={anchor_stats['n']}, Po={anchor_stats['agree_pct']}%, "
          f"κ={anchor_stats['cohen_kappa']}, PABAK={anchor_stats['pabak']}")

    # Anchor dimension
    anchor_dim = {}
    for dim in ["D1", "D2", "D3"]:
        ds_vals = [p["ds"][dim] for p in anchor_pairs]
        cl_vals = [p["cl"][dim] for p in anchor_pairs]
        anchor_dim[dim] = compute_pair_stats(ds_vals, cl_vals, f"DS vs CL Anchor [{dim}]")
else:
    anchor_stats = {"error": "n<3", "n": len(anchor_pairs)}
    anchor_dim = {}

# [9] BVI condition comparison
print("\n[9] Computing BVI Condition Cross-Tabulation...")
cond_pairs = [(p["ds"]["BVI_condition"], p["cl"]["BVI_condition"]) for p in pairs]
cond_counts = Counter(cond_pairs)
print(f"  Top 10 condition pairs:")
for (ds_cond, cl_cond), cnt in cond_counts.most_common(10):
    print(f"    DS:{ds_cond} → CL:{cl_cond}: {cnt}")

# [10] BVI change analysis (DS->CL direction)
print("\n[10] Computing BVI Change Direction...")
bvi_changes = Counter()
for p in pairs:
    ds_b = p["ds"]["BVI"]
    cl_b = p["cl"]["BVI"]
    if ds_b == cl_b:
        bvi_changes["same"] += 1
    elif ds_b < cl_b:
        bvi_changes[f"DS{ds_b}→CL{cl_b}(up)"] += 1
    else:
        bvi_changes[f"DS{ds_b}→CL{cl_b}(down)"] += 1
for k, v in bvi_changes.most_common():
    print(f"    {k}: {v} ({v/len(pairs)*100:.1f}%)")

# [11] Framing and role_line breakdown
print("\n[11] Computing Framing & Role-Line Metrics...")
framing_results = {}
for frm in sorted(set(p["framing"] for p in pairs)):
    sub = [p for p in pairs if p["framing"] == frm]
    ds_vals = [p["ds"]["BVI"] for p in sub]
    cl_vals = [p["cl"]["BVI"] for p in sub]
    framing_results[frm] = compute_pair_stats(ds_vals, cl_vals, f"DS vs CL [{frm}]")

role_results = {}
for rl in sorted(set(p["role_line"] for p in pairs)):
    sub = [p for p in pairs if p["role_line"] == rl]
    ds_vals = [p["ds"]["BVI"] for p in sub]
    cl_vals = [p["cl"]["BVI"] for p in sub]
    role_results[rl] = compute_pair_stats(ds_vals, cl_vals, f"DS vs CL [{rl}]")

# ── Save output ──
output = {
    "metadata": {
        "ds_file": str(DS_PATH),
        "cl_file": str(CL_PATH),
        "ds_valid": len(ds_map),
        "cl_valid": len(cl_map),
        "overlapping": len(common_ids),
        "paired": len(pairs),
    },
    "overall_bvi": overall,
    "dimensions": dim_results,
    "domains": domain_results,
    "domain_dimensions": domain_dim_results,
    "models": model_results,
    "anchor_subset": anchor_stats,
    "anchor_dimensions": anchor_dim,
    "framing": framing_results,
    "role_line": role_results,
    "bvi_condition_crosstab": {f"DS:{dc}→CL:{cc}": cnt for (dc, cc), cnt in cond_counts.most_common(30)},
    "bvi_changes": dict(bvi_changes),
}

OUT_PATH = BASE / "results/cross_judge_deepseek_claude.json"
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)

print(f"\n[OK] Output saved to: {OUT_PATH}")
print(f"     Keys: {list(output.keys())}")
print("=" * 80)
