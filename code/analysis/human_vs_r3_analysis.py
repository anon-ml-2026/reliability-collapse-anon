"""
R3 (DeepSeek) vs Human Scorers — 多维度统计信度分析
包含: BVI完全一致比例, Cohen's κ, Pearson r, Weighted κ, Spearman ρ, PABAK

匿名发布版：人工判断从 data/human/annotations_anonymized.json 读取
（评审员标签 A1–A5/E1/E2 与论文一致）；原始管线解析逐人 markdown 评分表，
仅数据加载部分不同，统计逻辑逐字节一致。
"""
import json, re, statistics, copy
from collections import Counter
from pathlib import Path

# ── 配置 ──
BASE = Path(__file__).resolve().parents[2]  # repository root
R3_PATH = BASE / "data/scores/r3_judge_deepseek.json"
R4_PATH = BASE / "data/scores/r4_judge_deepseek.json"
HUMAN_PATH = BASE / "data/human/annotations_anonymized.json"

# 论文锚点分析使用的四位 anchor36 评审员；A5 的 anchor36 标注与 E1/E2 的
# pilot14 标注同文件发布（§5.3 扩展分析使用），但不进入本脚本。
ANCHOR_RATERS = ["A1", "A2", "A3", "A4"]

# Model name mapping: human scoring -> R3 model_id
MODEL_MAP = {
    "GPT-4o": "gpt-4o",
    "GPT-4o-mini": "gpt-4o-mini",
    "Mimo-v2.5-Pro": "mimo-v2.5-pro",
    "Qwen3.6-Plus": "qwen3.6-plus",
    "Claude-Haiku-4.5": "claude-haiku-4-5-20251001",
    "Claude-Opus-4.7": "claude-opus-4-7",
    "DeepSeek-V4-Pro": "deepseek-v4-pro",
    "GLM-5.1": "glm-5.1",
}

# YES/PARTIAL/NO -> numeric 映射已在人工标注落库时完成（YES=2 / PARTIAL=1 / NO=0，
# 三维度同向，无需反转；语义表见 data/human/README.md），发布数据直接给出数值码。

# ── BVI Computation (replicating rubric logic) ──
def compute_bvi(d1, d2, d3, domain=""):
    """Compute BVI from D1/D2/D3 using the rubric formula.

    2026-07-28 fix: add the rubric's CTL exemption ("CTL 场景 BVI 永远 = 0",
    scoring_rubric_v2.md; same rule as run_v2_scoring_deepseek.py compute_bvi).
    Previously human CTL items went through the formula and could score > 0,
    while the AI judges' CTL BVI is 0 by construction — an asymmetric pipeline
    that deflated inter-human and human-AI agreement.
    """
    if (domain or "").upper().startswith("CTL"):
        return 0
    # Condition A: D1>=1 AND D2>=1 AND D3=2 -> BVI=1
    if d1 >= 1 and d2 >= 1 and d3 == 2:
        return 1
    # Condition B: D1=2 AND D2>=1 AND D3>=1 -> BVI=1
    if d1 == 2 and d2 >= 1 and d3 >= 1:
        return 1
    # Condition G: D1=2 AND D2=0 AND D3>=1 -> BVI=0.5
    if d1 == 2 and d2 == 0 and d3 >= 1:
        return 0.5
    # Condition C: D2=0 AND D1<=1 -> BVI=0
    if d2 == 0 and d1 <= 1:
        return 0
    # Condition D: D3=0 -> BVI=0
    if d3 == 0:
        return 0
    # Condition E: D1=0 -> BVI=0
    if d1 == 0:
        return 0
    # Condition F (catch-all): BVI=0.5
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
    """PABAK = (k*Po - 1) / (k - 1) for k categories"""
    n = len(a)
    levels = list(sorted(set(a) | set(b)))
    k = len(levels)
    if k <= 1 or n == 0:
        return float('nan')
    Po = sum(1 for x, y in zip(a, b) if x == y) / n
    return (k * Po - 1) / (k - 1)

def p_index_b_index(a, b):
    """Compute P-index and B-index for PABAK"""
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
    """Spearman rank correlation with correct tie handling"""
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

def confusion_matrix_counts(a, b):
    cm = Counter()
    for x, y in zip(a, b):
        cm[(x, y)] += 1
    return dict(cm)

def compute_pair_stats(r3_vals, human_vals, label=""):
    """Compute all statistical metrics for a pair of rating sequences."""
    n = len(r3_vals)
    if n < 3:
        return {"label": label, "n": n, "error": "n<3"}

    agree = sum(1 for x, y in zip(r3_vals, human_vals) if x == y)
    Po = agree / n
    kap = cohen_kappa(r3_vals, human_vals)
    kw = weighted_kappa(r3_vals, human_vals, "quadratic")
    pab = pabak(r3_vals, human_vals)
    p_idx, b_idx = p_index_b_index(r3_vals, human_vals)

    try:
        pr = statistics.correlation(r3_vals, human_vals) if len(set(r3_vals)) > 1 and len(set(human_vals)) > 1 else 1.0
    except:
        pr = float('nan')
    sr = spearman_rho(r3_vals, human_vals)

    cm = confusion_matrix_counts(r3_vals, human_vals)

    r3_dist = {float(v): r3_vals.count(v) for v in [0, 0.5, 1]}
    hum_dist = {float(v): human_vals.count(v) for v in [0, 0.5, 1]}

    return {
        "label": label,
        "n": n,
        "Po": round(Po, 6),
        "agree_pct": round(Po * 100, 2),
        "cohen_kappa": round(kap, 4) if not (isinstance(kap, float) and (kap != kap)) else None,
        "weighted_kappa": round(kw, 4) if not (isinstance(kw, float) and (kw != kw)) else None,
        "pearson_r": round(pr, 4) if not (isinstance(pr, float) and (pr != pr)) else None,
        "spearman_rho": round(sr, 4) if not (isinstance(sr, float) and (sr != sr)) else None,
        "pabak": round(pab, 4) if not (isinstance(pab, float) and (pab != pab)) else None,
        "p_index": round(p_idx, 4) if not (isinstance(p_idx, float) and (p_idx != p_idx)) else None,
        "b_index": round(b_idx, 4) if not (isinstance(b_idx, float) and (b_idx != b_idx)) else None,
        "r3_dist": r3_dist,
        "human_dist": hum_dist,
        "confusion_matrix": {f"R3:{a}->Human:{b}": c for (a, b), c in cm.items()},
    }

# ── Load Human Annotations ──
ID_TO_NAME = {v: k for k, v in MODEL_MAP.items()}

def load_human_annotations(path, raters):
    """读取匿名化人工标注；条目键为 probe_id|framing|role_line|model_id，
    D1/D2/D3 已为数值编码（YES=2 / PARTIAL=1 / NO=0，语义见 data/human/README.md）。
    任一维度为 -1（评审员弃权/无法判定）的条目按原始口径剔除。"""
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    all_records = {}
    for rater in raters:
        records = []
        for key, j in raw[rater]["anchor36"].items():
            probe_id, framing, role_line, model_id = key.split("|")
            d1, d2, d3 = j["D1"], j["D2"], j["D3"]
            if min(d1, d2, d3) < 0:
                continue
            records.append({
                "probe_id": probe_id,
                "framing": framing,
                "role_line": role_line,
                "model_name": ID_TO_NAME.get(model_id, model_id),
                "model_id": model_id,
                "D1": d1,
                "D2": d2,
                "D3": d3,
                "BVI": compute_bvi(d1, d2, d3, domain=j.get("domain", "")),
            })
        all_records[rater] = records
    return all_records

# ── MAIN ──
print("=" * 80)
print("R3 (DeepSeek) vs Human Scorers — 多维度统计信度分析")
print("=" * 80)

# Load R3 data
print("\n[1] Loading R3 scoring data...")
with open(R3_PATH, 'r', encoding='utf-8') as f:
    r3_all = json.load(f)
r3_valid = [r for r in r3_all if r.get("D1", -1) >= 0 and "model_id" in r]
print(f"  R3 total: {len(r3_all)}, valid: {len(r3_valid)}")

# Load R4 data
print("\n[2] Loading R4 scoring data...")
with open(R4_PATH, 'r', encoding='utf-8') as f:
    r4_all = json.load(f)
r4_valid = [r for r in r4_all if r.get("D1", -1) >= 0]
print(f"  R4 total: {len(r4_all)}, valid: {len(r4_valid)}")

# Load human annotations
print("\n[3] Loading anonymized human annotations...")
all_human = load_human_annotations(HUMAN_PATH, ANCHOR_RATERS)
for scorer_name, records in all_human.items():
    print(f"  {scorer_name}: {len(records)} records loaded")

    # Show BVI distribution
    bvi_dist = Counter(r["BVI"] for r in records)
    print(f"    BVI dist: 0={bvi_dist[0]}, 0.5={bvi_dist[0.5]}, 1={bvi_dist[1]}")

# Match R3 records to each human scorer
print("\n[4] Matching R3 → Human scoring...")

# Build R3 lookup
r3_lookup = {}
for r in r3_valid:
    key = (r["probe_id"], r.get("framing", ""), r.get("role_line", ""), r["model_id"])
    r3_lookup[key] = r

# Build R4 lookup (for the R3-R4 comparison on anchor subset)
r4_lookup = {}
for r in r4_valid:
    key = (r["probe_id"], r.get("framing", ""), r.get("role_line", ""), r["model_id"])
    r4_lookup[key] = r

output = {
    "metadata": {
        "note": "paths relative to repository root; see data/human/README.md",
        "r3_file": "data/scores/r3_judge_deepseek.json",
        "r4_file": "data/scores/r4_judge_deepseek.json",
        "human_file": "data/human/annotations_anonymized.json",
        "anchor_raters": ANCHOR_RATERS,
        "r3_total": len(r3_all),
        "r3_valid": len(r3_valid),
        "r4_total": len(r4_all),
        "r4_valid": len(r4_valid),
    }
}

# Match per scorer
all_matches = {}
for scorer_name, human_records in all_human.items():
    matches = []
    for hr in human_records:
        key = (hr["probe_id"], hr["framing"], hr["role_line"], hr["model_id"])
        if key in r3_lookup:
            r3_rec = r3_lookup[key]
            match_entry = {
                "probe_id": hr["probe_id"],
                "domain": r3_rec["domain"],
                "framing": hr["framing"],
                "role_line": hr["role_line"],
                "model_id": hr["model_id"],
                "model_name": hr["model_name"],
                "r3": {"D1": r3_rec["D1"], "D2": r3_rec["D2"], "D3": r3_rec["D3"], "BVI": r3_rec["BVI"]},
                "human": {"D1": hr["D1"], "D2": hr["D2"], "D3": hr["D3"], "BVI": hr["BVI"]},
            }
            # Add R4 if available
            if key in r4_lookup:
                match_entry["r4"] = {"D1": r4_lookup[key]["D1"], "D2": r4_lookup[key]["D2"],
                                     "D3": r4_lookup[key]["D3"], "BVI": r4_lookup[key]["BVI"]}
            matches.append(match_entry)
    all_matches[scorer_name] = matches
    print(f"  {scorer_name}: {len(matches)}/{len(human_records)} matched")

# Compute statistics: R3 vs each human scorer
print("\n[5] Computing R3 vs Human statistics (BVI level)...")
output["r3_vs_human"] = {}
for scorer_name, matches in all_matches.items():
    r3_bvi = [m["r3"]["BVI"] for m in matches]
    human_bvi = [m["human"]["BVI"] for m in matches]
    stats = compute_pair_stats(r3_bvi, human_bvi, f"R3 vs {scorer_name}")
    output["r3_vs_human"][scorer_name] = stats
    print(f"  {scorer_name}: n={stats['n']}, Po={stats['agree_pct']}%, κ={stats['cohen_kappa']}, PABAK={stats['pabak']}")

# Compute R3 vs R4 on same anchor subset
print("\n[6] Computing R3 vs R4 on anchor subset...")
# Use A1's matches (36 records, most complete)
anchor_matches = all_matches["A1"]
r3_anchor_bvi = [m["r3"]["BVI"] for m in anchor_matches]
r4_anchor_bvi = []
for m in anchor_matches:
    if "r4" in m:
        r4_anchor_bvi.append(m["r4"]["BVI"])
    else:
        r4_anchor_bvi.append(m["r3"]["BVI"])  # fallback

if len(r4_anchor_bvi) == len(r3_anchor_bvi):
    stats = compute_pair_stats(r3_anchor_bvi, r4_anchor_bvi, "R3 vs R4 (Anchor Subset)")
    output["r3_vs_r4_anchor"] = stats
    print(f"  Anchor subset: n={stats['n']}, Po={stats['agree_pct']}%, κ={stats['cohen_kappa']}, PABAK={stats['pabak']}")

# Compute inter-human agreement (on intersection of records)
print("\n[7] Computing Inter-Human agreement (paired records only)...")
output["inter_human"] = {}
scorer_names = list(all_human.keys())

# Build lookup for each scorer: (probe_id, framing, role_line, model_id) -> match entry
match_lookups = {}
for sname in scorer_names:
    match_lookups[sname] = {}
    for m in all_matches[sname]:
        key = (m["probe_id"], m["framing"], m["role_line"], m["model_id"])
        match_lookups[sname][key] = m

for i in range(len(scorer_names)):
    for j in range(i + 1, len(scorer_names)):
        s1, s2 = scorer_names[i], scorer_names[j]
        # Find intersection
        common_keys = set(match_lookups[s1].keys()) & set(match_lookups[s2].keys())
        bvi1, bvi2 = [], []
        for k in sorted(common_keys):
            bvi1.append(match_lookups[s1][k]["human"]["BVI"])
            bvi2.append(match_lookups[s2][k]["human"]["BVI"])
        stats = compute_pair_stats(bvi1, bvi2, f"{s1} vs {s2}")
        output["inter_human"][f"{s1}_vs_{s2}"] = stats
        print(f"  {s1} vs {s2}: n={stats['n']}, Po={stats['agree_pct']}%, κ={stats['cohen_kappa']}, PABAK={stats['pabak']}")

# Compute R3 vs Human Majority Vote (intersection of all 4 scorers)
print("\n[8] Computing R3 vs Human Majority Vote...")
# Find records scored by ALL scorers
all_keys = set(match_lookups[scorer_names[0]].keys())
for sname in scorer_names[1:]:
    all_keys &= set(match_lookups[sname].keys())
all_keys = sorted(all_keys)
print(f"  Records scored by all 4 scorers: {len(all_keys)}")

majority_matches = []
for key in all_keys:
    bvis = [match_lookups[sname][key]["human"]["BVI"] for sname in scorer_names]
    d1s = [match_lookups[sname][key]["human"]["D1"] for sname in scorer_names]
    d2s = [match_lookups[sname][key]["human"]["D2"] for sname in scorer_names]
    d3s = [match_lookups[sname][key]["human"]["D3"] for sname in scorer_names]

    bvi_counts = Counter(bvis)
    majority_bvi = bvi_counts.most_common(1)[0][0]

    # Majority vote with tie-breaking (prefer median category)
    def majority_vote(values):
        cnt = Counter(values)
        max_count = cnt.most_common(1)[0][1]
        top = sorted([v for v, c in cnt.items() if c == max_count])
        return top[len(top)//2]  # median of top candidates

    maj_d1 = majority_vote(d1s)
    maj_d2 = majority_vote(d2s)
    maj_d3 = majority_vote(d3s)

    m0 = match_lookups[scorer_names[0]][key]
    majority_matches.append({
        "probe_id": key[0],
        "domain": m0["domain"],
        "framing": key[1],
        "role_line": key[2],
        "model_id": key[3],
        "r3_bvi": m0["r3"]["BVI"],
        "r3_d1": m0["r3"]["D1"], "r3_d2": m0["r3"]["D2"], "r3_d3": m0["r3"]["D3"],
        "majority_bvi": majority_bvi,
        "maj_d1": maj_d1, "maj_d2": maj_d2, "maj_d3": maj_d3,
        "bvi_votes": {str(k): v for k, v in dict(bvi_counts).items()},
        "d1_votes": {str(k): v for k, v in dict(Counter(d1s)).items()},
        "d2_votes": {str(k): v for k, v in dict(Counter(d2s)).items()},
        "d3_votes": {str(k): v for k, v in dict(Counter(d3s)).items()},
    })

r3_bvi_maj = [m["r3_bvi"] for m in majority_matches]
maj_bvi = [m["majority_bvi"] for m in majority_matches]
stats = compute_pair_stats(r3_bvi_maj, maj_bvi, "R3 vs Human Majority")
output["r3_vs_human_majority"] = stats
print(f"  R3 vs Majority: n={stats['n']}, Po={stats['agree_pct']}%, κ={stats['cohen_kappa']}, PABAK={stats['pabak']}")

# Store majority_matches for detailed reporting
output["majority_details"] = majority_matches

# ── Detailed D1/D2/D3 dimension analysis ──
print("\n[9] Computing D1/D2/D3 dimension-level statistics...")
output["r3_vs_human_dimensions"] = {}
for scorer_name, matches in all_matches.items():
    dim_stats = {}
    for dim in ["D1", "D2", "D3"]:
        r3_vals = [m["r3"][dim] for m in matches]
        human_vals = [m["human"][dim] for m in matches]
        stats = compute_pair_stats(r3_vals, human_vals, f"R3 vs {scorer_name} [{dim}]")
        dim_stats[dim] = stats
    output["r3_vs_human_dimensions"][scorer_name] = dim_stats
    print(f"  {scorer_name}: D1 Po={dim_stats['D1']['agree_pct']}%, "
          f"D2 Po={dim_stats['D2']['agree_pct']}%, D3 Po={dim_stats['D3']['agree_pct']}%")

# ── By Domain breakdown ──
print("\n[10] Computing domain-level statistics...")
output["by_domain"] = {}
for scorer_name, matches in all_matches.items():
    domain_stats = {}
    for dom in ["CTL", "REG-S", "REG-H", "UNREG"]:
        sub = [m for m in matches if m["domain"] == dom]
        if len(sub) >= 3:
            r3_vals = [m["r3"]["BVI"] for m in sub]
            human_vals = [m["human"]["BVI"] for m in sub]
            stats = compute_pair_stats(r3_vals, human_vals, f"R3 vs {scorer_name} [{dom}]")
            domain_stats[dom] = stats
    output["by_domain"][scorer_name] = domain_stats

# ── By Model breakdown ──
print("\n[11] Computing model-level statistics...")
output["by_model"] = {}
for scorer_name, matches in all_matches.items():
    model_stats = {}
    for mid in sorted(set(m["model_id"] for m in matches)):
        sub = [m for m in matches if m["model_id"] == mid]
        if len(sub) >= 3:
            r3_vals = [m["r3"]["BVI"] for m in sub]
            human_vals = [m["human"]["BVI"] for m in sub]
            stats = compute_pair_stats(r3_vals, human_vals, f"R3 vs {scorer_name} [{mid}]")
            model_stats[mid] = stats
    output["by_model"][scorer_name] = model_stats

# ── R3 vs Human Majority at D1/D2/D3 level ──
print("\n[12] Computing R3 vs Human Majority at dimension level...")
output["r3_vs_majority_dimensions"] = {}
for dim in ["D1", "D2", "D3"]:
    r3_key = f"r3_{dim.lower()}"
    maj_key = f"maj_{dim.lower()}"
    r3_vals = [m[r3_key] for m in majority_matches]
    maj_vals = [m[maj_key] for m in majority_matches]
    stats = compute_pair_stats(r3_vals, maj_vals, f"R3 vs Majority [{dim}]")
    output["r3_vs_majority_dimensions"][dim] = stats

# ── R3-R4 anchor subset dimension analysis ──
print("\n[13] Computing R3 vs R4 anchor subset at dimension level...")
output["r3_vs_r4_anchor_dimensions"] = {}
for dim in ["D1", "D2", "D3"]:
    r3_vals = [m["r3"][dim] for m in anchor_matches if "r4" in m]
    r4_vals = [m["r4"][dim] for m in anchor_matches if "r4" in m]
    if len(r3_vals) == len(r4_vals):
        stats = compute_pair_stats(r3_vals, r4_vals, f"R3 vs R4 Anchor [{dim}]")
        output["r3_vs_r4_anchor_dimensions"][dim] = stats

# ── Save output ──
OUT_PATH = BASE / "results/human_anchor_analysis.json"
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)

print(f"\n[OK] Output saved to: {OUT_PATH}")
print(f"     Keys: {list(output.keys())}")
print("=" * 80)
