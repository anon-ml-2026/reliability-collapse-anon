"""
R3 vs R4 Test-Retest 全面统计分析（含 PABAK）
输出 JSON 供 MD 报告生成使用
"""
import json, statistics, sys
from collections import Counter
from pathlib import Path

# ── 数据路径 ──
BASE = Path(__file__).resolve().parents[2]  # repository root
R3_PATH = BASE / "data/scores/r3_judge_deepseek.json"
R4_PATH = BASE / "data/scores/r4_judge_deepseek.json"
OUT_PATH = BASE / "results/test_retest_stats.json"

print("=" * 80)
print("R3 vs R4 Test-Retest 统计分析（含 PABAK）")
print("=" * 80)

# ── Step 1: 数据加载 ──
print("\n[Step 1] 加载数据...")
r4_all = json.load(open(R4_PATH, encoding="utf-8"))
r3_all = json.load(open(R3_PATH, encoding="utf-8"))
print(f"  R4 原始: {len(r4_all)} records")
print(f"  R3 原始: {len(r3_all)} records")

r4_valid = [r for r in r4_all if r.get("D1", -1) >= 0]
r3_valid = [r for r in r3_all if r.get("D1", -1) >= 0]
print(f"  R4 有效(D1>=0): {len(r4_valid)}, 排除: {len(r4_all) - len(r4_valid)}")
print(f"  R3 有效(D1>=0): {len(r3_valid)}, 排除: {len(r3_all) - len(r3_valid)}")

# ── Step 2: 匹配 ──
print("\n[Step 2] 匹配 R3-R4 pairs...")
cn_models = {"deepseek-v4-pro", "qwen3.6-plus", "glm-5.1", "mimo-v2.5-pro"}
r4_lk = {}
for r in r4_valid:
    k = "|".join([r["probe_id"], r.get("framing", ""), r.get("role_line", ""), r["model_id"]])
    r4_lk[k] = r

pairs = []
unmatched = 0
for r in r3_valid:
    if r["model_id"] not in cn_models:
        continue
    k = "|".join([r["probe_id"], r.get("framing", ""), r.get("role_line", ""), r["model_id"]])
    if k in r4_lk:
        pairs.append({
            "trial_id": r["trial_id"],
            "probe_id": r["probe_id"],
            "domain": r["domain"],
            "model": r["model_id"],
            "framing": r.get("framing", ""),
            "role": r.get("role_line", ""),
            "r3": {"D1": r["D1"], "D2": r["D2"], "D3": r["D3"], "BVI": r["BVI"]},
            "r4": {"D1": r4_lk[k]["D1"], "D2": r4_lk[k]["D2"], "D3": r4_lk[k]["D3"], "BVI": r4_lk[k]["BVI"]},
        })
    else:
        unmatched += 1

print(f"  匹配成功: {len(pairs)}, 未匹配: {unmatched}")
print(f"  匹配率: {len(pairs)/(len(pairs)+unmatched)*100:.1f}%")

# ── Helper Functions ──
def cohen_kappa(a, b):
    """Cohen's Kappa: (Po - Pe) / (1 - Pe)"""
    levels = sorted(set(a) | set(b))
    n = len(a)
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
    """Weighted Kappa (quadratic = ICC equivalent)"""
    levels = sorted(set(a) | set(b))
    n, n_lev = len(a), len(levels)
    if n_lev <= 1:
        return 1.0
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
    """
    PABAK: Prevalence-Adjusted Bias-Adjusted Kappa (Byrt et al., 1993)

    公式推导:
    1. 原始 Cohen's κ 受两个因素影响:
       - Prevalence (P-index): 类别分布不均 (如大部分评分是 BVI=0)
       - Bias (B-index): 两轮评分系统性偏差 (如 R4 系统性高于 R3)

    2. PABAK 通过假设各类别均匀分布 + 两轮无系统偏差来修正:
       - 对 k 个类别, 每类观测比例调整为 1/k
       - Po 不变 (实际一致率)
       - Pe_adjusted = 1/k (均匀分布假设)
       - PABAK = (Po - 1/k) / (1 - 1/k) = (k*Po - 1) / (k - 1)

    3. 对于 3 类别 (BVI: 0, 0.5, 1):
       - PABAK = (3*Po - 1) / 2
    """
    n = len(a)
    levels = list(sorted(set(a) | set(b)))
    k = len(levels)
    if k <= 1:
        return 1.0
    Po = sum(1 for x, y in zip(a, b) if x == y) / n
    # PABAK formula for k categories
    pabak_val = (k * Po - 1) / (k - 1)
    return pabak_val

def pabak_index(a, b):
    """返回 P-index (prevalence) 和 B-index (bias)"""
    n = len(a)
    levels = list(sorted(set(a) | set(b)))
    k = len(levels)
    if k <= 1:
        return 1.0, 0.0
    # P-index: how far from uniform
    all_ratings = a + b
    total = len(all_ratings)
    props = [all_ratings.count(l) / total for l in levels]
    P_index = sum(abs(p - 1/k) for p in props) / (2 * (1 - 1/k))
    # B-index: systematic difference between the two sets of ratings
    # Using the Byrt formula: B_index = |(b-c)| / N for binary, generalized
    diff_sum = 0
    for l in levels:
        diff_sum += abs(a.count(l) - b.count(l)) / n
    B_index = diff_sum / (2 * (1 - 1/k))
    return P_index, B_index

def spearman_rho(a, b):
    """Spearman rank correlation (Pearson r on ranks, correct for ties)"""
    import statistics
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
        return 1.0 if len(set(a)) == 1 and len(set(b)) == 1 else 0.0

def confusion_counts(a, b):
    """混淆矩阵计数"""
    cm = Counter()
    for x, y in zip(a, b):
        cm[(x, y)] += 1
    return dict(cm)

def compute_bvi_stats(subset_pairs, label):
    """对一个子集计算全套 BVI 统计指标"""
    if len(subset_pairs) < 10:
        return {"label": label, "n": len(subset_pairs), "error": "n<10"}

    r3_bvi = [p["r3"]["BVI"] for p in subset_pairs]
    r4_bvi = [p["r4"]["BVI"] for p in subset_pairs]
    n = len(subset_pairs)

    # 基础计数
    agree_exact = sum(1 for a, b in zip(r3_bvi, r4_bvi) if a == b)
    Po = agree_exact / n

    # Kappa
    k = cohen_kappa(r3_bvi, r4_bvi)
    wk = weighted_kappa(r3_bvi, r4_bvi)
    pabak_val = pabak(r3_bvi, r4_bvi)
    p_idx, b_idx = pabak_index(r3_bvi, r4_bvi)

    # Correlations
    try:
        pr = statistics.correlation(r3_bvi, r4_bvi) if len(set(r3_bvi)) > 1 and len(set(r4_bvi)) > 1 else 1.0
    except:
        pr = 1.0
    sr = spearman_rho(r3_bvi, r4_bvi)

    # Confusion
    cm = confusion_counts(r3_bvi, r4_bvi)

    # Means
    r3_mean = sum(r3_bvi) / n
    r4_mean = sum(r4_bvi) / n

    # Distributions
    r3_dist = {float(v): r3_bvi.count(v) for v in [0, 0.5, 1]}
    r4_dist = {float(v): r4_bvi.count(v) for v in [0, 0.5, 1]}

    # Direction
    up = sum(1 for a, b in zip(r3_bvi, r4_bvi) if b > a)
    down = sum(1 for a, b in zip(r3_bvi, r4_bvi) if b < a)
    same = sum(1 for a, b in zip(r3_bvi, r4_bvi) if b == a)

    # Conditional agreement
    cond_agree = {}
    for level in [0, 0.5, 1]:
        sub = [(a, b) for a, b in zip(r3_bvi, r4_bvi) if a == level]
        if sub:
            match = sum(1 for a, b in sub if b == level)
            up_c = sum(1 for a, b in sub if b > level)
            down_c = sum(1 for a, b in sub if b < level)
            cond_agree[level] = {
                "n": len(sub),
                "stable_pct": round(match / len(sub) * 100, 2),
                "up_pct": round(up_c / len(sub) * 100, 2),
                "down_pct": round(down_c / len(sub) * 100, 2),
            }

    return {
        "label": label,
        "n": n,
        "Po": round(Po, 6),
        "agree_exact": agree_exact,
        "agree_pct": round(Po * 100, 2),
        "cohen_kappa": round(k, 4),
        "weighted_kappa": round(wk, 4),
        "pabak": round(pabak_val, 4),
        "pabak_p_index": round(p_idx, 4),
        "pabak_b_index": round(b_idx, 4),
        "pearson_r": round(pr, 4),
        "spearman_rho": round(sr, 4),
        "r3_mean": round(r3_mean, 4),
        "r4_mean": round(r4_mean, 4),
        "r3_dist": r3_dist,
        "r4_dist": r4_dist,
        "confusion_matrix": {f"{a}->{b}": c for (a, b), c in cm.items()},
        "direction": {"up": up, "down": down, "same": same},
        "conditional_agreement": cond_agree,
    }


# ── Step 3-11: 逐维度分析 ──
output = {}
output["metadata"] = {
    "total_pairs": len(pairs),
    "r3_file": R3_PATH,
    "r4_file": R4_PATH,
    "scorer": "DeepSeek-V4-Pro",
    "cn_models": sorted(list(cn_models)),
    "r3_total_scored": len(r3_valid),
    "r4_total_scored": len(r4_valid),
    "timestamp": __import__("datetime").datetime.now().isoformat(),
}

results_summary = []

# Step 3: Overall
print("\n[Step 3] 总体 test-retest...")
res = compute_bvi_stats(pairs, "Overall (总体)")
output["overall"] = res
results_summary.append(res)
print(f"  Po={res['Po']:.4f}, κ={res['cohen_kappa']:.4f}, PABAK={res['pabak']:.4f}")

# Step 4: By Domain
print("\n[Step 4] 按域分层...")
output["by_domain"] = {}
for d in ["CTL", "REG-S", "REG-H", "UNREG"]:
    sub = [p for p in pairs if p["domain"] == d]
    output["by_domain"][d] = compute_bvi_stats(sub, d)
    print(f"  {d}: Po={output['by_domain'][d]['Po']:.4f}, κ={output['by_domain'][d]['cohen_kappa']:.4f}")

# Step 5: By Model
print("\n[Step 5] 按模型分层...")
output["by_model"] = {}
for m in sorted(cn_models):
    sub = [p for p in pairs if p["model"] == m]
    output["by_model"][m] = compute_bvi_stats(sub, m)
    print(f"  {m}: Po={output['by_model'][m]['Po']:.4f}, κ={output['by_model'][m]['cohen_kappa']:.4f}")

# Step 6: By D1/D2/D3
print("\n[Step 6] D1/D2/D3 单维...")
for dim in ["D1", "D2", "D3"]:
    sub_pairs_d = [p for p in pairs]  # full set for agreement
    r3v = [p["r3"][dim] for p in pairs]
    r4v = [p["r4"][dim] for p in pairs]
    agree = sum(1 for a, b in zip(r3v, r4v) if a == b)
    k = cohen_kappa(r3v, r4v)
    pab = pabak(r3v, r4v)
    wk = weighted_kappa(r3v, r4v)
    sr = spearman_rho(r3v, r4v)
    r3_m = sum(r3v) / len(r3v)
    r4_m = sum(r4v) / len(r4v)
    output[f"dimension_{dim}"] = {
        "label": f"D{dim} 维度",
        "n": len(pairs),
        "agree_pct": round(agree/len(pairs)*100, 2),
        "cohen_kappa": round(k, 4),
        "pabak": round(pab, 4),
        "weighted_kappa": round(wk, 4),
        "spearman_rho": round(sr, 4),
        "r3_mean": round(r3_m, 4),
        "r4_mean": round(r4_m, 4),
        "r3_dist": {float(v): r3v.count(v) for v in [0, 1, 2]},
        "r4_dist": {float(v): r4v.count(v) for v in [0, 1, 2]},
    }
    # by domain
    for d in ["CTL", "REG-S", "REG-H", "UNREG"]:
        sub = [p for p in pairs if p["domain"] == d]
        if len(sub) < 10:
            continue
        a = sum(1 for p in sub if p["r3"][dim] == p["r4"][dim])
        kd = cohen_kappa([p["r3"][dim] for p in sub], [p["r4"][dim] for p in sub])
        pabd = pabak([p["r3"][dim] for p in sub], [p["r4"][dim] for p in sub])
        output[f"dimension_{dim}"][f"domain_{d}"] = {
            "n": len(sub), "agree_pct": round(a/len(sub)*100, 2),
            "cohen_kappa": round(kd, 4), "pabak": round(pabd, 4),
        }

# Step 7: By Framing
print("\n[Step 7] 按语用框架...")
output["by_framing"] = {}
for f in ["F-Expert", "F-Friend", "F-Tool"]:
    sub = [p for p in pairs if p["framing"] == f]
    output["by_framing"][f] = compute_bvi_stats(sub, f)

# Step 8: By Role
print("\n[Step 8] 按角色线...")
output["by_role"] = {}
for rl in ["no-role", "with-role", "boundary-role"]:
    sub = [p for p in pairs if p["role"] == rl]
    output["by_role"][rl] = compute_bvi_stats(sub, rl)

# Step 9: Domain × Model
print("\n[Step 9] 域 × 模型交叉...")
output["domain_x_model"] = {}
for d in ["CTL", "REG-S", "REG-H", "UNREG"]:
    output["domain_x_model"][d] = {}
    for m in sorted(cn_models):
        sub = [p for p in pairs if p["domain"] == d and p["model"] == m]
        if len(sub) >= 10:
            output["domain_x_model"][d][m] = compute_bvi_stats(sub, f"{d}×{m}")

# Step 10: Conditional agreement (BVI level)
print("\n[Step 10] BVI 等级条件一致率...")
output["conditional_bvi"] = {}
for level in [0, 0.5, 1]:
    sub = [p for p in pairs if p["r3"]["BVI"] == level]
    if sub:
        match = sum(1 for p in sub if p["r4"]["BVI"] == level)
        up = sum(1 for p in sub if p["r4"]["BVI"] > level)
        down = sum(1 for p in sub if p["r4"]["BVI"] < level)
        output["conditional_bvi"][f"R3_BVI={level}"] = {
            "n": len(sub),
            "stable_pct": round(match/len(sub)*100, 2),
            "up_pct": round(up/len(sub)*100, 2),
            "down_pct": round(down/len(sub)*100, 2),
        }
        # By domain
        for d in ["REG-S", "REG-H", "UNREG"]:
            subd = [p for p in sub if p["domain"] == d]
            if len(subd) >= 5:
                md = sum(1 for p in subd if p["r4"]["BVI"] == level)
                ch = sum(1 for p in subd if p["r4"]["BVI"] != level)
                output["conditional_bvi"][f"R3_BVI={level}"][f"domain_{d}"] = {
                    "n": len(subd), "stable_pct": round(md/len(subd)*100, 2),
                    "change_pct": round(ch/len(subd)*100, 2),
                }

# Step 11: PABAK 专题分析
print("\n[Step 11] PABAK 专题分析...")
pabak_analysis = {}
pabak_analysis["formula"] = "PABAK = (k * Po - 1) / (k - 1)"
pabak_analysis["k_categories"] = 3
pabak_analysis["formula_for_3cat"] = "PABAK = (3 * Po - 1) / 2"
pabak_analysis["interpretation"] = (
    "PABAK 修正了原始 Cohen's κ 的两个系统性偏差: "
    "(1) Prevalence → P-index: 类别分布不均衡 (如 CTL 域 100% BVI=0); "
    "(2) Bias → B-index: 两轮评分间系统性差异 (如 R4 系统性更保守)。"
    "PABAK 假设各类别均匀分布且无系统偏差, 因此反映了'除了分布不均衡和系统偏差之外的'一致率。"
    "当 PABAK >> κ 时, 说明低 κ 主要由 prevalence/bias 驱动, 而非评分员真正不一致。"
)
pabak_analysis["overall"] = {
    "Po": output["overall"]["Po"],
    "kappa": output["overall"]["cohen_kappa"],
    "pabak": output["overall"]["pabak"],
    "p_index": output["overall"]["pabak_p_index"],
    "b_index": output["overall"]["pabak_b_index"],
    "gap_kappa_to_pabak": round(output["overall"]["pabak"] - output["overall"]["cohen_kappa"], 4),
}
pabak_analysis["by_domain"] = {}
for d in ["CTL", "REG-S", "REG-H", "UNREG"]:
    po = output["by_domain"][d]["Po"]
    pab = output["by_domain"][d]["pabak"]
    kap = output["by_domain"][d]["cohen_kappa"]
    pi = output["by_domain"][d]["pabak_p_index"]
    bi = output["by_domain"][d]["pabak_b_index"]
    pabak_analysis["by_domain"][d] = {
        "Po": po, "kappa": kap, "pabak": pab,
        "p_index": pi, "b_index": bi,
        "gap": round(pab - kap, 4),
    }
output["pabak_analysis"] = pabak_analysis

# ── 保存 JSON ──
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n[完成] 输出: {OUT_PATH}")
print(f"      {len(output)} top-level keys")
print("=" * 80)
