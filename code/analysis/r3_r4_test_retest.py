"""
R3 vs R4 Test-Retest 全面统计分析
维度: 总体、域、模型、框架、角色、域×模型交叉、D1/D2/D3单维、BVI等级条件一致率
指标: BVI一致率、Cohen's κ、Weighted κ、Pearson r、Spearman ρ、混淆矩阵
"""
import json, statistics
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]  # repository root

# ── 加载数据 ──
r4_all = json.load(open(BASE / "data/scores/r4_judge_deepseek.json", encoding="utf-8"))
r3_all = json.load(open(BASE / "data/scores/r3_judge_deepseek.json", encoding="utf-8"))
r4 = [r for r in r4_all if r.get("D1", -1) >= 0]
r3 = [r for r in r3_all if r.get("D1", -1) >= 0]
cn_models = {"deepseek-v4-pro", "qwen3.6-plus", "glm-5.1", "mimo-v2.5-pro"}

# ── 匹配 ──
r4_lk = {}
for r in r4:
    k = "|".join([r["probe_id"], r.get("framing", ""), r.get("role_line", ""), r["model_id"]])
    r4_lk[k] = r

pairs = []
for r in r3:
    if r["model_id"] not in cn_models:
        continue
    k = "|".join([r["probe_id"], r.get("framing", ""), r.get("role_line", ""), r["model_id"]])
    if k in r4_lk:
        pairs.append({
            "r3": r, "r4": r4_lk[k],
            "domain": r["domain"], "model": r["model_id"],
            "framing": r.get("framing", ""), "role": r.get("role_line", "")
        })

n = len(pairs)
print(f"总匹配对: {n}")
print("=" * 85)

# ── 统计函数 ──
def cohen_kappa(r1, r2):
    levels = sorted(set(r1) | set(r2))
    nn = len(r1)
    obs = Counter()
    for a, b in zip(r1, r2):
        obs[(a, b)] += 1
    po = sum(obs[(l, l)] for l in levels) / nn
    m1 = Counter(r1); m2 = Counter(r2)
    pe = sum(m1[l] * m2[l] for l in levels) / (nn * nn)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def weighted_kappa(r1, r2, weight_type="quadratic"):
    levels = sorted(set(r1) | set(r2))
    nn = len(r1); n_lev = len(levels)
    if n_lev <= 1:
        return 1.0  # only one category → perfect agreement
    obs = Counter()
    for a, b in zip(r1, r2):
        obs[(a, b)] += 1
    w = {}
    for i, a in enumerate(levels):
        for j, b in enumerate(levels):
            w[(a, b)] = 1 - abs(i - j) / (n_lev - 1) if weight_type == "linear" else 1 - (i - j)**2 / (n_lev - 1)**2
    po = sum(obs[(a, b)] * w.get((a, b), 0) for a in levels for b in levels) / nn
    m1 = Counter(r1); m2 = Counter(r2)
    pe = 0
    for a in levels:
        for b in levels:
            pe += (m1[a] / nn) * (m2[b] / nn) * w.get((a, b), 0)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def spearman_r(a, b):
    def rank_vals(vals):
        s = sorted((v, i) for i, v in enumerate(vals))
        ranks = [0] * len(vals)
        i = 0
        while i < len(s):
            j = i
            while j < len(s) and s[j][0] == s[i][0]:
                j += 1
            avg = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                ranks[s[k][1]] = avg
            i = j
        return ranks
    ra = rank_vals(a); rb = rank_vals(b)
    try:
        return statistics.correlation(ra, rb)
    except statistics.StatisticsError:
        return 1.0 if len(set(a)) == 1 and len(set(b)) == 1 else 0.0

def analyze_subset(subset):
    if len(subset) < 10:
        return None
    r3v = [p["r3"]["BVI"] for p in subset]
    r4v = [p["r4"]["BVI"] for p in subset]
    agree = sum(1 for a, b in zip(r3v, r4v) if a == b)
    k = cohen_kappa(r3v, r4v)
    r_val = statistics.correlation(r3v, r4v) if len(set(r3v)) > 1 and len(set(r4v)) > 1 else 1.0
    rho = spearman_r(r3v, r4v)
    wk = weighted_kappa(r3v, r4v)
    r3_m = sum(r3v) / len(r3v)
    r4_m = sum(r4v) / len(r4v)
    cm = Counter()
    for a, b in zip(r3v, r4v):
        cm[(a, b)] += 1
    return {
        "n": len(subset), "agree_rate": agree / len(subset) * 100,
        "kappa": k, "pearson_r": r_val, "spearman_rho": rho,
        "weighted_kappa": wk, "r3_mean": r3_m, "r4_mean": r4_m,
        "confusion": dict(cm)
    }

def print_result(label, res, indent=2):
    if res is None:
        print(f"{' ' * indent}{label}: n<10, skipped")
        return
    pre = " " * indent
    print(f"{pre}{label} (n={res['n']})")
    print(f"{pre}  一致率={res['agree_rate']:.1f}%  "
          f"κ={res['kappa']:.3f}  r={res['pearson_r']:.3f}  "
          f"ρ={res['spearman_rho']:.3f}  κw={res['weighted_kappa']:.3f}")
    print(f"{pre}  R3 mean={res['r3_mean']:.3f}  R4 mean={res['r4_mean']:.3f}")
    cm = res["confusion"]
    print(f"{pre}  混淆(R3行→R4列): "
          f"0→0={cm.get((0,0),0)}  0→.5={cm.get((0,0.5),0)}  0→1={cm.get((0,1),0)}  "
          f".5→0={cm.get((0.5,0),0)}  .5→.5={cm.get((0.5,0.5),0)}  .5→1={cm.get((0.5,1),0)}  "
          f"1→0={cm.get((1,0),0)}  1→.5={cm.get((1,0.5),0)}  1→1={cm.get((1,1),0)}")


# ═══════════════════════════════════════════════════════
# 1. 总体
# ═══════════════════════════════════════════════════════
print("\n【1. 总体 Test-Retest 信度】")
r_all = analyze_subset(pairs)
print_result("Overall", r_all, indent=0)

# ═══════════════════════════════════════════════════════
# 2. 按域分层
# ═══════════════════════════════════════════════════════
print("\n【2. 按域分层 — 验证边界模糊性假说】")
print("  (假说预测: UNREG 一致率 < REG-H < REG-S < CTL)")
for d in ["CTL", "REG-S", "REG-H", "UNREG"]:
    sub = [p for p in pairs if p["domain"] == d]
    print_result(d, analyze_subset(sub))

# ═══════════════════════════════════════════════════════
# 3. 按模型分层
# ═══════════════════════════════════════════════════════
print("\n【3. 按模型分层 — 跨模型信度稳定性】")
for m in sorted(cn_models):
    sub = [p for p in pairs if p["model"] == m]
    print_result(m, analyze_subset(sub))

# ═══════════════════════════════════════════════════════
# 4. 按语用框架分层
# ═══════════════════════════════════════════════════════
print("\n【4. 按语用框架分层 — 查询方式对评分稳定性的影响】")
for f in ["F-Expert", "F-Friend", "F-Tool"]:
    sub = [p for p in pairs if p["framing"] == f]
    print_result(f, analyze_subset(sub))

# ═══════════════════════════════════════════════════════
# 5. 按角色线分层
# ═══════════════════════════════════════════════════════
print("\n【5. 按角色线分层 — 系统提示对评分稳定性的影响】")
for rl in ["no-role", "with-role", "boundary-role"]:
    sub = [p for p in pairs if p["role"] == rl]
    print_result(rl, analyze_subset(sub))

# ═══════════════════════════════════════════════════════
# 6. 域 × 模型 交叉
# ═══════════════════════════════════════════════════════
print("\n【6. 域 × 模型 交叉 — 识别特定模型-域组合的信度热点】")
for d in ["CTL", "REG-S", "REG-H", "UNREG"]:
    print(f"  {d}:")
    for m in sorted(cn_models):
        sub = [p for p in pairs if p["domain"] == d and p["model"] == m]
        if len(sub) >= 10:
            res = analyze_subset(sub)
            if res:
                print(f"    {m}: agree={res['agree_rate']:.1f}%  "
                      f"κ={res['kappa']:.3f}  r={res['pearson_r']:.3f}  n={res['n']}")

# ═══════════════════════════════════════════════════════
# 7. D1/D2/D3 单维度 Test-Retest
# ═══════════════════════════════════════════════════════
print("\n【7. D1/D2/D3 单维度 Test-Retest — 哪个维度最不稳定？】")
print("  (假说预测: D1建议特异性最主观 → 信度最低; D3转介行为最客观 → 信度最高)")
for dim in ["D1", "D2", "D3"]:
    r3v = [p["r3"][dim] for p in pairs]
    r4v = [p["r4"][dim] for p in pairs]
    agree = sum(1 for a, b in zip(r3v, r4v) if a == b)
    k = cohen_kappa(r3v, r4v)
    r_val = statistics.correlation(r3v, r4v) if len(set(r3v)) > 1 and len(set(r4v)) > 1 else 1.0
    wk = weighted_kappa(r3v, r4v)
    rho = spearman_r(r3v, r4v)
    r3_m = sum(r3v) / len(r3v)
    r4_m = sum(r4v) / len(r4v)
    print(f"  {dim}: agree={agree/len(pairs)*100:.1f}%  κ={k:.3f}  r={r_val:.3f}  "
          f"ρ={rho:.3f}  κw={wk:.3f}  R3μ={r3_m:.3f}  R4μ={r4_m:.3f}")

# D1/D2/D3 by domain
print("\n  ── D1/D2/D3 × 域 ──")
for dim in ["D1", "D2", "D3"]:
    print(f"  {dim}:")
    for d in ["CTL", "REG-S", "REG-H", "UNREG"]:
        sub = [p for p in pairs if p["domain"] == d]
        if len(sub) < 10:
            continue
        r3v = [p["r3"][dim] for p in sub]
        r4v = [p["r4"][dim] for p in sub]
        agree = sum(1 for a, b in zip(r3v, r4v) if a == b)
        k = cohen_kappa(r3v, r4v)
        print(f"    {d}: agree={agree/len(sub)*100:.1f}%  κ={k:.3f}")

# ═══════════════════════════════════════════════════════
# 8. BVI 等级条件一致率
# ═══════════════════════════════════════════════════════
print("\n【8. 条件一致率 — 给定R3 BVI等级，R4如何变化？】")
for bvi_level in [0, 0.5, 1]:
    sub = [p for p in pairs if p["r3"]["BVI"] == bvi_level]
    if sub:
        match = sum(1 for p in sub if p["r4"]["BVI"] == bvi_level)
        up = sum(1 for p in sub if p["r4"]["BVI"] > bvi_level)
        down = sum(1 for p in sub if p["r4"]["BVI"] < bvi_level)
        print(f"  P(R4|R3=BVI={bvi_level}): 一致={match/len(sub)*100:.1f}%  "
              f"升级={up/len(sub)*100:.1f}%  降级={down/len(sub)*100:.1f}%  n={len(sub)}")

# ═══════════════════════════════════════════════════════
# 9. BVI 条件一致率 × 域
# ═══════════════════════════════════════════════════════
print("\n【9. 条件一致率 × 域 — 分歧的方向性】")
for bvi_level in [0, 1]:
    print(f"  R3 BVI={bvi_level}:")
    for d in ["CTL", "REG-S", "REG-H", "UNREG"]:
        sub = [p for p in pairs if p["r3"]["BVI"] == bvi_level and p["domain"] == d]
        if len(sub) < 5:
            continue
        match = sum(1 for p in sub if p["r4"]["BVI"] == bvi_level)
        change = sum(1 for p in sub if p["r4"]["BVI"] != bvi_level)
        direction = "↑" if bvi_level == 0 else "↓"
        print(f"    {d}: 稳定={match/len(sub)*100:.1f}%  "
              f"变化={change/len(sub)*100:.1f}%({direction})  n={len(sub)}")

# ═══════════════════════════════════════════════════════
# 10. 汇总表
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 85)
print("【汇总】R3-R4 Test-Retest 信度矩阵")
print("=" * 85)
print(f"{'维度':<30} {'n':>5} {'一致率%':>8} {'κ':>7} {'r':>7} {'κw':>7}")
print("-" * 70)
for label, sub in [
    ("总体", pairs),
    ("CTL", [p for p in pairs if p["domain"] == "CTL"]),
    ("REG-S", [p for p in pairs if p["domain"] == "REG-S"]),
    ("REG-H", [p for p in pairs if p["domain"] == "REG-H"]),
    ("UNREG", [p for p in pairs if p["domain"] == "UNREG"]),
    ("deepseek-v4-pro", [p for p in pairs if p["model"] == "deepseek-v4-pro"]),
    ("qwen3.6-plus", [p for p in pairs if p["model"] == "qwen3.6-plus"]),
    ("glm-5.1", [p for p in pairs if p["model"] == "glm-5.1"]),
    ("mimo-v2.5-pro", [p for p in pairs if p["model"] == "mimo-v2.5-pro"]),
    ("F-Expert", [p for p in pairs if p["framing"] == "F-Expert"]),
    ("F-Friend", [p for p in pairs if p["framing"] == "F-Friend"]),
    ("F-Tool", [p for p in pairs if p["framing"] == "F-Tool"]),
    ("no-role", [p for p in pairs if p["role"] == "no-role"]),
    ("with-role", [p for p in pairs if p["role"] == "with-role"]),
    ("boundary-role", [p for p in pairs if p["role"] == "boundary-role"]),
]:
    res = analyze_subset(sub)
    if res:
        print(f"{label:<30} {res['n']:>5} {res['agree_rate']:>7.1f}% {res['kappa']:>7.3f} {res['pearson_r']:>7.3f} {res['weighted_kappa']:>7.3f}")

print("\nκ 解读: <0=随机以下  0-0.20=轻微  0.21-0.40=尚可  0.41-0.60=中等  0.61-0.80=高度  0.81-1.00=几乎完美")
print("完成")
