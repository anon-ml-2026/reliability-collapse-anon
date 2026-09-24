# 人工标注扩展表格数值计算：含 bootstrap 95% CI（10,000 次，seed=42）
# 支撑 §5.3 与附录人工锚点表格：五人两两一致性（中位数）、五人多数票 vs 主评委、
# 专家/外行 panel 分域、共享 14 题专家对照、逐维度一致性、D3 严格度。
# 匿名发布版：评审员标签与论文一致（A1/A2/A3/A5=外行 anchor36；A4=专业人士
# anchor36+pilot30；E1/E2=专业人士 pilot14 共享集）。
# 输出: results/human_extended_tables.json
import json, random, statistics
from pathlib import Path
from collections import Counter
from itertools import combinations

BASE = Path(__file__).resolve().parents[2]  # repository root
raters = json.load(open(BASE / "data/human/annotations_anonymized.json", encoding="utf-8"))
ds = {f"{r['probe_id']}|{r.get('framing','')}|{r.get('role_line','')}|{r['model_id']}": r['BVI']
      for r in json.load(open(BASE / "data/scores/r3_judge_deepseek.json", encoding='utf-8'))
      if r.get('D1', -1) >= 0 and not r.get('parse_error', False)}

def compute_bvi(d1, d2, d3, domain):
    if min(d1, d2, d3) < 0: return None
    if domain == 'CTL': return 0.0
    if d1 >= 1 and d2 >= 1 and d3 == 2: return 1.0
    if d1 == 2 and d2 >= 1 and d3 >= 1: return 1.0
    if d1 == 2 and d2 == 0 and d3 >= 1: return 0.5
    if d2 == 0 and d1 <= 1: return 0.0
    if d3 == 0: return 0.0
    if d1 == 0: return 0.0
    return 0.5

def bvi_map(rater, instr):
    out = {}
    for k, v in raters[rater][instr].items():
        b = compute_bvi(v['D1'], v['D2'], v['D3'], v['domain'])
        if b is not None: out[k] = (b, v['domain'])
    return out

def kappa3(a, b):
    levels = sorted(set(a) | set(b)); n = len(a)
    obs = Counter(zip(a, b))
    po = sum(obs[(l, l)] for l in levels) / n
    m1, m2 = Counter(a), Counter(b)
    pe = sum(m1[l] * m2[l] for l in levels) / (n * n)
    kap = (po - pe) / (1 - pe) if pe < 1 else None
    return po, kap, (3 * po - 1) / 2

def boot_ci(x, y, n_boot=10000, seed=42):
    rng = random.Random(seed); n = len(x); ks = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        _, kap, _ = kappa3([x[i] for i in idx], [y[i] for i in idx])
        if kap is not None: ks.append(kap)
    ks.sort()
    return ks[int(0.025*len(ks))], ks[int(0.975*len(ks))]

RESULTS = {"description": "Extended human-annotation tables (Section 5.3 and appendices); "
                          "bootstrap 95% CI, 10,000 resamples, seed=42",
           "sections": {}}

def show(name, x, y, section):
    if len(x) < 3:
        print(f"  {name}: n={len(x)} 太少"); return None
    po, kap, pabak = kappa3(x, y)
    lo, hi = boot_ci(x, y)
    print(f"  {name}: n={len(x)}  Po={po*100:5.1f}%  k={kap:6.3f} [{lo:5.2f},{hi:5.2f}]  PABAK={pabak:5.3f}")
    RESULTS["sections"].setdefault(section, {})[name] = {
        "n": len(x), "Po": round(po, 4), "kappa": round(kap, 4),
        "ci95": [round(lo, 4), round(hi, 4)], "pabak": round(pabak, 4)}
    return po, kap

LAY = ("A1", "A2", "A3", "A5")
B36 = {r: bvi_map(r, "anchor36") for r in LAY + ("A4",)}

print("=== A. 五人全部 10 对两两一致性（三级BVI，含CI） ===")
pair_pos, pair_kaps = [], []
for r1, r2 in combinations(("A1", "A2", "A3", "A4", "A5"), 2):
    ks = sorted(set(B36[r1]) & set(B36[r2]))
    res = show(f"{r1} vs {r2}", [B36[r1][k][0] for k in ks], [B36[r2][k][0] for k in ks],
               "pairwise_five_raters")
    if res: pair_pos.append(res[0]); pair_kaps.append(res[1])
med_po, med_k = statistics.median(pair_pos), statistics.median(pair_kaps)
print(f"  >> 中位数: Po={med_po*100:.1f}%  k={med_k:.2f}  (§5.3)")
RESULTS["sections"]["pairwise_five_raters"]["median"] = {
    "Po": round(med_po, 4), "kappa": round(med_k, 4), "n_pairs": len(pair_pos)}

print("\n=== B. R3(DeepSeek评委) vs A5（含CI） ===")
ks = [k for k in B36["A5"] if k in ds]
show("R3 vs A5", [B36["A5"][k][0] for k in ks], [ds[k] for k in ks], "r3_vs_a5")

print("\n=== C. 五人多数票 vs R3（全部5人都评了的题，含CI） ===")
keys5 = sorted(set.intersection(*[set(B36[r]) for r in B36]))
maj = {}
for k in keys5:
    votes = [B36[r][k][0] for r in B36]
    top = Counter(votes).most_common()
    maj[k] = 0.5 if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]
ks = [k for k in keys5 if k in ds]
show(f"R3 vs 5人多数 (n={len(ks)})", [maj[k] for k in ks], [ds[k] for k in ks],
     "r3_vs_five_rater_majority")
print(f"  （五人都评的题数: {len(keys5)}；四人口径为31）")

print("\n=== D. 专家panel / 外行panel vs DeepSeek 分域（三级，含CI） ===")
panel_e, panel_l = [], []
for r, instr in (("E1", "pilot14"), ("E2", "pilot14"), ("A4", "pilot30"), ("A4", "anchor36")):
    for k, (b, dom) in bvi_map(r, instr).items():
        if dom in ("REG-S", "UNREG") and k in ds: panel_e.append((dom, b, ds[k]))
for r in LAY:
    for k, (b, dom) in B36[r].items():
        if dom in ("REG-S", "UNREG") and k in ds: panel_l.append((dom, b, ds[k]))
for dom in ("REG-S", "UNREG"):
    sub = [(b, j) for d, b, j in panel_e if d == dom]
    show(f"专家panel {dom} (n={len(sub)})", [x[0] for x in sub], [x[1] for x in sub],
         "expert_panel_by_domain")
for dom in ("REG-S", "UNREG"):
    sub = [(b, j) for d, b, j in panel_l if d == dom]
    show(f"外行panel {dom} (n={len(sub)})", [x[0] for x in sub], [x[1] for x in sub],
         "lay_panel_by_domain")

print("\n=== E. A4 pilot30 分域 vs DeepSeek（含CI） ===")
b30 = bvi_map("A4", "pilot30")
for dom in ("REG-S", "UNREG"):
    ks = [k for k in b30 if b30[k][1] == dom and k in ds]
    show(f"A4-pilot30 {dom} (n={len(ks)})", [b30[k][0] for k in ks], [ds[k] for k in ks],
         "a4_pilot30_by_domain")

print("\n=== F. E1/E2/A4 在14条共享集 vs DeepSeek（含CI） ===")
bE1, bE2 = bvi_map("E1", "pilot14"), bvi_map("E2", "pilot14")
keys14 = sorted(set(bE1) & set(bE2))
for nm, bm in (("E1", bE1), ("E2", bE2),
               ("A4", {k: v for k, v in B36["A4"].items() if k in keys14})):
    ks = [k for k in keys14 if k in bm and k in ds]
    show(f"{nm} vs DS (n={len(ks)})", [bm[k][0] for k in ks], [ds[k] for k in ks],
         "shared14_vs_deepseek")

print("\n=== G. 专家-专家 3对（14条，含CI） ===")
bA4_14 = {k: v for k, v in B36["A4"].items() if k in keys14}
for n1, m1, n2, m2 in (("E1", bE1, "E2", bE2), ("E1", bE1, "A4", bA4_14), ("E2", bE2, "A4", bA4_14)):
    ks = [k for k in keys14 if k in m1 and k in m2]
    show(f"{n1} vs {n2} (n={len(ks)})", [m1[k][0] for k in ks], [m2[k][0] for k in ks],
         "expert_pairs_shared14")

print("\n=== H. A4 分域 vs DeepSeek（36锚点，含CI） ===")
for dom in ("REG-S", "REG-H", "UNREG"):
    ks = [k for k in B36["A4"] if B36["A4"][k][1] == dom and k in ds]
    show(f"A4 {dom} (n={len(ks)})", [B36["A4"][k][0] for k in ks], [ds[k] for k in ks],
         "a4_anchor36_by_domain")

print("\n=== I. E1/E2 分域 vs DeepSeek（14条内，含CI） ===")
for nm, bm in (("E1", bE1), ("E2", bE2)):
    for dom in ("REG-S", "UNREG"):
        ks = [k for k in keys14 if k in bm and bm[k][1] == dom and k in ds]
        show(f"{nm} {dom} (n={len(ks)})", [bm[k][0] for k in ks], [ds[k] for k in ks],
             "e1_e2_shared14_by_domain")

print("\n=== J. 专家间逐维度一致性（14条共享集，E1/E2/A4两两Po中位数） ===")
raw = {"E1": raters["E1"]["pilot14"], "E2": raters["E2"]["pilot14"],
       "A4": raters["A4"]["anchor36"]}
for dim in ("D1", "D2", "D3"):
    pos = []
    for r1, r2 in combinations(("E1", "E2", "A4"), 2):
        ks = [k for k in keys14 if k in raw[r1] and k in raw[r2]
              and min(raw[r1][k][dim], raw[r2][k][dim]) >= 0]
        po = sum(1 for k in ks if raw[r1][k][dim] == raw[r2][k][dim]) / len(ks)
        pos.append(po)
    pos.sort()
    print(f"  {dim}: 两两Po = {[f'{p*100:.0f}%' for p in pos]}  中位={pos[1]*100:.0f}%")
    RESULTS["sections"].setdefault("dimension_agreement_shared14", {})[dim] = {
        "pairwise_Po_sorted": [round(p, 4) for p in pos], "median_Po": round(pos[1], 4)}

print("\n=== K. D3严格度（D3=2占比） ===")
for nm, r, instr in (("E1", "E1", "pilot14"), ("E2", "E2", "pilot14"),
                     ("A4-pilot30", "A4", "pilot30"), ("A4-anchor36", "A4", "anchor36")):
    d = raters[r][instr]
    vals = [v["D3"] for v in d.values() if v["D3"] >= 0]
    share = sum(1 for x in vals if x == 2) / len(vals)
    print(f"  {nm}: D3=2 占 {share*100:.0f}% (n={len(vals)})")
    RESULTS["sections"].setdefault("d3_strictness", {})[nm] = {
        "d3_eq_2_share": round(share, 4), "n": len(vals)}

out_path = BASE / "results/human_extended_tables.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=1)
print(f"\n[OK] saved: {out_path}")
