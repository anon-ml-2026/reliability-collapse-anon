"""
S3: Qwen 版 R3–R4 test-retest + R4 跨裁判交叉表（预注册: preregistration/validation_suite_20260730.md）
================================================================================
H1：Qwen 版 test-retest 方向复现 REG-S > REG-H,UNREG；PSI=0.5 条件稳定性 < 50%
   （参照：同脚本同口径重算 DeepSeek 版，论文值为 93.7/80.4/81.6 与 23.2%）。
H2：R4 上 Qwen×Kimi 与 Qwen×DeepSeek 两对，UNREG 均为最不可靠域（排序层面）。
join 键 = trial_id；二值 flag = BVI>=0.5；统计函数字节同 _cross_judge_qwen_claude.py。
"""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]  # repository root
OUT = BASE / "results/test_retest_qwen_and_r4_cross.json"
DOMS = ["CTL", "REG-S", "REG-H", "UNREG"]


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["trial_id"]: r for r in data
            if r.get("D1", -1) >= 0 and not r.get("parse_error", False)}


def cohen_kappa(a, b):
    levels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    obs = Counter()
    for x, y in zip(a, b):
        obs[(x, y)] += 1
    Po = sum(obs[(l, l)] for l in levels) / n
    m1, m2 = Counter(a), Counter(b)
    Pe = sum(m1[l] * m2[l] for l in levels) / (n * n)
    if Pe >= 1:
        return 1.0
    return (Po - Pe) / (1 - Pe)


def pabak(a, b):
    n = len(a)
    levels = list(sorted(set(a) | set(b)))
    k = len(levels)
    if k <= 1 or n == 0:
        return float("nan")
    Po = sum(1 for x, y in zip(a, b) if x == y) / n
    return (k * Po - 1) / (k - 1)


def binary_stats(pairs):
    out = {}
    for dom in DOMS:
        sub = [(a, b) for a, b in pairs if a["domain"] == dom]
        if not sub:
            continue
        fa = [1 if a["BVI"] >= 0.5 else 0 for a, b in sub]
        fb = [1 if b["BVI"] >= 0.5 else 0 for a, b in sub]
        po = sum(1 for x, y in zip(fa, fb) if x == y) / len(fa)
        out[dom] = {"n": len(fa), "Po": round(po * 100, 2),
                    "kappa": round(cohen_kappa(fa, fb), 4),
                    "PABAK": round(pabak(fa, fb), 4)}
    return out


def band_stability(pairs):
    """PSI=0.5 条件稳定性：src 判 0.5 的 trial 中 tgt 仍判 0.5 的比例（分域+整体）。"""
    out = {}
    for dom in [None] + DOMS:
        sub = [(a, b) for a, b in pairs if dom is None or a["domain"] == dom]
        src05 = [(a, b) for a, b in sub if a["BVI"] == 0.5]
        stay = sum(1 for a, b in src05 if b["BVI"] == 0.5)
        out[dom or "Overall"] = {"n_src_0.5": len(src05),
                                 "stay_0.5_pct": round(stay / len(src05) * 100, 1) if src05 else None}
    return out


def main():
    r3_qw = load(BASE / "data/scores/r3_judge_qwen.json")
    r4_qw = load(BASE / "data/scores/r4_judge_qwen.json")
    r3_ds = load(BASE / "data/scores/r3_judge_deepseek.json")
    r4_ds = load(BASE / "data/scores/r4_judge_deepseek.json")
    r4_kimi = load(BASE / "data/scores/r4_judge_kimi_run1.json")
    print(f"valid: R3-QW={len(r3_qw)} R4-QW={len(r4_qw)} R3-DS={len(r3_ds)} "
          f"R4-DS={len(r4_ds)} R4-Kimi={len(r4_kimi)}")

    # --- 1) test-retest: 同一 judge 跨轮（同 trial_id）---
    def tr_pairs(r3, r4):
        common = sorted(set(r3) & set(r4))
        return [(r3[t], r4[t]) for t in common]

    qw_tr = tr_pairs(r3_qw, r4_qw)
    ds_tr = tr_pairs(r3_ds, r4_ds)
    print(f"test-retest overlap: Qwen={len(qw_tr)} DeepSeek={len(ds_tr)}")

    res = {"overlap": {"qwen": len(qw_tr), "deepseek": len(ds_tr)}}
    for name, prs in [("qwen", qw_tr), ("deepseek_reference", ds_tr)]:
        bs = binary_stats(prs)
        band = band_stability(prs)
        res[f"test_retest_{name}"] = {"binary": bs, "band_0.5": band}
        print(f"\n== test-retest {name} ==")
        for d in DOMS:
            if d in bs:
                s = bs[d]
                print(f"  {d:6s} n={s['n']:5d} Po={s['Po']:6.2f}% κ={s['kappa']:7.4f} "
                      f"PABAK={s['PABAK']:7.4f} | 0.5-stay {band[d]['stay_0.5_pct']}% "
                      f"(n={band[d]['n_src_0.5']})")
        print(f"  Overall 0.5-stay {band['Overall']['stay_0.5_pct']}% "
              f"(n={band['Overall']['n_src_0.5']})")

    # H1 判定（Qwen 版）
    qb = res["test_retest_qwen"]["binary"]
    h1_dir = qb["REG-S"]["Po"] > qb["REG-H"]["Po"] and qb["REG-S"]["Po"] > qb["UNREG"]["Po"]
    h1_band = (res["test_retest_qwen"]["band_0.5"]["Overall"]["stay_0.5_pct"] or 100) < 50
    res["H1_direction_REG-S_highest"] = h1_dir
    res["H1_band05_below_50pct"] = h1_band

    # --- 2) R4 跨裁判交叉表：Qwen×Kimi、Qwen×DeepSeek ---
    for label, other in [("qwen_x_kimi", r4_kimi), ("qwen_x_deepseek", r4_ds)]:
        common = sorted(set(r4_qw) & set(other))
        prs = [(r4_qw[t], other[t]) for t in common]
        bs = binary_stats(prs)
        tri = {}
        for dom in DOMS:
            sub = [(a, b) for a, b in prs if a["domain"] == dom]
            if sub:
                po3 = sum(1 for a, b in sub if a["BVI"] == b["BVI"]) / len(sub)
                tri[dom] = {"n": len(sub), "Po3": round(po3 * 100, 2)}
        res[f"r4_cross_{label}"] = {"n_overlap": len(prs), "binary": bs, "three_level": tri}
        print(f"\n== R4 cross {label} (n={len(prs)}) ==")
        for d in DOMS:
            if d in bs:
                print(f"  {d:6s} Po={bs[d]['Po']:6.2f}% κ={bs[d]['kappa']:7.4f} "
                      f"PABAK={bs[d]['PABAK']:7.4f} Po3={tri[d]['Po3']:6.2f}%")

    nonctl = ["REG-S", "REG-H", "UNREG"]
    h2 = all(min(res[f"r4_cross_{p}"]["binary"][d]["Po"] for d in nonctl)
             == res[f"r4_cross_{p}"]["binary"]["UNREG"]["Po"]
             for p in ["qwen_x_kimi", "qwen_x_deepseek"])
    res["H2_UNREG_least_reliable_in_both_pairs"] = h2
    print(f"\nH1: REG-S highest = {h1_dir}, 0.5-stability<50% = {h1_band}")
    print(f"H2: UNREG least reliable in both R4 pairs = {h2}")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] {OUT}")


if __name__ == "__main__":
    main()
