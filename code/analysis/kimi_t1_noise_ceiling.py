"""
S5: Kimi T=1 双跑噪声天花板 — 预注册判定分析（2026-07-30）
================================================================================
判定规则（preregistration/validation_suite_20260730.md 预注册）：
- 计算 Kimi run1×run2 自身一致性（同模型同温度 T=1，两次独立采样判分）
- 若 between-judge gap（DS×Kimi、QW×Kimi）≫ Kimi 自身双跑 gap
  -> 跨裁判不一致主要来自真实判断差异，而非采样噪声
- 0.5-band 自稳定：run1 判 0.5 的 trial 中 run2 仍判 0.5 的比例，
  与 DS/Qwen 源 0.5 带不稳定（23.2%/33.3%）对照
join 键 = trial_id；二值 flag = BVI>=0.5；统计函数字节同 r4_qwen_test_retest.py。
"""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]  # repository root
OUT = BASE / "results/kimi_t1_noise_ceiling.json"
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
    out = {}
    for dom in [None] + DOMS:
        sub = [(a, b) for a, b in pairs if dom is None or a["domain"] == dom]
        src05 = [(a, b) for a, b in sub if a["BVI"] == 0.5]
        stay = sum(1 for a, b in src05 if b["BVI"] == 0.5)
        out[dom or "Overall"] = {"n_src_0.5": len(src05),
                                 "stay_0.5_pct": round(stay / len(src05) * 100, 1) if src05 else None}
    return out


def pairs_of(d1, d2):
    common = sorted(set(d1) & set(d2))
    return [(d1[t], d2[t]) for t in common]


def main():
    k1 = load(BASE / "data/scores/r4_judge_kimi_run1.json")
    k2 = load(BASE / "data/scores/r4_judge_kimi_run2.json")
    ds = load(BASE / "data/scores/r4_judge_deepseek.json")
    qw = load(BASE / "data/scores/r4_judge_qwen.json")
    print(f"valid: Kimi1={len(k1)} Kimi2={len(k2)} DS={len(ds)} QW={len(qw)}")

    res = {}

    # --- 1) Kimi 自一致（噪声天花板）---
    kk = pairs_of(k1, k2)
    bs_kk = binary_stats(kk)
    band_kk = band_stability(kk)
    res["kimi_self_T1"] = {"n_overlap": len(kk), "binary": bs_kk, "band_0.5": band_kk}
    print(f"\n== Kimi run1 x run2 self-agreement (n={len(kk)}) ==")
    for d in DOMS:
        if d in bs_kk:
            s = bs_kk[d]
            print(f"  {d:6s} n={s['n']:5d} Po={s['Po']:6.2f}% κ={s['kappa']:7.4f} "
                  f"PABAK={s['PABAK']:7.4f} | 0.5-stay {band_kk[d]['stay_0.5_pct']}% "
                  f"(n={band_kk[d]['n_src_0.5']})")
    print(f"  Overall 0.5-stay {band_kk['Overall']['stay_0.5_pct']}% "
          f"(n={band_kk['Overall']['n_src_0.5']})")

    # --- 2) between-judge 参照：DS×Kimi、QW×Kimi（各自与 Kimi run1）---
    for label, other in [("ds_x_kimi", ds), ("qwen_x_kimi", qw)]:
        prs = pairs_of(other, k1)
        bs = binary_stats(prs)
        res[f"between_{label}"] = {"n_overlap": len(prs), "binary": bs}
        print(f"\n== between-judge {label} (n={len(prs)}) ==")
        for d in DOMS:
            if d in bs:
                print(f"  {d:6s} Po={bs[d]['Po']:6.2f}% κ={bs[d]['kappa']:7.4f}")

    # --- 3) 判定：between-judge gap 是否 ≫ 噪声 ---
    verdicts = {}
    for d in ["REG-S", "REG-H", "UNREG"]:
        noise_gap = 100 - bs_kk[d]["Po"]
        bj_ds = 100 - res["between_ds_x_kimi"]["binary"][d]["Po"]
        bj_qw = 100 - res["between_qwen_x_kimi"]["binary"][d]["Po"]
        verdicts[d] = {
            "self_disagreement_pp": round(noise_gap, 2),
            "ds_x_kimi_disagreement_pp": round(bj_ds, 2),
            "qwen_x_kimi_disagreement_pp": round(bj_qw, 2),
            "ds_ratio": round(bj_ds / noise_gap, 2) if noise_gap > 0 else None,
            "qwen_ratio": round(bj_qw / noise_gap, 2) if noise_gap > 0 else None,
        }
    res["noise_vs_between"] = verdicts
    res["reference_band_stability"] = {
        "DS_source_0.5_stay_overall": 23.2,
        "QW_source_0.5_stay_overall": 33.3,
    }
    print("\n== noise ceiling vs between-judge disagreement (pp) ==")
    for d, v in verdicts.items():
        print(f"  {d:6s} self={v['self_disagreement_pp']:5.2f} "
              f"DSxK={v['ds_x_kimi_disagreement_pp']:5.2f} ({v['ds_ratio']}x) "
              f"QWxK={v['qwen_x_kimi_disagreement_pp']:5.2f} ({v['qwen_ratio']}x)")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] {OUT}")


if __name__ == "__main__":
    main()
