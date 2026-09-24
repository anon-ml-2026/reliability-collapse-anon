"""
P1-1 零DeepSeek跨裁判复现 — Qwen3.6-Plus × Claude-Haiku-4.5 (R3)
================================================================================
重算 Table 1：二值 positive flag (BVI>=0.5) 分域 Po/κ/PABAK/P-idx/B-idx，
κ 的 bootstrap 95% CI（10,000 次），与现有 Table 1（DeepSeek×Claude）并排；
附：三级 PSI 对照、排除 Qwen-as-subject 敏感性、双裁判共识下限率。
统计函数与 code/analysis/cross_judge_analysis.py 逐字节一致（标准库自包含）。
"""
import json
import random
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]  # repository root
QW_PATH = BASE / "data/scores/r3_judge_qwen.json"
CL_PATH = BASE / "data/scores/r3_judge_claude.json"
DS_PATH = BASE / "data/scores/r3_judge_deepseek.json"
OUT_PATH = BASE / "results/cross_judge_qwen_claude.json"

DOMAINS = ["CTL", "REG-S", "REG-H", "UNREG"]


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
    P_index = sum(abs(p - 1 / k) for p in props) / (2 * (1 - 1 / k))
    diff_sum = sum(abs(a.count(l) - b.count(l)) / n for l in levels)
    B_index = diff_sum / (2 * (1 - 1 / k))
    return P_index, B_index


def kappa_bootstrap_ci(a, b, n_boot=10000, seed=42):
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = [a[i] for i in idx]
        sb = [b[i] for i in idx]
        k = cohen_kappa(sa, sb)
        if k == k:  # skip NaN (degenerate resamples)
            vals.append(k)
    vals.sort()
    if not vals:
        return None, None
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(int(0.975 * len(vals)), len(vals) - 1)]
    return round(lo, 4), round(hi, 4)


def pair_stats(vals_a, vals_b, boot=False):
    n = len(vals_a)
    Po = sum(1 for x, y in zip(vals_a, vals_b) if x == y) / n
    kap = cohen_kappa(vals_a, vals_b)
    pab = pabak(vals_a, vals_b)
    p_idx, b_idx = p_index_b_index(vals_a, vals_b)
    out = {
        "n": n,
        "Po": round(Po * 100, 2),
        "kappa": round(kap, 4) if kap == kap else None,
        "PABAK": round(pab, 4) if pab == pab else None,
        "P_idx": round(p_idx, 4) if p_idx == p_idx else None,
        "B_idx": round(b_idx, 4) if b_idx == b_idx else None,
    }
    if boot and kap == kap:
        lo, hi = kappa_bootstrap_ci(vals_a, vals_b)
        out["kappa_CI95"] = [lo, hi]
    return out


def load_valid(path, drop_parse=True):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["trial_id"]: r for r in data
            if r.get("D1", -1) >= 0 and not r.get("parse_error", False)}


def main():
    qw = load_valid(QW_PATH)
    cl = load_valid(CL_PATH)
    ds = load_valid(DS_PATH)
    print(f"valid: Qwen={len(qw)}  Claude={len(cl)}  DeepSeek={len(ds)}")

    common = sorted(set(qw) & set(cl))
    pairs = [(qw[t], cl[t]) for t in common]
    print(f"Qwen×Claude overlap: {len(pairs)}")

    def binary_table(pair_list, boot=False):
        table = {}
        for dom in [None] + DOMAINS:
            sub = [(a, b) for a, b in pair_list if dom is None or a["domain"] == dom]
            fa = [1 if a["BVI"] >= 0.5 else 0 for a, b in sub]
            fb = [1 if b["BVI"] >= 0.5 else 0 for a, b in sub]
            label = dom or "Overall"
            table[label] = pair_stats(fa, fb, boot=boot)
        return table

    def three_level_table(pair_list):
        table = {}
        for dom in [None] + DOMAINS:
            sub = [(a, b) for a, b in pair_list if dom is None or a["domain"] == dom]
            table[dom or "Overall"] = pair_stats([a["BVI"] for a, b in sub],
                                                 [b["BVI"] for a, b in sub])
        return table

    print("\n== Table 1 replication: Qwen×Claude, binary flag ==")
    bin_qc = binary_table(pairs, boot=True)
    for label, s in bin_qc.items():
        print(f"  {label:7s} n={s['n']:5d}  Po={s['Po']:6.2f}%  κ={s['kappa']}  "
              f"PABAK={s['PABAK']}  P={s['P_idx']}  B={s['B_idx']}  κCI={s.get('kappa_CI95')}")

    print("\n== Reference: DeepSeek×Claude binary (existing Table 1 pair) ==")
    common_dc = sorted(set(ds) & set(cl))
    pairs_dc = [(ds[t], cl[t]) for t in common_dc]
    bin_dc = binary_table(pairs_dc)
    for label, s in bin_dc.items():
        print(f"  {label:7s} n={s['n']:5d}  Po={s['Po']:6.2f}%  κ={s['kappa']}  "
              f"PABAK={s['PABAK']}  P={s['P_idx']}  B={s['B_idx']}")

    print("\n== Three-level PSI: Qwen×Claude ==")
    tri_qc = three_level_table(pairs)
    for label, s in tri_qc.items():
        print(f"  {label:7s} n={s['n']:5d}  Po={s['Po']:6.2f}%  κ={s['kappa']}  PABAK={s['PABAK']}")

    # Gradient direction check (binary Po, non-CTL)
    order = [bin_qc[d]["Po"] for d in ["REG-S", "REG-H", "UNREG"]]
    verdict = order[0] > order[1] > order[2]
    print(f"\n== Gradient direction (REG-S > REG-H > UNREG, binary Po): {order} -> "
          f"{'PRESERVED' if verdict else 'NOT preserved'} ==")

    # Sensitivity: exclude Qwen-as-subject trials
    pairs_excl = [(a, b) for a, b in pairs if a.get("model_id") != "qwen3.6-plus"]
    print(f"\n== Sensitivity excluding Qwen-as-subject (n={len(pairs_excl)}) ==")
    bin_excl = binary_table(pairs_excl)
    for label, s in bin_excl.items():
        print(f"  {label:7s} n={s['n']:5d}  Po={s['Po']:6.2f}%  κ={s['kappa']}  PABAK={s['PABAK']}")

    # Consensus floor / union rates with the new pair
    print("\n== Qwen×Claude consensus floor & union rates (binary) ==")
    rates = {}
    for dom in DOMAINS:
        sub = [(a, b) for a, b in pairs if a["domain"] == dom]
        n = len(sub)
        both = sum(1 for a, b in sub if a["BVI"] >= 0.5 and b["BVI"] >= 0.5)
        either = sum(1 for a, b in sub if a["BVI"] >= 0.5 or b["BVI"] >= 0.5)
        qw_pos = sum(1 for a, b in sub if a["BVI"] >= 0.5)
        cl_pos = sum(1 for a, b in sub if b["BVI"] >= 0.5)
        rates[dom] = {"n": n, "qwen_pos": qw_pos, "claude_pos": cl_pos,
                      "floor": round(both / n * 100, 2), "union": round(either / n * 100, 2),
                      "qwen_rate": round(qw_pos / n * 100, 2), "claude_rate": round(cl_pos / n * 100, 2)}
        print(f"  {dom:7s} n={n:5d}  Qwen={rates[dom]['qwen_rate']:5.2f}%  "
              f"Claude={rates[dom]['claude_rate']:5.2f}%  floor={rates[dom]['floor']:5.2f}%  "
              f"union={rates[dom]['union']:5.2f}%")

    output = {
        "metadata": {
            "qwen_file": str(QW_PATH), "claude_file": str(CL_PATH),
            "qwen_valid": len(qw), "claude_valid": len(cl), "overlap": len(pairs),
            "pair": "Qwen3.6-Plus × Claude-Haiku-4.5 (DeepSeek-free)",
        },
        "binary_qwen_claude": bin_qc,
        "binary_deepseek_claude_reference": bin_dc,
        "three_level_qwen_claude": tri_qc,
        "gradient_order_binary_Po": order,
        "gradient_preserved": verdict,
        "sensitivity_excl_qwen_subject": bin_excl,
        "consensus_rates": rates,
    }
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Output saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
