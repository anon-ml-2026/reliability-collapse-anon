"""
S2: prevalence-matched 跨裁判一致性 — Qwen×Claude 对（复用 Appendix K 协议）
================================================================================
与 code/analysis/prevalence_matched.py 同一协议（probe 级分箱匹配，500 次抽取，
seed 20260728），作用于 P1-1 新对（Qwen3.6-Plus × Claude-Haiku-4.5）。
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]  # repository root
OUT = BASE / "results/prevalence_matched_qwen_claude.json"
SEED = 20260728
N_DRAWS = 500
DOMS = ["REG-S", "REG-H", "UNREG"]
BINS = [(-1e-9, 1e-9, "=0"), (1e-9, 0.20, "(0,20%]"), (0.20, 0.40, "(20,40%]"), (0.40, 1.01, "(40%+]")]


def load(path):
    raw = json.load(open(path, encoding="utf-8"))
    return {r["trial_id"]: r for r in raw
            if r.get("D1", -1) >= 0 and not r.get("parse_error", False)}


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in (0, 1)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agree_stats(pairs):
    a = [p["fa"] for p in pairs]
    b = [p["fb"] for p in pairs]
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n if n else float("nan")
    return {"n": n, "Po": po, "kappa": kappa(a, b),
            "rateA": sum(a) / n if n else float("nan"), "rateB": sum(b) / n if n else float("nan")}


def main():
    qw = load(BASE / "data/scores/r3_judge_qwen.json")
    cl = load(BASE / "data/scores/r3_judge_claude.json")
    pairs = []
    for tid in sorted(set(qw) & set(cl)):
        ra, rb = qw[tid], cl[tid]
        pairs.append({"tid": tid, "probe": ra["probe_id"], "domain": ra["domain"],
                      "model": ra["model_id"],
                      "fa": int(ra["BVI"] >= 0.5), "fb": int(rb["BVI"] >= 0.5)})
    nonctl = [p for p in pairs if p["domain"] in DOMS]

    probe_trials = defaultdict(list)
    for p in nonctl:
        probe_trials[p["probe"]].append(p)
    probe_info = {}
    for pid, trs in probe_trials.items():
        rate = sum((t["fa"] + t["fb"]) / 2 for t in trs) / len(trs)
        probe_info[pid] = {"domain": trs[0]["domain"], "rate": rate, "n": len(trs)}

    full = {d: agree_stats([p for p in nonctl if p["domain"] == d]) for d in DOMS}

    rng = np.random.default_rng(SEED)
    bin_probes = {bi: {d: [] for d in DOMS} for bi in range(len(BINS))}
    for pid, info in probe_info.items():
        for bi, (lo, hi, _) in enumerate(BINS):
            if lo < info["rate"] <= hi or (bi == 0 and info["rate"] == 0):
                bin_probes[bi][info["domain"]].append(pid)
                break
    draws = []
    for _ in range(N_DRAWS):
        sel = []
        for bi in range(len(BINS)):
            counts = {d: len(bin_probes[bi][d]) for d in DOMS}
            m = min(counts.values())
            if m == 0:
                continue
            for d in DOMS:
                sel.extend(rng.choice(bin_probes[bi][d], size=m, replace=False).tolist())
        sub = [p for p in nonctl if p["probe"] in set(sel)]
        st = {d: agree_stats([p for p in sub if p["domain"] == d]) for d in DOMS}
        matched_rates = {d: float(np.mean([probe_info[pid]["rate"] for pid in sel
                                           if probe_info[pid]["domain"] == d])) for d in DOMS}
        draws.append({"stats": st, "rates": matched_rates, "n_probes": len(sel)})
    po_med = {d: float(np.median([w["stats"][d]["Po"] for w in draws])) for d in DOMS}
    po_q = {d: [float(np.percentile([w["stats"][d]["Po"] for w in draws], q)) for q in (2.5, 97.5)] for d in DOMS}
    kap_med = {d: float(np.median([w["stats"][d]["kappa"] for w in draws])) for d in DOMS}
    rate_med = {d: float(np.median([w["rates"][d] for w in draws])) for d in DOMS}
    n_med = int(np.median([w["n_probes"] for w in draws]))

    out = {
        "pair": "Qwen3.6-Plus x Claude-Haiku-4.5 (DeepSeek-free)",
        "protocol": "identical to code/analysis/prevalence_matched.py (Appendix K)",
        "full_sample": {d: {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}
                        for d, s in full.items()},
        "bin_counts": {BINS[bi][2]: {d: len(bin_probes[bi][d]) for d in DOMS} for bi in range(len(BINS))},
        "bin_matched": {
            "median_n_probes": n_med,
            "matched_mean_rate": {d: round(rate_med[d], 4) for d in DOMS},
            "Po_median": {d: round(po_med[d], 4) for d in DOMS},
            "Po_ci95": {d: [round(x, 4) for x in po_q[d]] for d in DOMS},
            "kappa_median": {d: round(kap_med[d], 4) for d in DOMS},
        },
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
