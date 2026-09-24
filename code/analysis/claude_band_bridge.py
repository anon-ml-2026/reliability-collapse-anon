"""
S1: Claude-subject 0.5-band 跨裁判桥接分析（预注册: preregistration/validation_suite_20260730.md）
================================================================================
§7 "test-retest 未覆盖两个 Claude 模型" 的解析替代（无新评分）。
R3 既有三裁判（DeepSeek / Claude-Haiku / Qwen）下，DeepSeek 的 PSI=0.5 标签
在 Claude-subject trials vs 其他 6 模型 trials 上的跨裁判保持率对比；
bootstrap（10,000 次）差值 95% CI。
预声明判定：Claude 组保持率 ≤ 其他组 + 10pp ⇒ band 不稳定推广至 Claude 模型成立。
"""
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]  # repository root
OUT = BASE / "results/claude_band_bridge.json"
CLAUDE_MODELS = {"claude-haiku-4-5-20251001", "claude-opus-4-7"}
N_BOOT = 10000
SEED = 20260730


def load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["trial_id"]: r for r in data
            if r.get("D1", -1) >= 0 and not r.get("parse_error", False)}


def retention(src, tgt, tids):
    sub = [t for t in tids if t in tgt]
    n = len(sub)
    if n == 0:
        return None
    stay = sum(1 for t in sub if tgt[t]["BVI"] == 0.5)
    return {"n": n, "stay": stay, "rate": stay / n}


def boot_diff(tids_a, tids_b, tgt, seed):
    rng = random.Random(seed)
    diffs = []
    for _ in range(N_BOOT):
        def one(tids):
            if not tids:
                return None
            samp = [tids[rng.randrange(len(tids))] for _ in range(len(tids))]
            use = [t for t in samp if t in tgt]
            if not use:
                return None
            return sum(1 for t in use if tgt[t]["BVI"] == 0.5) / len(use)
        ra, rb = one(tids_a), one(tids_b)
        if ra is not None and rb is not None:
            diffs.append(ra - rb)
    diffs.sort()
    if not diffs:
        return None
    return [round(diffs[int(0.025 * len(diffs))], 4),
            round(diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)], 4)]


def main():
    ds = load(BASE / "data/scores/r3_judge_deepseek.json")
    cl = load(BASE / "data/scores/r3_judge_claude.json")
    qw = load(BASE / "data/scores/r3_judge_qwen.json")
    print(f"valid: DS={len(ds)} CL={len(cl)} QW={len(qw)}")

    groups = {
        "claude_subject": lambda r: r.get("model_id") in CLAUDE_MODELS,
        "other_6_models": lambda r: r.get("model_id") not in CLAUDE_MODELS,
    }

    comparisons = [
        ("DS 0.5 -> read by Qwen", ds, qw),
        ("DS 0.5 -> read by Claude", ds, cl),
        ("Claude 0.5 -> read by Qwen", cl, qw),
        ("Qwen 0.5 -> read by DS", qw, ds),
    ]

    out = {"prespec": "preregistration/validation_suite_20260730.md",
           "rule": "claude_subject retention <= other + 10pp => instability generalizes",
           "comparisons": {}}
    all_pass = True
    for label, src, tgt in comparisons:
        entry = {}
        tids_by_group = {}
        for gname, pred in groups.items():
            tids = [t for t, r in src.items() if r["BVI"] == 0.5 and pred(r)]
            tids_by_group[gname] = tids
            r = retention(src, tgt, tids)
            entry[gname] = {"n": r["n"], "stay0.5_pct": round(r["rate"] * 100, 1)} if r else {"n": 0}
        diff_ci = boot_diff(tids_by_group["claude_subject"], tids_by_group["other_6_models"],
                            tgt, SEED)
        a, b = entry["claude_subject"], entry["other_6_models"]
        delta = round(a["stay0.5_pct"] - b["stay0.5_pct"], 1) if a.get("n") and b.get("n") else None
        passes = (delta is not None) and (a["stay0.5_pct"] <= b["stay0.5_pct"] + 10)
        all_pass = all_pass and passes
        entry.update({"delta_pp": delta, "diff_CI95": diff_ci, "pass_rule": passes})
        out["comparisons"][label] = entry
        print(f"{label}: Claude {a} vs other {b}  Δ={delta}pp  CI={diff_ci}  pass={passes}")

    # 参考：两组二值跨裁判 Po（Qwen×Claude 对）
    common = sorted(set(qw) & set(cl))
    ref = {}
    for gname, pred in groups.items():
        sub = [t for t in common if pred(qw[t])]
        po = sum(1 for t in sub if (qw[t]["BVI"] >= 0.5) == (cl[t]["BVI"] >= 0.5)) / len(sub)
        ref[gname] = {"n": len(sub), "binary_Po_pct": round(po * 100, 1)}
    out["reference_qwen_claude_binary_Po"] = ref
    print("Qwen×Claude binary Po by group:", ref)

    out["verdict"] = ("0.5-band instability generalizes to Claude-subject trials"
                      if all_pass else "bridge FAILED on at least one comparison")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nverdict: {out['verdict']}")
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
