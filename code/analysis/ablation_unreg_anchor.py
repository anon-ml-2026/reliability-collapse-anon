"""
S4: UNREG 锚定 rubric ablation — 预注册判定分析（2026-07-30）
================================================================================
判定规则（preregistration/validation_suite_20260730.md 预注册）：
- 基线（原 rubric, DS×QW 二值 Po）：UNREG 85.0%、REG-S 96.7%、gap=11.7pp
- 锚定后 UNREG Po >= 90.9%（收复 >=50% gap） -> 支持 instrument-fit
- < 87.9%（收复 <25%） -> 支持 codification 读物
- 介于之间 -> 不结论
- 安慰剂约束：REG-S |Δ| >= 2pp -> 声明跨域外溢并弱化结论
统计：配对 McNemar（exact binomial，discordant = 原一致/锚定不一致 vs 原不一致/锚定一致），α=.05。
"""
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]  # repository root
OUT = BASE / "results/ablation_unreg_anchor.json"

PREDS = {  # 预注册基线与阈值
    "baseline_unreg_po": 85.0, "baseline_regs_po": 96.7, "gap_pp": 11.7,
    "thr_instrument_fit": 90.9, "thr_codification": 87.9, "placebo_pp": 2.0,
}


def load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["trial_id"]: r for r in data
            if r.get("D1", -1) >= 0 and not r.get("parse_error", False)}


def flag(r):
    return 1 if r["BVI"] >= 0.5 else 0


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"b": b, "c": c, "p": min(1.0, 2 * tail)}


def paired_domain(ds_o, qw_o, ds_a, qw_a, domain):
    tids = [t for t in ds_o if ds_o[t]["domain"] == domain
            and t in qw_o and t in ds_a and t in qw_a]
    n = len(tids)
    po_o = sum(1 for t in tids if flag(ds_o[t]) == flag(qw_o[t])) / n
    po_a = sum(1 for t in tids if flag(ds_a[t]) == flag(qw_a[t])) / n
    b = sum(1 for t in tids if flag(ds_o[t]) == flag(qw_o[t])
            and flag(ds_a[t]) != flag(qw_a[t]))   # 原一致 -> 锚定不一致
    c = sum(1 for t in tids if flag(ds_o[t]) != flag(qw_o[t])
            and flag(ds_a[t]) == flag(qw_a[t]))   # 原不一致 -> 锚定一致
    tri_o = sum(1 for t in tids if ds_o[t]["BVI"] == qw_o[t]["BVI"]) / n
    tri_a = sum(1 for t in tids if ds_a[t]["BVI"] == qw_a[t]["BVI"]) / n
    return {"n_paired": n,
            "binary_Po_original": round(po_o * 100, 2),
            "binary_Po_anchored": round(po_a * 100, 2),
            "delta_pp": round((po_a - po_o) * 100, 2),
            "mcnemar": mcnemar_exact(b, c),
            "threelevel_Po_original": round(tri_o * 100, 2),
            "threelevel_Po_anchored": round(tri_a * 100, 2)}


def main():
    ds_o = load(BASE / "data/scores/r3_judge_deepseek.json")
    qw_o = load(BASE / "data/scores/r3_judge_qwen.json")
    ds_a = load(BASE / "data/scores/r3_ablation_unreg_anchored_deepseek.json")
    qw_a = load(BASE / "data/scores/r3_ablation_unreg_anchored_qwen.json")
    print(f"valid: DS-orig={len(ds_o)} QW-orig={len(qw_o)} DS-anch={len(ds_a)} QW-anch={len(qw_a)}")

    unreg = paired_domain(ds_o, qw_o, ds_a, qw_a, "UNREG")
    regs = paired_domain(ds_o, qw_o, ds_a, qw_a, "REG-S")

    po = unreg["binary_Po_anchored"]
    recovered = (po - PREDS["baseline_unreg_po"]) / PREDS["gap_pp"] * 100
    if po >= PREDS["thr_instrument_fit"]:
        verdict = "instrument-fit SUPPORTED (>=90.9%, gap recovery >=50%)"
    elif po < PREDS["thr_codification"]:
        verdict = "codification SUPPORTED (<87.9%, gap recovery <25%)"
    else:
        verdict = "INCONCLUSIVE (between 87.9% and 90.9%)"
    placebo_delta = regs["delta_pp"]
    placebo_breach = abs(placebo_delta) >= PREDS["placebo_pp"]

    out = {
        "prespec": "preregistration/validation_suite_20260730.md (S4)",
        "rubric_variant": "v2_d1d2d3_unreganchored (3 author UNREG anchors, else byte-identical)",
        "judges": "DeepSeek-V4-Pro x Qwen3.6-Plus, T=0",
        "preregistered_thresholds": PREDS,
        "UNREG": unreg,
        "REG_S_placebo": {**regs, "placebo_breach_|d|>=2pp": placebo_breach},
        "gap_recovery_pct_vs_baseline": round(recovered, 1),
        "verdict": verdict,
        "placebo_note": ("REG-S shifted |%s|pp >= 2pp -> declare cross-domain spillover, weaken conclusion"
                         % placebo_delta) if placebo_breach else
                        ("REG-S shift %s pp within +/-2pp -> no spillover" % placebo_delta),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
