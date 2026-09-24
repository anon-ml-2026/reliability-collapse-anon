# Fable5 P0-1: prevalence-matched cross-judge agreement (probe-level matching).
# Question: does the REG-S > REG-H > UNREG agreement gradient survive matching
# probes across domains by violation-flag rate? Pure re-slicing of existing R3 data.
# Also P1-2: probe-level cross-model flag-rate variance by domain.
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]  # repository root
DS_PATH = BASE / "data/scores/r3_judge_deepseek.json"
CL_PATH = BASE / "data/scores/r3_judge_claude.json"
OUT = BASE / "results/prevalence_matched_deepseek_claude.json"
SEED = 20260728
N_DRAWS = 500
DOMS = ["REG-S", "REG-H", "UNREG"]
BINS = [(-1e-9, 1e-9, "=0"), (1e-9, 0.20, "(0,20%]"), (0.20, 0.40, "(20,40%]"), (0.40, 1.01, "(40%+]")]


def load(path, drop_parse_error):
    raw = json.load(open(path, encoding="utf-8"))
    recs = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
    out = {}
    for r in recs:
        if r.get("D1", -1) < 0:
            continue
        if drop_parse_error and r.get("parse_error", False):
            continue
        out[r["trial_id"]] = r
    return out


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
    ds = load(DS_PATH, False)
    cl = load(CL_PATH, True)
    pairs = []
    for tid in sorted(set(ds) & set(cl)):
        ra, rb = ds[tid], cl[tid]
        pairs.append({"tid": tid, "probe": ra["probe_id"], "domain": ra["domain"],
                      "model": ra["model_id"],
                      "fa": int(ra["BVI"] >= 0.5), "fb": int(rb["BVI"] >= 0.5)})
    nonctl = [p for p in pairs if p["domain"] in DOMS]

    # per-probe violation rate = mean of the two judges' flags
    probe_trials = defaultdict(list)
    for p in nonctl:
        probe_trials[p["probe"]].append(p)
    probe_info = {}
    for pid, trs in probe_trials.items():
        rate = sum((t["fa"] + t["fb"]) / 2 for t in trs) / len(trs)
        probe_info[pid] = {"domain": trs[0]["domain"], "rate": rate, "n": len(trs)}

    # sanity: full-sample stats vs Table 4
    full = {d: agree_stats([p for p in nonctl if p["domain"] == d]) for d in DOMS}

    # --- bin-matched sampling, repeated draws ---
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
        per_bin_take = []
        for bi in range(len(BINS)):
            counts = {d: len(bin_probes[bi][d]) for d in DOMS}
            m = min(counts.values())
            if m == 0:
                continue
            for d in DOMS:
                sel.extend(rng.choice(bin_probes[bi][d], size=m, replace=False).tolist())
            per_bin_take.append((bi, m))
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

    # --- greedy nearest-neighbour triples (no replacement) ---
    by_dom = {d: sorted((pid for pid in probe_info if probe_info[pid]["domain"] == d),
                        key=lambda x: probe_info[x]["rate"]) for d in DOMS}
    triples, used = [], {d: set() for d in DOMS}
    for ps in by_dom["REG-S"]:
        if ps in used["REG-S"]:
            continue
        best = None
        for ph in by_dom["REG-H"]:
            if ph in used["REG-H"]:
                continue
            for pu in by_dom["UNREG"]:
                if pu in used["UNREG"]:
                    continue
                rates = [probe_info[x]["rate"] for x in (ps, ph, pu)]
                spread = max(rates) - min(rates)
                if best is None or spread < best[0]:
                    best = (spread, ph, pu)
        if best:
            triples.append((ps, best[1], best[2], best[0]))
            used["REG-S"].add(ps); used["REG-H"].add(best[1]); used["UNREG"].add(best[2])
    sel3 = [pid for t in triples for pid in t[:3]]
    sub3 = [p for p in nonctl if p["probe"] in set(sel3)]
    triple_stats = {d: agree_stats([p for p in sub3 if p["domain"] == d]) for d in DOMS}
    triple_rates = {d: float(np.mean([probe_info[pid]["rate"] for pid in sel3
                                      if probe_info[pid]["domain"] == d])) for d in DOMS}

    # --- P1-2: probe-level cross-model flag-rate variance (DeepSeek judge) ---
    pm = defaultdict(dict)
    for t in nonctl:
        pm[(t["probe"], t["model"])].setdefault("v", []).append(t["fa"])
    probe_model_rate = defaultdict(dict)
    for (pid, mid), d0 in pm.items():
        probe_model_rate[pid][mid] = sum(d0["v"]) / len(d0["v"])
    var_by_dom = defaultdict(list)
    for pid, mr in probe_model_rate.items():
        if len(mr) >= 6:
            var_by_dom[probe_info[pid]["domain"]].append(float(np.var(list(mr.values()), ddof=1)))
    p12 = {d: {"n_probes": len(v), "mean_var": float(np.mean(v)), "median_var": float(np.median(v))}
           for d, v in var_by_dom.items()}

    out = {
        "note": "probe rate = mean of both judges' binary flags; matching on probe rate; "
                "bin matching repeated %d draws (seed %d); greedy triples without replacement" % (N_DRAWS, SEED),
        "full_sample": {d: {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}
                        for d, s in full.items()},
        "probe_rate_by_domain": {d: sorted(round(probe_info[pid]["rate"], 3)
                                           for pid in probe_info if probe_info[pid]["domain"] == d)
                                 for d in DOMS},
        "bin_counts": {BINS[bi][2]: {d: len(bin_probes[bi][d]) for d in DOMS} for bi in range(len(BINS))},
        "bin_matched": {
            "median_n_probes": n_med,
            "matched_mean_rate": {d: round(rate_med[d], 4) for d in DOMS},
            "Po_median": {d: round(po_med[d], 4) for d in DOMS},
            "Po_ci95": {d: [round(x, 4) for x in po_q[d]] for d in DOMS},
            "kappa_median": {d: round(kap_med[d], 4) for d in DOMS},
        },
        "greedy_triples": {
            "n_triples": len(triples),
            "spreads": [round(t[3], 3) for t in triples],
            "matched_mean_rate": {d: round(triple_rates[d], 4) for d in DOMS},
            "stats": {d: {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in triple_stats[d].items()} for d in DOMS},
        },
        "p12_cross_model_variance": p12,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
