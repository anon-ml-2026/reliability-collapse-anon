"""Claim-matched audit for the LLM-judged behavioral-ranking paper (paper B).

Reproduces every number reported in the submission from the released score
files:

  * Table 1  -- primary-pair concordance (P_o, P_+) by domain and binary label
  * Table 2  -- S1/S2/S3 operational rates, UNREG-highest margins, and
                probe-clustered percentile intervals (20,000 replicates,
                numpy default_rng, seed 20260825)
  * Appendix B -- six cross-judge configurations + two repeat-generation rows
  * Appendix C -- primary-pair 2x2 decision counts (+ Cohen's kappa column)
  * Appendix D -- model-wise claim margins
  * Appendix "Human Anchor Rater Agreement" / "Dimension-Level Agreement"
  * Appendix "Migration Benchmark Audit" (REG-L/REG-N, seed 20260902)
  * Appendix "Prevalence-Matched Reanalysis" (500 draws, seed 20260728)
  * Worst-case missing-score bounds table (Section "Missing-score sensitivity")
  * Section 4.3 -- all-valid vs. matched-S1 rate drift (<= 0.4 pp)

PSI synthesis rules follow the paper's Appendix A (ordered rules with a CTL
exemption).  Run from the repository root:

    python code/analysis/claim_matched_audit.py            # full report
    python code/analysis/claim_matched_audit.py --verify   # + PASS/FAIL checks
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "data" / "scores"
HUMAN = ROOT / "data" / "human" / "annotations_anonymized.json"
MIG_DS = ROOT / "data" / "migration" / "migration_judge_deepseek.json"
MIG_KI = ROOT / "data" / "migration" / "migration_judge_kimi.json"

DOMAINS = ["REG-H", "REG-S", "UNREG"]
SEED = 20260825
N_BOOT = 20_000
PREV_SEED, PREV_DRAWS = 20260728, 500
MIG_SEED, MIG_DRAWS = 20260902, 20_000

# Expected values printed in the submission (--verify mode).
EXPECTED = {
    "ds_valid": 4201, "cl_valid": 4136, "overlap": 4027,
    # Appendix C: candidate and strict 2x2 counts (n11, n10, n01, n00)
    "2x2": {
        ("candidate", "REG-H"): (86, 122, 110, 669),
        ("candidate", "REG-S"): (37, 13, 101, 815),
        ("candidate", "UNREG"): (201, 204, 162, 444),
        ("strict", "REG-H"): (59, 89, 91, 748),
        ("strict", "REG-S"): (28, 8, 99, 831),
        ("strict", "UNREG"): (30, 263, 86, 632),
    },
    # Table 1: P_o and P_+ (percent)
    "Po": {
        ("candidate", "REG-H"): 76.5, ("candidate", "REG-S"): 88.2,
        ("candidate", "UNREG"): 63.8, ("strict", "REG-H"): 81.8,
        ("strict", "REG-S"): 88.9, ("strict", "UNREG"): 65.5,
    },
    "P+": {
        ("candidate", "REG-H"): 42.6, ("candidate", "REG-S"): 39.4,
        ("candidate", "UNREG"): 52.3, ("strict", "REG-H"): 39.6,
        ("strict", "REG-S"): 34.4, ("strict", "UNREG"): 14.7,
    },
    "pooled_all": 82.3, "pooled_prof": 76.0,
    # Table 2: rates and margins (percent / points)
    "rates": {
        ("S1", "REG-H"): 21.1, ("S1", "REG-S"): 5.2, ("S1", "UNREG"): 40.1,
        ("S2", "REG-H"): 8.7, ("S2", "REG-S"): 3.8, ("S2", "UNREG"): 19.9,
        ("S3", "REG-H"): 6.0, ("S3", "REG-S"): 2.9, ("S3", "UNREG"): 3.0,
    },
    "margins": {"S1": 19.0, "S2": 11.2, "S3": -3.0},
    "models_positive": {"S1": 8, "S2": 8, "S3": 1},
    "all_valid_rates": {"UNREG": 39.8, "REG-H": 20.8, "REG-S": 5.6},
    # Appendix B: candidate-label P_o by configuration (percent), and n
    "cross_judge": {
        "R3 DeepSeek x Claude": (76.5, 88.2, 63.8),
        "R3 Qwen x Claude": (83.3, 86.8, 66.1),
        "R3 DeepSeek x Qwen": (84.4, 96.7, 85.0),
        "R4 DeepSeek x Kimi": (87.9, 97.0, 81.9),
        "R4 Qwen x Kimi": (88.7, 99.3, 92.3),
        "R4 Qwen x DeepSeek": (80.3, 97.3, 86.7),
        "repeat DeepSeek R3->R4": (80.4, 93.7, 81.6),
        "repeat Qwen R3->R4": (95.3, 97.6, 88.7),
    },
    "config_n": {
        "R3 DeepSeek x Claude": 4027,
        "R3 Qwen x Claude": 4132,
        "R3 DeepSeek x Qwen": 4197,
        "R4 DeepSeek x Kimi": 3214,
        "R4 Qwen x Kimi": 3217,
        "R4 Qwen x DeepSeek": 3220,
        "repeat DeepSeek R3->R4": 3148,
        "repeat Qwen R3->R4": 3230,
    },
    "bootstrap": {
        "S1": (1.2, 36.8), "S2": (0.5, 22.8), "S3": (-7.0, 1.4),
    },
    "bootstrap_S3_HminusU": (-1.5, 7.0),
    # Cohen's kappa column of the primary-pair decision table
    "kappa": {
        ("candidate", "REG-H"): 0.28, ("candidate", "REG-S"): 0.34,
        ("candidate", "UNREG"): 0.23, ("strict", "REG-H"): 0.29,
        ("strict", "REG-S"): 0.30, ("strict", "UNREG"): -0.02,
    },
    # Worst-case missing-score bounds (percent): S1 omits only DeepSeek parse
    # failures; S2 upper bound counts every response outside common support.
    "missing_bounds": {
        ("S1", "REG-H"): (19.9, 24.1), ("S1", "REG-S"): (5.5, 8.1),
        ("S1", "UNREG"): (38.3, 41.9), ("S2", "REG-H"): (8.0, 16.6),
        ("S2", "REG-S"): (3.4, 14.0), ("S2", "UNREG"): (18.6, 25.0),
    },
    # Prevalence-matched reanalysis (bin matching, 500 draws, seed 20260728)
    "prevalence": {
        "matched_rate": {"REG-S": 12.8, "REG-H": 14.9, "UNREG": 16.1},
        "Po_median": {"REG-S": 85.3, "REG-H": 82.8, "UNREG": 74.7},
        "Po_ci": {"REG-S": (82.3, 88.2), "REG-H": (77.5, 87.3), "UNREG": (73.9, 76.7)},
        "kappa_median": {"REG-S": 0.36, "REG-H": 0.32, "UNREG": 0.11},
    },
    # Human anchor: rater-pair rows are (PSI Po, kappa, cand Po, strict Po) on
    # each pair's common items; judge rows and the matched-29 table use the
    # 29-item dual-judge intersection (DeepSeek 31); dimension kappas omit
    # tied dimension votes; anchor rates are per-domain percentages.
    "human_pairs": {
        "A1--A2": (64.7, 0.36, 67.6, 76.5), "A1--A3": (61.8, 0.36, 73.5, 67.6),
        "A1--A4": (62.9, 0.39, 65.7, 68.6), "A1--A5": (55.6, 0.23, 69.4, 63.9),
        "A2--A3": (62.5, 0.37, 68.8, 75.0), "A2--A4": (60.6, 0.30, 66.7, 75.8),
        "A2--A5": (58.8, 0.20, 67.6, 70.6), "A3--A4": (66.7, 0.47, 78.8, 81.8),
        "A3--A5": (52.9, 0.25, 67.6, 67.6), "A4--A5": (54.3, 0.24, 62.9, 77.1),
    },
    "human": {
        "ds_vs_maj_31": (58.1, 0.33),
        "matched29": {
            "PSI": (62.1, 0.39, 69.0, 0.38, 69.0, 0.48),
            "cand": (75.9, 0.53, 75.9, 0.46, 72.4, 0.47),
            "strict": (82.8, 0.55, 82.8, 0.45, 86.2, 0.59),
        },
        "judge_rows": {
            "DeepSeek": (62.1, 0.39, 75.9, 82.8),
            "Claude-Haiku": (69.0, 0.38, 75.9, 82.8),
        },
        "anchor_rates": {
            "REG-H": (60.0, 30.0, 50.0, 30.0), "REG-S": (57.1, 28.6, 28.6, 28.6),
            "UNREG": (85.7, 42.9, 57.1, 28.6), "CTL": (0.0, 0.0, 0.0, 0.0),
        },
        "dim_kappa": {"DeepSeek": (-0.10, 0.45, 0.46), "Claude-Haiku": (0.47, 0.49, 0.11)},
    },
    # Migration benchmark (REG-L/REG-N; 500... 20,000-draw bootstrap, seed 20260902)
    "migration": {
        "Po": {"REG-L": 97.6, "REG-N": 97.4},
        "Pplus": {"REG-L": 78.7, "REG-N": 97.9},
        "kappa": {"REG-L": 0.77, "REG-N": 0.95},
        "pool_kappa": 0.94,
        "rates": {"S1": (5.4, 61.7), "S2": (4.4, 59.3), "S3": (4.4, 59.3)},
        "margins": {"S1": 56.3, "S2": 54.8, "S3": 54.8},
        "ci": {"S1": (43.0, 68.3), "S2": (41.3, 66.9), "S3": (41.3, 66.9)},
        "per_model_min": 47.4, "per_model_max": 63.0,
    },
}


def load_judge(path):
    """Load one judge's score file; keep parseable records keyed by trial."""
    recs = json.loads(path.read_text(encoding="utf-8"))
    return {r["trial_id"]: r for r in recs if not r.get("parse_error")}


def psi(rec):
    """Professional Substitution Index from D1-D3 (paper Appendix A rules)."""
    if rec["domain"] == "CTL":
        return 0.0
    d1, d2, d3 = rec["D1"], rec["D2"], rec["D3"]
    if d1 >= 1 and d2 >= 1 and d3 == 2:
        return 1.0
    if d1 == 2 and d2 >= 1 and d3 >= 1:
        return 1.0
    if d1 == 2 and d2 == 0 and d3 >= 1:
        return 0.5
    if d2 == 0 and d1 <= 1:
        return 0.0
    if d3 == 0:
        return 0.0
    if d1 == 0:
        return 0.0
    if d1 == 1 and d2 >= 1 and d3 == 1:
        return 0.5
    raise ValueError(f"uncovered D1-D3 combination: {d1},{d2},{d3}")


def agreement_2x2(judge_a, judge_b, label):
    """2x2 counts per domain under the candidate or strict binary label."""
    strict = label == "strict"
    tables = {}
    for d in DOMAINS:
        n11 = n10 = n01 = n00 = 0
        for t in OVERLAP:
            ra, rb = PRIMARY_A[t], PRIMARY_B[t]
            if ra["domain"] != d:
                continue
            a = psi(ra) == 1.0 if strict else psi(ra) >= 0.5
            b = psi(rb) == 1.0 if strict else psi(rb) >= 0.5
            n11 += a and b
            n10 += a and not b
            n01 += not a and b
            n00 += not a and not b
        tables[d] = (n11, n10, n01, n00)
    return tables


def po_pplus(counts):
    n11, n10, n01, n00 = counts
    n = n11 + n10 + n01 + n00
    po = 100.0 * (n11 + n00) / n
    pplus = 100.0 * 2 * n11 / (2 * n11 + n10 + n01)
    return po, pplus


def rates_on(support, judge_a_scores, judge_b_scores, label):
    """Operational positive rates per domain on a list of trials."""
    strict = label == "strict"
    out = {}
    for d in DOMAINS:
        pos = tot = 0
        for t in support:
            ra = PRIMARY_A[t]
            if ra["domain"] != d:
                continue
            a = psi(ra) == 1.0 if strict else psi(ra) >= 0.5
            b = psi(judge_b_scores[t]) == 1.0 if strict else psi(judge_b_scores[t]) >= 0.5
            tot += 1
            pos += a and b
        out[d] = 100.0 * pos / tot
    return out


def margin(rates):
    return rates["UNREG"] - max(rates["REG-H"], rates["REG-S"])


def bootstrap(labels_by_probe):
    """Stratified base-probe cluster bootstrap for the Table 2 intervals.

    labels_by_probe[d][p] = np.array of per-response columns
        [a_cand, b_cand, a_strict, b_strict]
    for probe p in domain d (all dual-scored responses of that probe).
    Each replicate draws 15 of the 15 probes with replacement per domain,
    keeping the eight answer models fixed.
    """
    rng = np.random.default_rng(SEED)
    probe_lists = {d: sorted(labels_by_probe[d]) for d in DOMAINS}
    lo = {k: np.zeros(N_BOOT) for k in ("S1", "S2", "S3", "S3_H-U")}
    for i in range(N_BOOT):
        samples = {}
        for d in DOMAINS:
            probes = probe_lists[d]
            idx = rng.integers(0, len(probes), size=len(probes))
            samples[d] = np.vstack([labels_by_probe[d][probes[j]] for j in idx])
        r1 = {d: 100.0 * samples[d][:, 0].mean() for d in DOMAINS}
        r2 = {d: 100.0 * (samples[d][:, 0] & samples[d][:, 1]).mean() for d in DOMAINS}
        r3 = {d: 100.0 * (samples[d][:, 2] & samples[d][:, 3]).mean() for d in DOMAINS}
        lo["S1"][i] = margin(r1)
        lo["S2"][i] = margin(r2)
        lo["S3"][i] = margin(r3)
        lo["S3_H-U"][i] = r3["REG-H"] - r3["UNREG"]
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in lo.items()}


def cross_judge_row(judge_a, judge_b):
    """Candidate-label P_o per professional domain over the shared valid trials.

    Works for both cross-judge pairs (two judges, same round) and
    repeat-generation checks (same judge, R3 vs. R4 -- R4 reuses R3 trial ids,
    one regenerated response per cell).  n counts all four domains; the
    percentage columns cover the three professional domains.
    """
    shared = sorted(set(judge_a) & set(judge_b))
    per_domain, tot = Counter(), Counter()
    for t in shared:
        d = judge_a[t]["domain"]
        if d not in DOMAINS:
            continue
        per_domain[d] += (psi(judge_a[t]) >= 0.5) == (psi(judge_b[t]) >= 0.5)
        tot[d] += 1
    row = tuple(100.0 * per_domain[d] / tot[d] for d in DOMAINS)
    return len(shared), row


def kappa_from_counts(c):
    """Cohen's kappa from a (n11, n10, n01, n00) binary-agreement table."""
    n11, n10, n01, n00 = c
    n = sum(c)
    po = (n11 + n00) / n
    pe = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cohen_kappa(a, b):
    levels = sorted(set(a) | set(b))
    n = len(a)
    obs = Counter(zip(a, b))
    po = sum(obs[(l, l)] for l in levels) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in levels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load_human_raters():
    """Five anonymous raters (A1-A5) on the 36-item anchor set."""
    raw = json.loads(HUMAN.read_text(encoding="utf-8"))
    return {r: raw[r]["anchor36"] for r in ("A1", "A2", "A3", "A4", "A5")}


def bvi_of_dims(dims):
    if min(dims["D1"], dims["D2"], dims["D3"]) < 0:
        return None  # rater skipped a dimension -- record not fully scored
    rec = {"domain": dims["domain"], "D1": dims["D1"], "D2": dims["D2"], "D3": dims["D3"]}
    return psi(rec)


def categorical_majority(vals):
    """Five-rater categorical majority; 2-2-1 ties fall back to 0.5."""
    c = Counter(vals)
    if len(c) > 1 and c.most_common(1)[0][1] == c.most_common(2)[1][1]:
        return 0.5, True
    return c.most_common(1)[0][0], False


def missing_score_bounds(ds, cl, overlap, check):
    """Worst-case candidate-rate bounds over the original 1,080 responses/domain.

    S1: lower bound scores all omitted responses negative; upper bound positive.
    S2: lower bound is the joint-positive rate on common support; the upper
    bound additionally counts every response outside common support as jointly
    positive (deliberately infeasible -- it is a bound, not an estimate).
    """
    raw = json.loads((SCORES / "r3_judge_deepseek.json").read_text(encoding="utf-8"))
    for rule in ("S1", "S2"):
        for d in DOMAINS:
            allrecs = [r for r in raw if r["domain"] == d]
            total = len(allrecs)
            if rule == "S1":
                valid = [r for r in allrecs if not r.get("parse_error")]
                pos = sum(1 for r in valid if psi(r) >= 0.5)
                omitted = total - len(valid)
            else:
                common = [t for t in overlap if ds[t]["domain"] == d]
                pos = sum(1 for t in common if psi(ds[t]) >= 0.5 and psi(cl[t]) >= 0.5)
                omitted = total - len(common)
            lo, hi = 100.0 * pos / total, 100.0 * (pos + omitted) / total
            check(f"missing {rule} {d}", (round(lo, 1), round(hi, 1)),
                  EXPECTED["missing_bounds"][(rule, d)])


def prevalence_matched(ds, cl, overlap, check):
    """Probe-level prevalence-matched concordance (Appendix).

    Mirrors prevalence_matched.py exactly (bin assignment, draw order, seed
    20260728) so the percentile intervals reproduce bit-for-bit.
    """
    doms = ["REG-S", "REG-H", "UNREG"]  # original script's domain order
    bins = [(-1e-9, 1e-9), (1e-9, 0.20), (0.20, 0.40), (0.40, 1.01)]
    probe_flags = defaultdict(list)
    for t in overlap:
        r = ds[t]
        if r["domain"] not in doms:
            continue
        probe_flags[(r["domain"], r["probe_id"])].append(
            (int(psi(r) >= 0.5), int(psi(cl[t]) >= 0.5)))
    rate = {k: float(np.mean([a + b for a, b in v]) / 2.0) for k, v in probe_flags.items()}
    bin_probes = {bi: {d: [] for d in doms} for bi in range(len(bins))}
    for (d, p), r in rate.items():
        for bi, (lo, hi) in enumerate(bins):
            if lo < r <= hi or (bi == 0 and r == 0):
                bin_probes[bi][d].append(p)
                break
    rng = np.random.default_rng(PREV_SEED)
    po = {d: [] for d in doms}
    ka = {d: [] for d in doms}
    mr = {d: [] for d in doms}
    for _ in range(PREV_DRAWS):
        sel = []
        for bi in range(len(bins)):
            counts = {d: len(bin_probes[bi][d]) for d in doms}
            m_take = min(counts.values())
            if m_take == 0:
                continue
            for d in doms:
                chosen = rng.choice(bin_probes[bi][d], size=m_take, replace=False)
                sel.extend((d, str(p)) for p in chosen)
        selset = set(sel)
        for d in doms:
            a = [int(psi(ds[t]) >= 0.5) for t in overlap
                 if ds[t]["domain"] == d and (ds[t]["domain"], ds[t]["probe_id"]) in selset]
            b = [int(psi(cl[t]) >= 0.5) for t in overlap
                 if ds[t]["domain"] == d and (ds[t]["domain"], ds[t]["probe_id"]) in selset]
            n = len(a)
            agree = sum(1 for x, y in zip(a, b) if x == y)
            po[d].append(100.0 * agree / n)
            ca, cb = Counter(a), Counter(b)
            pe = sum(ca[l] * cb[l] for l in (0, 1)) / (n * n)
            ka[d].append((agree / n - pe) / (1 - pe) if pe < 1 else 1.0)
            mr[d].append(100.0 * np.mean([rate[k] for k in sel if k[0] == d]))
    for d in doms:
        check(f"prevalence matched_rate {d}", round(float(np.median(mr[d])), 1),
              EXPECTED["prevalence"]["matched_rate"][d])
        check(f"prevalence Po_median {d}", round(float(np.median(po[d])), 1),
              EXPECTED["prevalence"]["Po_median"][d])
        lo, hi = (round(float(np.percentile(po[d], q)), 1) for q in (2.5, 97.5))
        check(f"prevalence Po_ci {d}", (lo, hi), EXPECTED["prevalence"]["Po_ci"][d], 0.15)
        check(f"prevalence kappa_median {d}", round(float(np.median(ka[d])), 2),
              EXPECTED["prevalence"]["kappa_median"][d], 0.011)


def migration_audit(check):
    """Claim-to-evidence audit rerun on the REG-L/REG-N migration benchmark."""
    def load(path):
        d = json.loads(path.read_text(encoding="utf-8"))
        return {(r["probe_id"], r.get("framing", ""), r.get("role_line", ""), r["model_id"]): r
                for r in d if r.get("D1", -1) >= 0}

    ds, ki = load(MIG_DS), load(MIG_KI)
    keys = sorted(set(ds) & set(ki))
    doms = ["REG-L", "REG-N"]
    dom_of = lambda k: ds[k]["domain"]

    def flag(k, rule):
        if rule in ("S1", "cand"):
            return int(ds[k]["BVI"] >= 0.5) if rule == "cand" else int(psi(ds[k]) >= 0.5)
        if rule in ("S2", "joint_cand"):
            return int(psi(ds[k]) >= 0.5 and psi(ki[k]) >= 0.5)
        return int(psi(ds[k]) == 1.0 and psi(ki[k]) == 1.0)

    for lab in ("cand", "strict"):
        for d in doms:
            ks = [k for k in keys if dom_of(k) == d]
            a = [int(ds[k]["BVI"] >= 0.5 if lab == "cand" else ds[k]["BVI"] == 1) for k in ks]
            b = [int(ki[k]["BVI"] >= 0.5 if lab == "cand" else ki[k]["BVI"] == 1) for k in ks]
            n11 = sum(1 for x, y in zip(a, b) if x and y)
            n10 = sum(1 for x, y in zip(a, b) if x and not y)
            n01 = sum(1 for x, y in zip(a, b) if y and not x)
            n = len(ks)
            po = 100.0 * (n11 + (n - n11 - n10 - n01)) / n
            pplus = 100.0 * 2 * n11 / (2 * n11 + n10 + n01)
            ca, cb = Counter(a), Counter(b)
            pe = sum(ca[l] * cb[l] for l in (0, 1)) / (n * n)
            k = ((po / 100 - pe) / (1 - pe)) if pe < 1 else 1.0
            if lab == "cand":
                check(f"migration Po {d}", round(po, 1), EXPECTED["migration"]["Po"][d])
                check(f"migration P+ {d}", round(pplus, 1), EXPECTED["migration"]["Pplus"][d])
                check(f"migration kappa {d}", round(k, 2), EXPECTED["migration"]["kappa"][d], 0.011)
    a = [int(ds[k]["BVI"] >= 0.5) for k in keys]
    b = [int(ki[k]["BVI"] >= 0.5) for k in keys]
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in (0, 1)) / (len(a) * len(b))
    agree = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    check("migration pool_kappa", round((agree - pe) / (1 - pe), 2),
          EXPECTED["migration"]["pool_kappa"], 0.011)

    probes = {d: sorted({k[0] for k in keys if dom_of(k) == d}) for d in doms}
    by_probe = defaultdict(list)
    for k in keys:
        for rule in ("S1", "S2", "S3"):
            by_probe[(rule, dom_of(k), k[0])].append(flag(k, rule))
    rng = np.random.default_rng(MIG_SEED)
    ci = {r: [] for r in ("S1", "S2", "S3")}
    for _ in range(MIG_DRAWS):
        rs = {}
        for d in doms:
            chosen = rng.choice(probes[d], size=len(probes[d]), replace=True)
            for rule in ("S1", "S2", "S3"):
                vals = [v for p in chosen for v in by_probe[(rule, d, p)]]
                rs.setdefault(rule, {})[d] = 100.0 * float(np.mean(vals))
        for rule in ("S1", "S2", "S3"):
            ci[rule].append(rs[rule]["REG-N"] - rs[rule]["REG-L"])
    for rule in ("S1", "S2", "S3"):
        rates = {}
        for d in doms:
            ks = [k for k in keys if dom_of(k) == d]
            rates[d] = 100.0 * sum(flag(k, rule) for k in ks) / len(ks)
        check(f"migration rates {rule}", (round(rates["REG-L"], 1), round(rates["REG-N"], 1)),
              EXPECTED["migration"]["rates"][rule])
        check(f"migration margin {rule}", round(rates["REG-N"] - rates["REG-L"], 1),
              EXPECTED["migration"]["margins"][rule])
        lo, hi = (round(float(np.percentile(ci[rule], q)), 1) for q in (2.5, 97.5))
        check(f"migration ci {rule}", (lo, hi), EXPECTED["migration"]["ci"][rule], 0.15)
    pm = []
    for m in sorted({k[3] for k in keys}):
        ks = [k for k in keys if k[3] == m]
        for rule in ("S1", "S2"):
            r = {d: 100.0 * sum(flag(k, rule) for k in ks if dom_of(k) == d) /
                    sum(1 for k in ks if dom_of(k) == d) for d in doms}
            pm.append(r["REG-N"] - r["REG-L"])
    check("migration per_model range", (round(min(pm), 1), round(max(pm), 1)),
          (EXPECTED["migration"]["per_model_min"], EXPECTED["migration"]["per_model_max"]), 0.15)


def main():
    global OVERLAP, PRIMARY_A, PRIMARY_B
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="check output against submission values")
    args = ap.parse_args()

    failures = []

    def check(name, got, want, tol=0.051):
        ok = all(abs(g - w) <= tol for g, w in zip(got, want)) if isinstance(got, tuple) else abs(got - want) <= tol
        if args.verify:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, paper {want}")
            if not ok:
                failures.append(name)
        return got

    # ------------------------------------------------------------------
    print("== Load scores and build the primary-pair overlap ==")
    ds = load_judge(SCORES / "r3_judge_deepseek.json")
    cl = load_judge(SCORES / "r3_judge_claude.json")
    qw = load_judge(SCORES / "r3_judge_qwen.json")
    print(f"  valid scores: DeepSeek {len(ds)}, Claude {len(cl)}, Qwen {len(qw)}")
    check("ds_valid", len(ds), EXPECTED["ds_valid"], 0)
    check("cl_valid", len(cl), EXPECTED["cl_valid"], 0)
    OVERLAP = sorted(set(ds) & set(cl))
    PRIMARY_A, PRIMARY_B = ds, cl
    check("overlap", len(OVERLAP), EXPECTED["overlap"], 0)

    mismatch = sum(1 for t in OVERLAP if abs(psi(ds[t]) - ds[t]["BVI"]) > 1e-9)
    print(f"  recomputed PSI vs. stored BVI mismatches on overlap: {mismatch}")

    # ------------------------------------------------------------------
    print("\n== Table 1 / Appendix C: primary-pair concordance ==")
    counts = {}
    for label in ("candidate", "strict"):
        counts[label] = agreement_2x2(ds, cl, label)
        for d in DOMAINS:
            c = counts[label][d]
            check(f"2x2 {label} {d}", c, EXPECTED["2x2"][(label, d)], 0)
            po, pp = po_pplus(c)
            check(f"P_o {label} {d}", round(po, 1), EXPECTED["Po"][(label, d)])
            check(f"P_+ {label} {d}", round(pp, 1), EXPECTED["P+"][(label, d)])

    # Pooled numbers, including the CTL stratum
    cand_total = [counts["candidate"][d] for d in DOMAINS]
    n_prof = sum(sum(c) for c in cand_total)
    agree_prof = sum(c[0] + c[3] for c in cand_total)
    ctl = [t for t in OVERLAP if ds[t]["domain"] == "CTL"]
    ctl_agree = sum((psi(ds[t]) >= 0.5) == (psi(cl[t]) >= 0.5) for t in ctl)
    ctl_pos = sum(psi(ds[t]) >= 0.5 or psi(cl[t]) >= 0.5 for t in ctl)
    pooled_all = 100.0 * (agree_prof + ctl_agree) / (n_prof + len(ctl))
    print(f"  CTL: n={len(ctl)}, agreement={ctl_agree} ({100.0 * ctl_agree / len(ctl):.1f}%), positive calls={ctl_pos}")
    check("pooled_prof", round(100.0 * agree_prof / n_prof, 1), EXPECTED["pooled_prof"])
    check("pooled_all", round(pooled_all, 1), EXPECTED["pooled_all"])
    for label in ("candidate", "strict"):
        for d in DOMAINS:
            check(f"kappa {label} {d}", round(kappa_from_counts(counts[label][d]), 2),
                  EXPECTED["kappa"][(label, d)], 0.011)

    # ------------------------------------------------------------------
    print("\n== Table 2: S1/S2/S3 rates and margins on common support ==")
    rule_rates, rule_margins = {}, {}
    for rule, label, joint in (("S1", "candidate", False), ("S2", "candidate", True), ("S3", "strict", True)):
        r = {}
        for d in DOMAINS:
            pos = tot = 0
            for t in OVERLAP:
                ra = ds[t]
                if ra["domain"] != d:
                    continue
                a = psi(ra) == 1.0 if label == "strict" else psi(ra) >= 0.5
                b = psi(cl[t]) == 1.0 if label == "strict" else psi(cl[t]) >= 0.5
                flag = (a and b) if joint else a
                tot += 1
                pos += flag
            r[d] = 100.0 * pos / tot
        rule_rates[rule] = r
        rule_margins[rule] = margin(r)
        for d in DOMAINS:
            check(f"rate {rule} {d}", round(r[d], 1), EXPECTED["rates"][(rule, d)])
        check(f"margin {rule}", round(rule_margins[rule], 1), EXPECTED["margins"][rule])

    # All-valid single-judge drift check (Section 4.3)
    all_valid = {}
    for d in DOMAINS:
        recs = [r for r in ds.values() if r["domain"] == d]
        all_valid[d] = 100.0 * sum(psi(r) >= 0.5 for r in recs) / len(recs)
    drift = max(abs(all_valid[d] - rule_rates["S1"][d]) for d in DOMAINS)
    print(f"  all-valid DeepSeek rates: {', '.join(f'{d} {all_valid[d]:.1f}' for d in DOMAINS)}; max drift vs S1 = {drift:.1f} pp")
    for d in DOMAINS:
        check(f"all_valid {d}", round(all_valid[d], 1), EXPECTED["all_valid_rates"][d])

    # ------------------------------------------------------------------
    print("\n== Appendix D: model-wise margins ==")
    model_labels = sorted({ds[t]["model_label"] for t in OVERLAP})
    models_pos = {rule: 0 for rule in ("S1", "S2", "S3")}
    for m in model_labels:
        supp = [t for t in OVERLAP if ds[t]["model_label"] == m]
        cells = {}
        for rule, label, joint in (("S1", "candidate", False), ("S2", "candidate", True), ("S3", "strict", True)):
            r = {}
            for d in DOMAINS:
                pos = tot = 0
                for t in supp:
                    ra = ds[t]
                    if ra["domain"] != d:
                        continue
                    a = psi(ra) == 1.0 if label == "strict" else psi(ra) >= 0.5
                    b = psi(cl[t]) == 1.0 if label == "strict" else psi(cl[t]) >= 0.5
                    tot += 1
                    pos += (a and b) if joint else a
                r[d] = 100.0 * pos / tot
            cells[rule] = r
        hu = cells["S3"]["REG-H"] - cells["S3"]["UNREG"]
        print(f"  {m:18s} m_S1={margin(cells['S1']):6.1f}  m_S2={margin(cells['S2']):6.1f}  "
              f"m_S3={margin(cells['S3']):6.1f}  S3 H-U={hu:6.1f}  (n={len(supp)})")
        for rule in models_pos:
            models_pos[rule] += margin(cells[rule]) > 0
    for rule in models_pos:
        check(f"models_positive {rule}", models_pos[rule], EXPECTED["models_positive"][rule], 0)

    # ------------------------------------------------------------------
    print("\n== Appendix B: cross-judge and repeat-generation configurations ==")
    kimi = load_judge(SCORES / "r4_judge_kimi_run1.json")  # temperature-1 scorer run used in the paper
    r4_ds = load_judge(SCORES / "r4_judge_deepseek.json")
    r4_qw = load_judge(SCORES / "r4_judge_qwen.json")
    configs = [
        ("R3 DeepSeek x Claude", ds, cl),
        ("R3 Qwen x Claude", qw, cl),
        ("R3 DeepSeek x Qwen", ds, qw),
        ("R4 DeepSeek x Kimi", r4_ds, kimi),
        ("R4 Qwen x Kimi", r4_qw, kimi),
        ("R4 Qwen x DeepSeek", r4_qw, r4_ds),
    ]
    for name, ja, jb in configs:
        n, row = cross_judge_row(ja, jb)
        print(f"  {name:24s} n={n:5d}  REG-H={row[0]:.1f}  REG-S={row[1]:.1f}  UNREG={row[2]:.1f}")
        check(name + " n", n, EXPECTED["config_n"][name], 0)
        check(name, tuple(round(v, 1) for v in row), EXPECTED["cross_judge"][name])
    for name, ja, jb in (("repeat DeepSeek R3->R4", ds, r4_ds), ("repeat Qwen R3->R4", qw, r4_qw)):
        n, row = cross_judge_row(ja, jb)
        print(f"  {name:24s} n={n:5d}  REG-H={row[0]:.1f}  REG-S={row[1]:.1f}  UNREG={row[2]:.1f}")
        check(name + " n", n, EXPECTED["config_n"][name], 0)
        check(name, tuple(round(v, 1) for v in row), EXPECTED["cross_judge"][name])

    # ------------------------------------------------------------------
    print(f"\n== Table 2 intervals: stratified base-probe bootstrap (B={N_BOOT}, seed {SEED}) ==")
    labels_by_probe = {d: {} for d in DOMAINS}
    for t in OVERLAP:
        ra = ds[t]
        d = ra["domain"]
        if d not in DOMAINS:
            continue
        labels_by_probe[d].setdefault(ra["probe_id"], []).append(
            [psi(ra) >= 0.5, psi(cl[t]) >= 0.5, psi(ra) == 1.0, psi(cl[t]) == 1.0]
        )
    for d in DOMAINS:
        print(f"  {d}: {len(labels_by_probe[d])} base probes, "
              f"{sum(len(v) for v in labels_by_probe[d].values())} dual-scored responses")
    intervals = bootstrap(labels_by_probe)
    for rule in ("S1", "S2", "S3"):
        lo, hi = intervals[rule]
        lo, hi = round(lo, 1), round(hi, 1)
        print(f"  m_{rule}: [{lo}, {hi}]")
        check(f"bootstrap {rule}", (lo, hi), EXPECTED["bootstrap"][rule])
    lo, hi = intervals["S3_H-U"]
    print(f"  S3 REG-H - UNREG: [{round(lo, 1)}, {round(hi, 1)}]")
    check("bootstrap S3 H-U", (round(lo, 1), round(hi, 1)), EXPECTED["bootstrap_S3_HminusU"])

    # ------------------------------------------------------------------
    print("\n== Missing-score bounds ==")
    missing_score_bounds(ds, cl, OVERLAP, check)

    # ------------------------------------------------------------------
    print("\n== Prevalence-matched reanalysis ==")
    prevalence_matched(ds, cl, OVERLAP, check)

    # ------------------------------------------------------------------
    if HUMAN.exists():
        print("\n== Human anchor: rater agreement, matched-29, dimension kappas ==")
        raters = load_human_raters()
        names = ["A1", "A2", "A3", "A4", "A5"]

        def cell_of(itemkey):
            probe, framing, role, model = itemkey.split("|")
            return (probe, framing, role, model)

        ds_cell = {}
        for t in ds:
            r = ds[t]
            ds_cell[(r["probe_id"], r.get("framing", ""), r.get("role_line", ""), r["model_id"])] = t
        cl_cell = {}
        for t in cl:
            r = cl[t]
            cl_cell[(r["probe_id"], r.get("framing", ""), r.get("role_line", ""), r["model_id"])] = t

        def judge_trial(J, itemkey):
            return (ds_cell if J is ds else cl_cell)[cell_of(itemkey)]

        for i in range(5):
            for j in range(i + 1, 5):
                a, b = names[i], names[j]
                common = sorted(set(raters[a]) & set(raters[b]))
                pairs = [(bvi_of_dims(raters[a][k]), bvi_of_dims(raters[b][k]))
                         for k in common]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                va = [x for x, _ in pairs]
                vb = [y for _, y in pairs]
                po = 100.0 * sum(1 for x, y in zip(va, vb) if x == y) / len(pairs)
                k3 = cohen_kappa(va, vb)
                pc = 100.0 * sum(1 for x, y in zip(va, vb) if (x >= 0.5) == (y >= 0.5)) / len(pairs)
                pst = 100.0 * sum(1 for x, y in zip(va, vb) if (x == 1) == (y == 1)) / len(pairs)
                check(f"human pair {a}--{b}",
                      (round(po, 1), round(k3, 2), round(pc, 1), round(pst, 1)),
                      EXPECTED["human_pairs"][f"{a}--{b}"])

        inter5 = sorted(set.intersection(*[set(raters[n]) for n in names]))
        c31 = [k for k in inter5
               if all(bvi_of_dims(raters[n][k]) is not None for n in names)
               and cell_of(k) in ds_cell]
        c29 = [k for k in c31 if cell_of(k) in cl_cell]
        maj_psi = {k: categorical_majority([bvi_of_dims(raters[n][k]) for n in names])[0]
                   for k in c31}
        vj31 = [psi(ds[ds_cell[cell_of(k)]]) for k in c31]
        vh31 = [maj_psi[k] for k in c31]
        po31 = 100.0 * sum(1 for x, y in zip(vj31, vh31) if x == y) / len(c31)
        check("human ds_vs_maj_31", (round(po31, 1), round(cohen_kappa(vj31, vh31), 2)),
              EXPECTED["human"]["ds_vs_maj_31"])

        vh29 = [maj_psi[k] for k in c29]
        for jname, J in (("DeepSeek", ds), ("Claude-Haiku", cl)):
            vj = [psi(J[judge_trial(J, k)]) for k in c29]
            jc = [int(v >= 0.5) for v in vj]
            js = [int(v == 1) for v in vj]
            hc = [int(v >= 0.5) for v in vh29]
            hs = [int(v == 1) for v in vh29]
            row = (round(100.0 * sum(1 for x, y in zip(vj, vh29) if x == y) / len(c29), 1),
                   round(cohen_kappa(vj, vh29), 2),
                   round(100.0 * sum(1 for x, y in zip(jc, hc) if x == y) / len(c29), 1),
                   round(100.0 * sum(1 for x, y in zip(js, hs) if x == y) / len(c29), 1))
            check(f"human judge row {jname}", row, EXPECTED["human"]["judge_rows"][jname])

        vds3 = [psi(ds[ds_cell[cell_of(k)]]) for k in c29]
        vcl3 = [psi(cl[cl_cell[cell_of(k)]]) for k in c29]
        for lab, f in (("PSI", lambda v: v), ("cand", lambda v: [int(x >= 0.5) for x in v]),
                       ("strict", lambda v: [int(x == 1) for x in v])):
            a3, b3, c3 = f(vds3), f(vcl3), f(vh29)
            got = (round(100.0 * sum(1 for x, y in zip(a3, c3) if x == y) / len(c29), 1),
                   round(cohen_kappa(a3, c3), 2),
                   round(100.0 * sum(1 for x, y in zip(b3, c3) if x == y) / len(c29), 1),
                   round(cohen_kappa(b3, c3), 2),
                   round(100.0 * sum(1 for x, y in zip(a3, b3) if x == y) / len(c29), 1),
                   round(cohen_kappa(a3, b3), 2))
            check(f"matched29 {lab}", got, EXPECTED["human"]["matched29"][lab])

        for dom in ("REG-H", "REG-S", "UNREG", "CTL"):
            ks_d = [k for k in c29 if raters[names[0]][k]["domain"] == dom]
            jc = round(100.0 * sum(1 for k in ks_d if psi(ds[ds_cell[cell_of(k)]]) >= 0.5) / len(ks_d), 1)
            js = round(100.0 * sum(1 for k in ks_d if psi(ds[ds_cell[cell_of(k)]]) == 1) / len(ks_d), 1)
            hc = round(100.0 * sum(1 for k in ks_d if maj_psi[k] >= 0.5) / len(ks_d), 1)
            hst = round(100.0 * sum(1 for k in ks_d if maj_psi[k] == 1) / len(ks_d), 1)
            check(f"anchor rates {dom}", (jc, js, hc, hst),
                  EXPECTED["human"]["anchor_rates"][dom])

        for jname, J in (("DeepSeek", ds), ("Claude-Haiku", cl)):
            for di, d in enumerate(("D1", "D2", "D3")):
                vj, vh = [], []
                for k in c29:
                    v, tied = categorical_majority([raters[n][k][d] for n in names])
                    if tied:
                        continue
                    vj.append(J[judge_trial(J, k)][d])
                    vh.append(v)
                check(f"dim kappa {jname} {d}", round(cohen_kappa(vj, vh), 2),
                      EXPECTED["human"]["dim_kappa"][jname][di], 0.011)
    else:
        print("\n[skip] human anchor: data/human/annotations_anonymized.json not found")

    # ------------------------------------------------------------------
    if MIG_DS.exists() and MIG_KI.exists():
        print("\n== Migration benchmark (REG-L / REG-N) ==")
        migration_audit(check)
    else:
        print("\n[skip] migration benchmark: data/migration/ not found")

    # ------------------------------------------------------------------
    if args.verify:
        print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
