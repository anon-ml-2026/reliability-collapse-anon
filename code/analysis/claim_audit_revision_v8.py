"""Revision-v8 additions to the claim-to-evidence audit (paperB_v8.tex).

All analyses here were added during the 2026-09 revision in response to
reviews; they are EXPLORATORY (chosen after the original analysis) and run
on the same frozen R3 primary-pair common support (4,027 responses) as
claim_matched_audit.py, whose loader and PSI rules this module reuses.

  A. Judge-rule x threshold specification matrix
     2 thresholds (candidate, strict) x 4 judge rules (DeepSeek-only,
     Claude-only, intersection, union) = 8 specifications, each with domain
     rates, UNREG-highest margin, a stratified probe-cluster bootstrap
     interval, and the count of answer models with a positive margin.
  B. Leave-one-probe-out influence on the S1/S2 margins (45 base probes).
  C. Probe-count sensitivity: bootstrap CI width of the S1 margin with
     k = 5/10/15 probes per domain.
  D. Human anchor: Wilson intervals for the matched-29 agreement rates,
     item-bootstrap intervals for the paired agreement differences,
     tied-vote denominators for dimension-level kappa, and a
     threshold-then-majority vs majority-then-threshold sensitivity check.
  E. domain x judge fixed-effect logistic regression on the candidate label
     (probe-clustered standard errors, hand-rolled IRLS + sandwich).

New seeds, declared here (the main-text seed 20260825 is untouched):
  SEED_MATRIX = 20260912  (specification-matrix bootstrap)
  SEED_COUNT  = 20260913  (probe-count sensitivity)
  SEED_HUMAN  = 20260914  (item bootstrap for paired anchor differences)

Run from the repository root:
    python code/analysis/claim_audit_revision_v8.py           # report
    python code/analysis/claim_audit_revision_v8.py --verify  # + PASS/FAIL
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from claim_matched_audit import (
    DOMAINS, SCORES, HUMAN, load_judge, psi, margin, cohen_kappa,
    categorical_majority, bvi_of_dims, load_human_raters,
)

SEED_MATRIX = 20260912
SEED_COUNT = 20260913
SEED_HUMAN = 20260914
N_BOOT = 20_000

# Specification keys: (threshold, rule).  S1/S2/S3 from the paper are
# (cand, DS), (cand, AND), (strict, AND) respectively.
SPECS = [
    ("cand", "DS"), ("cand", "CL"), ("cand", "AND"), ("cand", "OR"),
    ("strict", "DS"), ("strict", "CL"), ("strict", "AND"), ("strict", "OR"),
]

EXPECTED = {
    "matrix": {
        # (threshold, rule): (REG-H, REG-S, UNREG, margin,
        #                     ci_lo, ci_hi, models_positive)
        # All eight cells use SEED_MATRIX (20260912); the main-text Table 2
        # intervals for S1/S2/S3 keep the original seed 20260825, whose
        # values differ from these by a few tenths of a point only.
        ("cand", "DS"):   (21.1, 5.2, 40.1, 19.0, 1.6, 37.1, 8),
        ("cand", "CL"):   (19.9, 14.3, 35.9, 16.0, 1.2, 30.0, 8),
        ("cand", "AND"):  (8.7, 3.8, 19.9, 11.2, 0.5, 22.8, 8),
        ("cand", "OR"):   (32.2, 15.6, 56.1, 23.9, 5.3, 42.0, 8),
        ("strict", "DS"): (15.0, 3.7, 29.0, 14.0, 0.5, 27.5, 8),
        ("strict", "CL"): (15.2, 13.1, 11.5, -3.7, -13.3, 5.6, 1),
        ("strict", "AND"): (6.0, 2.9, 3.0, -3.0, -7.0, 1.4, 1),
        ("strict", "OR"): (24.2, 14.0, 37.5, 13.3, -0.9, 27.0, 7),
    },
    "loo": {
        # rule: (min_margin, argmin domain, argmin probe, sign flips of 45)
        "S1": (14.8, "UNREG", "UNREG-12", 0),
        "S2": (7.7, "UNREG", "UNREG-08", 0),
    },
    "probe_count": {
        # k: median CI width of the S1 margin (pp)
        5: 61.9, 10: 43.6, 15: 36.0,
    },
    "human_wilson": {
        # rep: (rate, lo, hi) per pair, paper Appendix values
        "PSI":  {"DS-hum": (62.1, 44.0, 77.3), "CL-hum": (69.0, 50.8, 82.7),
                 "DS-CL": (69.0, 50.8, 82.7)},
        "cand": {"DS-hum": (75.9, 57.9, 87.8), "CL-hum": (75.9, 57.9, 87.8),
                 "DS-CL": (72.4, 54.3, 85.3)},
        "strict": {"DS-hum": (82.8, 65.5, 92.4), "CL-hum": (82.8, 65.5, 92.4),
                   "DS-CL": (86.2, 69.4, 94.5)},
    },
    "human_paired_diff": {
        # (rep, pairA, pairB): (diff_pp, n_A-only, n_B-only, boot_lo, boot_hi)
        ("cand", "DS-hum", "DS-CL"): (3.4, 4, 3, -13.8, 20.7),
        ("strict", "DS-CL", "DS-hum"): (3.4, 3, 2, -10.3, 17.2),
        ("PSI", "DS-CL", "DS-hum"): (6.9, 4, 2, -10.3, 24.1),
    },
    "dim_tied": {"D1": (2, 27), "D2": (2, 27), "D3": (4, 25)},
    "majority_order": {
        "flips": (0, 0),
        "alt_rates": {("cand", "DS"): 75.9, ("cand", "CL"): 75.9,
                      ("strict", "DS"): 82.8, ("strict", "CL"): 82.8},
    },
    "logit": {
        # term: (coef, cluster-SE, z, p)
        "intercept": (-1.320, 0.286, -4.62, 0.0000),
        "REG-S": (-1.588, 0.406, -3.91, 0.0001),
        "UNREG": (0.917, 0.434, 2.11, 0.0347),
        "Claude": (-0.075, 0.320, -0.23, 0.8155),
        "REG-S:Claude": (1.191, 0.391, 3.04, 0.0024),
        "UNREG:Claude": (-0.102, 0.433, -0.23, 0.8143),
        "gaps": {"REG-H": -1.2, "REG-S": 9.1, "UNREG": -4.2},
    },
}


# ----------------------------------------------------------------------
# Shared loading
# ----------------------------------------------------------------------

def load_primary():
    ds = load_judge(SCORES / "r3_judge_deepseek.json")
    cl = load_judge(SCORES / "r3_judge_claude.json")
    overlap = sorted(set(ds) & set(cl))
    return ds, cl, overlap


def probe_labels(ds, cl, overlap):
    """labels[d][probe_id] = list of [a_cand, b_cand, a_str, b_str] bools."""
    lab = {d: {} for d in DOMAINS}
    for t in overlap:
        r = ds[t]
        d = r["domain"]
        if d not in DOMAINS:
            continue
        lab[d].setdefault(r["probe_id"], []).append(
            [psi(r) >= 0.5, psi(cl[t]) >= 0.5, psi(r) == 1.0, psi(cl[t]) == 1.0]
        )
    return lab


def spec_columns(threshold, rule, arr):
    """Per-response positive indicator for one specification.

    arr is an (n, 4) boolean array [a_cand, b_cand, a_str, b_str].
    """
    col = (0, 2)[threshold == "strict"]
    a, b = arr[:, col], arr[:, col + 1]
    if rule == "DS":
        return a
    if rule == "CL":
        return b
    if rule == "AND":
        return a & b
    return a | b


def spec_rates_on(labels, threshold, rule):
    rates = {}
    for d in DOMAINS:
        vals = np.concatenate([spec_columns(threshold, rule, np.array(v, dtype=bool))
                               for v in labels[d].values()])
        rates[d] = 100.0 * vals.mean()
    return rates


# ----------------------------------------------------------------------
# A. Specification matrix
# ----------------------------------------------------------------------

def spec_matrix(labels, ds, cl, overlap, check, verify):
    print("== A. Specification matrix (2 thresholds x 4 judge rules) ==")
    model_labels = sorted({ds[t]["model_label"] for t in overlap})
    by_model = {m: probe_labels(
        {t: v for t, v in ds.items() if v["model_label"] == m},
        {t: v for t, v in cl.items() if v["model_label"] == m},
        [t for t in overlap if ds[t]["model_label"] == m]) for m in model_labels}

    rng = np.random.default_rng(SEED_MATRIX)
    probe_lists = {d: sorted(labels[d]) for d in DOMAINS}
    boot = {s: np.zeros(N_BOOT) for s in SPECS}
    for i in range(N_BOOT):
        stacked = {}
        for d in DOMAINS:
            ps = probe_lists[d]
            idx = rng.integers(0, len(ps), size=len(ps))
            stacked[d] = np.vstack([np.array(labels[d][ps[j]], dtype=bool) for j in idx])
        for s in SPECS:
            r = {d: 100.0 * spec_columns(s[0], s[1], stacked[d]).mean() for d in DOMAINS}
            boot[s][i] = margin(r)

    out = {}
    for s in SPECS:
        rates = spec_rates_on(labels, s[0], s[1])
        m = margin(rates)
        lo, hi = np.percentile(boot[s], 2.5), np.percentile(boot[s], 97.5)
        models_pos = 0
        for mlab in model_labels:
            r = spec_rates_on(by_model[mlab], s[0], s[1])
            models_pos += margin(r) > 0
        out[s] = (rates, m, lo, hi, models_pos)
        row = (round(rates["REG-H"], 1), round(rates["REG-S"], 1),
               round(rates["UNREG"], 1), round(m, 1),
               round(lo, 1), round(hi, 1), models_pos)
        print(f"  {s[0]:6s} {s[1]:3s}  H={row[0]:5.1f} S={row[1]:5.1f} "
              f"U={row[2]:5.1f}  m={row[3]:6.1f}  CI=[{row[4]:.1f}, {row[5]:.1f}]  models+={models_pos}/8")
        if verify and s in EXPECTED["matrix"] and EXPECTED["matrix"][s][0]:
            check(f"matrix {s}", row, EXPECTED["matrix"][s])
    return out


# ----------------------------------------------------------------------
# B. Leave-one-probe-out influence
# ----------------------------------------------------------------------

def loo(labels, check, verify):
    print("== B. Leave-one-probe-out influence on the margins ==")
    out = {}
    for threshold, rule, name in (("cand", "DS", "S1"), ("cand", "AND", "S2")):
        rows = []
        for d in DOMAINS:
            for p in sorted(labels[d]):
                reduced = {dd: {pp: v for pp, v in labels[dd].items()}
                           for dd in DOMAINS}
                del reduced[d][p]
                rows.append((margin(spec_rates_on(reduced, threshold, rule)), d, p))
        rows.sort()
        flips = sum(1 for m, _, _ in rows if m <= 0)
        out[name] = rows
        print(f"  {name}: full-sample margin drops from "
              f"{margin(spec_rates_on(labels, threshold, rule)):.1f} to a minimum of "
              f"{rows[0][0]:.1f} when {rows[0][1]}.{rows[0][2]} is removed "
              f"(max {rows[-1][0]:.1f}); sign flips: {flips}/45")
        if verify and EXPECTED["loo"][name]:
            check(f"loo {name}", (round(rows[0][0], 1), rows[0][1], rows[0][2], flips),
                  EXPECTED["loo"][name])
    return out


# ----------------------------------------------------------------------
# C. Probe-count sensitivity
# ----------------------------------------------------------------------

def probe_count(labels, check, verify):
    print("== C. S1-margin CI width vs probes per domain (median of 20 runs x 2,000 draws) ==")
    rng = np.random.default_rng(SEED_COUNT)
    out = {}
    for k in (5, 10, 15):
        widths = []
        for _ in range(20):
            draws = []
            for _ in range(2_000):
                r = {}
                for d in DOMAINS:
                    ps = sorted(labels[d])
                    idx = rng.integers(0, len(ps), size=k)
                    stacked = np.vstack([np.array(labels[d][ps[j]], dtype=bool) for j in idx])
                    r[d] = 100.0 * spec_columns("cand", "DS", stacked).mean()
                draws.append(margin(r))
            lo, hi = np.percentile(draws, 2.5), np.percentile(draws, 97.5)
            widths.append(hi - lo)
        med = float(np.median(widths))
        out[k] = med
        print(f"  k={k:2d} probes/domain: median 95% CI width = {med:.1f} pp "
              f"(half-width ~{med / 2:.1f})")
        if verify and EXPECTED["probe_count"][k] is not None:
            check(f"probe_count {k}", round(med, 1), EXPECTED["probe_count"][k], 1.0)
    return out


# ----------------------------------------------------------------------
# D. Human anchor additions
# ----------------------------------------------------------------------

def wilson(k, n, z=1.959963984540054):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100.0 * (centre - half), 100.0 * (centre + half)


def human_anchor(ds, cl, overlap, check, verify):
    print("== D. Human anchor: Wilson CIs, paired differences, ties, majority order ==")
    raters = load_human_raters()
    names = ["A1", "A2", "A3", "A4", "A5"]

    def cell_of(itemkey):
        return tuple(itemkey.split("|"))

    ds_cell = {(r["probe_id"], r.get("framing", ""), r.get("role_line", ""), r["model_id"]): t
               for t, r in ds.items()}
    cl_cell = {(r["probe_id"], r.get("framing", ""), r.get("role_line", ""), r["model_id"]): t
               for t, r in cl.items()}

    inter5 = sorted(set.intersection(*[set(raters[n]) for n in names]))
    c31 = [k for k in inter5
           if all(bvi_of_dims(raters[n][k]) is not None for n in names)
           and cell_of(k) in ds_cell]
    c29 = [k for k in c31 if cell_of(k) in cl_cell]
    maj_psi = {k: categorical_majority([bvi_of_dims(raters[n][k]) for n in names])[0]
               for k in c31}
    print(f"  anchor items: 36 scored, {len(c31)} five-rater + DeepSeek, {len(c29)} triple intersection")

    vds = [psi(ds[ds_cell[cell_of(k)]]) for k in c29]
    vcl = [psi(cl[cl_cell[cell_of(k)]]) for k in c29]
    vhu = [maj_psi[k] for k in c29]
    reps = {"PSI": lambda v: v,
            "cand": lambda v: [int(x >= 0.5) for x in v],
            "strict": lambda v: [int(x == 1) for x in v]}

    # Wilson intervals for the nine headline rates
    print("  Wilson 95% intervals (n=29):")
    wil = {}
    for rep, f in reps.items():
        a, b, c = f(vds), f(vcl), f(vhu)
        row = {}
        for pair_name, (x, y) in (("DS-hum", (a, c)), ("CL-hum", (b, c)), ("DS-CL", (a, b))):
            k_ = sum(1 for u, v in zip(x, y) if u == v)
            lo, hi = wilson(k_, len(x))
            row[pair_name] = (round(100.0 * k_ / len(x), 1), round(lo, 1), round(hi, 1))
            print(f"    {rep:6s} {pair_name}: {row[pair_name][0]:5.1f}  [{row[pair_name][1]:.1f}, {row[pair_name][2]:.1f}]")
            if verify:
                check(f"wilson {rep} {pair_name}", row[pair_name],
                      EXPECTED["human_wilson"][rep][pair_name])
        wil[rep] = row

    # Paired differences with item bootstrap
    rng = np.random.default_rng(SEED_HUMAN)
    idx = np.arange(len(c29))
    diffs = {}
    combos = [("cand", "DS-hum", "DS-CL"), ("strict", "DS-CL", "DS-hum"),
              ("PSI", "DS-CL", "DS-hum")]
    print("  Paired differences (A - B), item bootstrap 95% CI, McNemar counts:")
    for rep, pa, pb in combos:
        f = reps[rep]
        a, b, c = f(vds), f(vcl), f(vhu)
        ia = {"DS-hum": [u == v for u, v in zip(a, c)],
              "CL-hum": [u == v for u, v in zip(b, c)],
              "DS-CL": [u == v for u, v in zip(a, b)]}
        xa, xb = np.array(ia[pa], dtype=float), np.array(ia[pb], dtype=float)
        d = xa - xb
        boots = []
        for _ in range(N_BOOT):
            s = rng.integers(0, len(idx), size=len(idx))
            boots.append(d[s].mean())
        lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        n_ab = int(np.sum((xa == 1) & (xb == 0)))
        n_ba = int(np.sum((xa == 0) & (xb == 1)))
        diffs[(rep, pa, pb)] = (round(100.0 * d.mean(), 1), n_ab, n_ba,
                                round(100.0 * lo, 1), round(100.0 * hi, 1))
        print(f"    {rep:6s} {pa} - {pb}: {diffs[(rep, pa, pb)][0]:+5.1f} pp "
              f"(A only {n_ab}, B only {n_ba})  CI=[{lo * 100:.1f}, {hi * 100:.1f}]")
        if verify:
            check(f"paired diff {rep} {pa}-{pb}", diffs[(rep, pa, pb)],
                  EXPECTED["human_paired_diff"][(rep, pa, pb)])

    # Tied dimension votes among the 29 items
    tied = {}
    print("  Dimension-majority ties among the 29 items:")
    for dname in ("D1", "D2", "D3"):
        keep = []
        for k in c29:
            v, t = categorical_majority([raters[n][k][dname] for n in names])
            if not t:
                keep.append(v)
        tied[dname] = (29 - len(keep), len(keep))
        print(f"    {dname}: {tied[dname][0]} tied -> dropped, n used = {tied[dname][1]}")
        if verify:
            check(f"dim tied {dname}", tied[dname], EXPECTED["dim_tied"][dname], 0)

    # Threshold-then-majority vs majority-then-threshold
    flips_c = flips_s = 0
    alt_c, alt_s = [], []
    for k in c29:
        psis = [bvi_of_dims(raters[n][k]) for n in names]
        alt_c.append(int(sum(p >= 0.5 for p in psis) >= 3))
        alt_s.append(int(sum(p == 1.0 for p in psis) >= 3))
    cur_c = [int(maj_psi[k] >= 0.5) for k in c29]
    cur_s = [int(maj_psi[k] == 1.0) for k in c29]
    flips_c = sum(a != b for a, b in zip(alt_c, cur_c))
    flips_s = sum(a != b for a, b in zip(alt_s, cur_s))
    print(f"  Majority order: threshold-then-majority flips {flips_c}/{len(c29)} candidate "
          f"and {flips_s}/{len(c29)} strict labels vs the paper's rule")
    if verify:
        check("majority flips", (flips_c, flips_s), EXPECTED["majority_order"]["flips"], 0)
    alt_rates = {}
    for rep, alt in (("cand", alt_c), ("strict", alt_s)):
        for jname, v in (("DS", [int(x >= 0.5) if rep == "cand" else int(x == 1) for x in vds]),
                         ("CL", [int(x >= 0.5) if rep == "cand" else int(x == 1) for x in vcl])):
            agree = sum(1 for u, w in zip(v, alt) if u == w)
            alt_rates[(rep, jname)] = round(100.0 * agree / len(alt), 1)
            if verify:
                check(f"alt rule {rep} {jname}", alt_rates[(rep, jname)],
                      EXPECTED["majority_order"]["alt_rates"][(rep, jname)])
    print(f"  Alternative-rule agreement: DS-hum cand {alt_rates[('cand','DS')]} "
          f"(paper {wil['cand']['DS-hum'][0]}), strict {alt_rates[('strict','DS')]}; "
          f"CL-hum cand {alt_rates[('cand','CL')]}, strict {alt_rates[('strict','CL')]}")

    return {"wilson": wil, "diffs": diffs, "tied": tied,
            "flips": (flips_c, flips_s), "alt_rates": alt_rates}


# ----------------------------------------------------------------------
# E. domain x judge fixed-effect logistic regression
# ----------------------------------------------------------------------

def logit_cluster(ds, cl, overlap, check, verify):
    print("== E. Logistic regression: candidate-positive ~ domain * judge (probe-clustered SE) ==")
    rows, clusters = [], []
    for t in overlap:
        r = ds[t]
        d = r["domain"]
        if d not in DOMAINS:
            continue
        rows.append((d, 0, int(psi(r) >= 0.5), r["probe_id"]))
        rows.append((d, 1, int(psi(cl[t]) >= 0.5), r["probe_id"]))

    terms = ["intercept", "REG-S", "UNREG", "Claude", "REG-S:Claude", "UNREG:Claude"]

    def design(d, j):
        return [1.0, float(d == "REG-S"), float(d == "UNREG"), float(j),
                float(d == "REG-S") * j, float(d == "UNREG") * j]

    X = np.array([design(d, j) for d, j, _, _ in rows])
    y = np.array([float(y_) for _, _, y_, _ in rows])
    probe = [p for *_, p in rows]
    n, p_ = X.shape

    beta = np.zeros(p_)
    for _ in range(200):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - mu)
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))

    # cluster sandwich by base probe
    clusters = {}
    for i, pr in enumerate(probe):
        clusters.setdefault(pr, []).append(i)
    meat = np.zeros((p_, p_))
    for idxs in clusters.values():
        Xi = X[idxs]
        ei = (y[idxs] - mu[idxs])
        s = Xi.T @ ei
        meat += np.outer(s, s)
    bread = np.linalg.inv(H)
    V = bread @ meat @ bread
    se = np.sqrt(np.diag(V))

    print(f"  n = {n} rows ({len(clusters)} base probes); judge coded 0=DeepSeek, 1=Claude; base domain REG-H")
    coefs = {}
    for i, tname in enumerate(terms):
        z = beta[i] / se[i]
        pv = math.erfc(abs(z) / math.sqrt(2))
        coefs[tname] = (round(float(beta[i]), 3), round(float(se[i]), 3), round(z, 2), round(pv, 4))
        print(f"    {tname:14s} b={coefs[tname][0]:+7.3f}  SE={coefs[tname][1]:.3f}  z={coefs[tname][2]:+6.2f}  p={coefs[tname][3]:.4f}")
        if verify:
            check(f"logit {tname}", coefs[tname], EXPECTED["logit"][tname],
                  (0.002, 0.002, 0.02, 0.002))

    # Fitted marginal positive-rate gaps Claude - DeepSeek per domain
    print("  Fitted Claude - DeepSeek positive-rate gap (pp):")
    gaps = {}
    for d in DOMAINS:
        x0 = np.array(design(d, 0.0)); x1 = np.array(design(d, 1.0))
        g = 100.0 * (1 / (1 + np.exp(-x1 @ beta)) - 1 / (1 + np.exp(-x0 @ beta)))
        gaps[d] = round(float(g), 1)
        print(f"    {d}: {gaps[d]:+.1f}")
        if verify:
            check(f"logit gap {d}", gaps[d], EXPECTED["logit"]["gaps"][d], 0.15)
    return {"coefs": coefs, "gaps": gaps}


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    failures = []

    def check(name, got, want, tol=0.051):
        def eq(g, w, t):
            if isinstance(g, (int, float)) and isinstance(w, (int, float)) \
                    and not isinstance(g, bool) and not isinstance(w, bool):
                return abs(g - w) <= t
            return g == w
        if isinstance(got, tuple):
            tols = tol if isinstance(tol, tuple) else (tol,) * len(got)
            ok = all(eq(g, w, t) for g, w, t in zip(got, want, tols))
        else:
            ok = eq(got, want, tol)
        if args.verify:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, paper {want}")
            if not ok:
                failures.append(name)
        return got

    ds, cl, overlap = load_primary()
    labels = probe_labels(ds, cl, overlap)
    matrix = spec_matrix(labels, ds, cl, overlap, check, args.verify)
    loo_out = loo(labels, check, args.verify)
    pc = probe_count(labels, check, args.verify)
    human = human_anchor(ds, cl, overlap, check, args.verify)
    logit = logit_cluster(ds, cl, overlap, check, args.verify)

    out = {
        "matrix": {f"{s[0]}|{s[1]}": [round(v, 2) if isinstance(v, float) else v
                                      for v in (m["REG-H"], m["REG-S"], m["UNREG"], mg, lo, hi, mp)]
                   for s, (m, mg, lo, hi, mp) in matrix.items()},
        "loo": {r: {"min": round(v[0][0], 2), "argmin": f"{v[0][1]}.{v[0][2]}",
                    "max": round(v[-1][0], 2), "flips": sum(1 for x, _, _ in v if x <= 0)}
                for r, v in loo_out.items()},
        "probe_count": {str(k): round(v, 1) for k, v in pc.items()},
        "human": {"wilson": human["wilson"],
                  "diffs": {f"{r}|{a}-{b}": v for (r, a, b), v in human["diffs"].items()},
                  "tied": human["tied"],
                  "flips": human["flips"], "alt_rates": {f"{k[0]}|{k[1]}": v
                                                          for k, v in human["alt_rates"].items()}},
        "logit": logit,
    }
    path = Path(__file__).resolve().parent / "revision_v8_numbers.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float),
                    encoding="utf-8")
    print(f"\nnumbers written to {path}")
    if args.verify:
        print("ALL V8 CHECKS PASSED" if not failures else f"FAILURES: {failures}")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
