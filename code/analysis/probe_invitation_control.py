"""Probe invitation-strength control (paper B revision extension 3).

Pre-registered in V16/probe_coding/protocol.md before coding; codings for 60
base probes by three independent coders (two rounds) are archived in
data/probes/invitation_strength_codes.json.  Analyses:

  A1  domain distribution of coded strength + probe bootstrap CIs of the
      UNREG - REG-H / UNREG - REG-S mean differences (2,000 draws, seed
      20260915)
  A2  response-level logistic regression of the S1 candidate label
      (DeepSeek, common support) on domain alone (M0) and domain + coded
      strength (M1), probe-clustered sandwich SEs over the 45 professional
      probes; reports the attenuation of the UNREG coefficient
  A3  strength-stratified S1 rates and UNREG-highest margins (direction)
      within strata
  J   pre-registered verdicts J1/J2/J3 (see protocol.md)

Run from the repository root:
    python code/analysis/probe_invitation_control.py            # report
    python code/analysis/probe_invitation_control.py --verify   # + checks
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from claim_matched_audit import DOMAINS, ROOT, load_judge, psi

CODES = ROOT / "data" / "probes" / "invitation_strength_codes.json"
SEED_A1 = 20260915
N_BOOT_A1 = 2_000

EXPECTED = {
    "domain_means": {"REG-H": 1.178, "REG-S": 1.222, "UNREG": 0.844, "CTL": 0.0},
    "fleiss_60": 0.941,
    "fleiss_45": 0.888,
    "pairwise": {"A-B": 60, "A-C": 57, "B-C": 57},
    "a1_ci": {"UNREG-REG-H": (-0.42, -0.25), "UNREG-REG-S": (-0.46, -0.30)},
    "a2": {"m0_unreg": 0.917, "m1_unreg": 0.456, "m1_strength": -1.581,
           "m0_p": 0.0347, "m1_p": 0.2657, "attenuation": 0.503},
    "a3": {"0": {"counts": {"REG-H": 0, "REG-S": 0, "UNREG": 3}, "rates": None, "margin": None},
           "1": {"counts": {"REG-H": 12, "REG-S": 12, "UNREG": 11},
                 "rates": {"REG-H": 20.7, "REG-S": 5.7, "UNREG": 27.6}, "margin": 6.9},
           "2": {"counts": {"REG-H": 3, "REG-S": 3, "UNREG": 1}, "rates": None, "margin": None}},
}


def fleiss(codes_by_rater, items, cats=(0, 1, 2)):
    p_cat = np.zeros(len(cats))
    Pi = []
    for it in items:
        counts = Counter(codes_by_rater[r][it] for r in codes_by_rater)
        n_i = sum(counts.values())
        Pi.append((sum(c * c for c in counts.values()) - n_i) / (n_i * (n_i - 1)))
        for ci, c in enumerate(cats):
            p_cat[ci] += counts.get(c, 0) / (n_i * len(items))
    P_bar = float(np.mean(Pi))
    Pe = float(sum(p * p for p in p_cat))
    return (P_bar - Pe) / (1 - Pe) if Pe < 1 else 1.0


def logit_cluster(rows, terms):
    """rows: list of (y, x-dict); returns coefs, cluster-SEs (probe key)."""
    X = np.array([[1.0] + [float(r[1][t]) for t in terms] for r in rows])
    y = np.array([float(r[0]) for r in rows])
    cl = np.array([r[1]["_probe"] for r in rows])
    p = X.shape[1]
    beta = np.zeros(p)
    for _ in range(200):
        mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
    meat = np.zeros((p, p))
    for pr in sorted(set(cl)):
        idx = cl == pr
        s = X[idx].T @ (y[idx] - mu[idx])
        meat += np.outer(s, s)
    V = np.linalg.inv(H) @ meat @ np.linalg.inv(H)
    se = np.sqrt(np.diag(V))
    out = {}
    for i, t in enumerate(["intercept"] + terms):
        z = beta[i] / se[i]
        out[t] = (round(float(beta[i]), 3), round(float(se[i]), 3),
                  round(float(z), 2), round(float(math.erfc(abs(z) / math.sqrt(2))), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    failures = []

    def check(name, got, want, tol=0.051):
        if want is None:
            return
        ok = abs(got - want) <= tol
        if args.verify:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, paper {want}")
            if not ok:
                failures.append(name)

    raw = json.loads(CODES.read_text(encoding="utf-8"))
    refined = raw["rounds"]["refined"]
    panel = raw["panel"]
    probes = sorted(panel)
    dom = lambda p: panel[p]["domain"]
    strength = {p: panel[p]["mean"] for p in probes}
    maj = {p: panel[p]["majority"] for p in probes}

    # ---------------- coder agreement ----------------
    print("== Coder agreement (refined round, 3 coders) ==")
    pairs = {"A-B": 0, "A-C": 0, "B-C": 0}
    for p in probes:
        for key in pairs:
            a, b = key.split("-")
            pairs[key] += refined[a][p] == refined[b][p]
    for k, v in pairs.items():
        print(f"  exact {k}: {v}/60")
        check(f"pairwise {k}", v, EXPECTED["pairwise"][k], 0)
    k60 = fleiss(refined, probes)
    k45 = fleiss(refined, [p for p in probes if dom(p) != "CTL"])
    print(f"  Fleiss kappa: 60 items {k60:.3f}, 45 professional {k45:.3f}")
    check("fleiss_60", round(k60, 3), EXPECTED["fleiss_60"], 0.005)
    check("fleiss_45", round(k45, 3), EXPECTED["fleiss_45"], 0.005)

    # ---------------- A1: distribution ----------------
    print("\n== A1. Coded strength by domain ==")
    means = {}
    for d in ("REG-H", "REG-S", "UNREG", "CTL"):
        vals = [strength[p] for p in probes if dom(p) == d]
        means[d] = float(np.mean(vals))
        dist = Counter(maj[p] for p in probes if dom(p) == d)
        print(f"  {d}: mean {means[d]:.3f}  majority-code counts {dict(sorted(dist.items()))}")
        check(f"mean {d}", round(means[d], 3), EXPECTED["domain_means"][d], 0.005)

    rng = np.random.default_rng(SEED_A1)
    ci = {}
    for d in ("REG-H", "REG-S"):
        diffs = []
        plist = {dd: [p for p in probes if dom(p) == dd] for dd in ("UNREG", d)}
        for _ in range(N_BOOT_A1):
            du = [strength[p] for p in plist["UNREG"]
                  for p in rng.choice(plist["UNREG"], size=len(plist["UNREG"]), replace=True)]
            dd_ = [strength[p] for p in plist[d]
                   for p in rng.choice(plist[d], size=len(plist[d]), replace=True)]
            diffs.append(np.mean(du) - np.mean(dd_))
        lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
        ci[f"UNREG-{d}"] = (round(float(lo), 2), round(float(hi), 2))
        print(f"  UNREG - {d}: {means['UNREG'] - means[d]:+.3f}  CI {ci[f'UNREG-{d}']}")
    if EXPECTED["a1_ci"]["UNREG-REG-H"] is not None:
        for k, v in ci.items():
            check(f"a1_ci {k}", v[0], EXPECTED["a1_ci"][k][0], 0.11)
            check(f"a1_ci {k}", v[1], EXPECTED["a1_ci"][k][1], 0.11)

    # ---------------- A2: regression ----------------
    print("\n== A2. S1 candidate label ~ domain (+ strength), probe-clustered ==")
    ds = load_judge(ROOT / "data" / "scores" / "r3_judge_deepseek.json")
    cl = load_judge(ROOT / "data" / "scores" / "r3_judge_claude.json")
    overlap = sorted(set(ds) & set(cl))
    rows = []
    for t in overlap:
        r = ds[t]
        if r["domain"] not in DOMAINS:
            continue
        rows.append((int(psi(r) >= 0.5),
                     {"REG-S": float(r["domain"] == "REG-S"),
                      "UNREG": float(r["domain"] == "UNREG"),
                      "strength": strength[r["probe_id"]],
                      "_probe": r["probe_id"]}))
    print(f"  rows: {len(rows)} responses, {len({r[1]['_probe'] for r in rows})} probes")
    m0 = logit_cluster(rows, ["REG-S", "UNREG"])
    m1 = logit_cluster(rows, ["REG-S", "UNREG", "strength"])
    for name, m in (("M0 domain-only", m0), ("M1 + strength", m1)):
        print(f"  {name}:")
        for t in ("UNREG", "strength"):
            if t in m:
                b, se, z, pv = m[t]
                print(f"    {t:9s} b={b:+.3f} SE={se:.3f} z={z:+.2f} p={pv:.4f}")
    att = 1.0 - m1["UNREG"][0] / m0["UNREG"][0]
    print(f"  UNREG attenuation after strength: {att * 100:.1f}%")
    if EXPECTED["a2"]["m0_unreg"] is not None:
        check("m0_unreg", m0["UNREG"][0], EXPECTED["a2"]["m0_unreg"], 0.005)
        check("m1_unreg", m1["UNREG"][0], EXPECTED["a2"]["m1_unreg"], 0.005)
        check("m1_strength", m1["strength"][0], EXPECTED["a2"]["m1_strength"], 0.005)
        check("attenuation", round(att, 3), EXPECTED["a2"]["attenuation"], 0.02)

    # ---------------- A3: stratified margins ----------------
    print("\n== A3. Strength-stratified S1 rates and margins ==")
    strata = {}
    for p in probes:
        if dom(p) == "CTL":
            continue
        strata.setdefault(maj[p], {}).setdefault(dom(p), []).append(p)
    a3 = {}
    for s in sorted(strata):
        counts = {d: len(strata[s].get(d, [])) for d in DOMAINS}
        if min(counts.values()) < 2:
            a3[s] = {"counts": counts, "rates": None, "margin": None}
            print(f"  stratum {s}: n={counts}  sparse (<2 probes in some domain) -> margin not computed")
            continue
        rates = {}
        for d in DOMAINS:
            ps = strata[s][d]
            ts = [t for t in overlap if ds[t]["probe_id"] in ps]
            rates[d] = 100.0 * sum(psi(ds[t]) >= 0.5 for t in ts) / len(ts)
        margin = rates["UNREG"] - max(rates["REG-H"], rates["REG-S"])
        a3[s] = {"counts": counts, "rates": {d: round(rates[d], 1) for d in DOMAINS},
                 "margin": round(margin, 1)}
        print(f"  stratum {s}: n={counts}  rates {a3[s]['rates']}  margin {margin:+.1f}")
    if EXPECTED["a3"] is not None:
        same = json.dumps(a3, sort_keys=True) == json.dumps(EXPECTED["a3"], sort_keys=True)
        if args.verify:
            print(f"  [{'PASS' if same else 'FAIL'}] a3: exact match {same}")
            if not same:
                failures.append("a3")

    # ---------------- J verdicts ----------------
    print("\n== Pre-registered verdicts ==")
    j3 = means["UNREG"] <= means["REG-H"] or means["UNREG"] <= means["REG-S"] or means["CTL"] > 0.5
    att_pct = att * 100
    j1 = att_pct >= 50.0 or m1["UNREG"][3] >= 0.05
    j2 = (not j1)
    print(f"  J3 (validity anchor): UNREG strength not above both regulated domains "
          f"or CTL>0.5 -> {j3}")
    print(f"  J1 (attribution): attenuation {att_pct:.1f}% >= 50 or M1 UNREG p>=.05 -> {j1}")
    print(f"  J2 (independence): {j2}")
    if j3:
        print("  -> Branch C per protocol: exploratory reporting, soften S4.1, "
              "rewrite S6.2 as construct-difference wording, appendix only.")
    elif j2:
        print("  -> Branch A: margin independent of invitation strength.")
    else:
        print("  -> Branch B: margin largely attributable to invitation strength.")

    if args.verify:
        print(f"\n{'ALL INVITATION CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
