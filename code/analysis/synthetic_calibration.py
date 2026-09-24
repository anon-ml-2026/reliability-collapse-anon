"""Synthetic calibration of the codified verdict rule (paper B extension 1).

Pre-registered in V16/synthetic_calibration/protocol.md BEFORE running:
the simulation model, condition grid, verdict criteria, and both wording
branches were frozen first.  The latent-variable model matches the real
benchmark's structure (3 professional domains x 15 probes x 66 responses,
8 answer-model slots, two judges with domain marginals and a calibrated
judge-agreement level); ground truth is known by construction.

Panels (protocol section 3):
  A  power/sensitivity: P(survives) vs true UNREG-highest margin Delta,
     unbiased judges, S1 and S2 rules
  B  one-sided judge bias: Delta=0, DeepSeek over-flags UNREG by delta;
     false-survival of S1 vs S2 (the intersection rule's robustness value)
  C  shared rubric bias: Delta=0, both judges over-flag UNREG by delta;
     false-survival of S2 (the residual threat the design cannot exclude)

Verdict rule (paper S3.4, codified): a direction survives when its
probe-clustered 95% interval excludes zero AND >= 7/8 fixed answer-model
margins share the sign.

Run from the repository root:
    python code/analysis/synthetic_calibration.py            # report
    python code/analysis/synthetic_calibration.py --verify   # + checks
"""

import argparse
import json
import math
import sys

import numpy as np

SEED_W = 20260916          # w (signal-weight) calibration
REPS = 300
N_BOOT = 2_000
PANEL_A_DELTAS = [0, 2.5, 5, 7.5, 10, 12.5, 15, 20, 25]
BIAS_DELTAS = [5, 10, 15, 19]
BASE = {"REG-H": 0.205, "REG-S": 0.0975, "UNREG": 0.38}
EQUAL_RATE = 0.205          # Panel B/C: all domains at the REG-H level
N_PROBES, N_RESP, N_MODELS = 15, 66, 8
TARGET_KAPPA, KAPPA_TOL = 0.28, 0.05

EXPECTED = {
    "w": 0.95, "kappa_sim": 0.273,
    "panelA": {("S1", 0): 0.013, ("S1", 5): 0.490, ("S1", 7.5): 0.847,
               ("S1", 10): 0.973, ("S1", 12.5): 1.000,
               ("S2", 0): 0.010, ("S2", 7.5): 0.760, ("S2", 12.5): 0.990},
    "panelB": {("S1", 19): 1.000, ("S2", 19): 0.460, ("S2", 10): 0.110},
    "panelC": {("S2", 10): 0.827, ("S2", 19): 1.000},
}


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def norm_ppf(p):
    """Acklam-free: use rational approximation via math.erfinv-free Newton on erf."""
    # small-domain inverse via bisection on norm_cdf (vector of scalars)
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if norm_cdf(np.array([mid]))[0] < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# 1-D grid for bivariate-normal agreement integrals
GU = np.linspace(-7.0, 7.0, 2801)
GW = np.exp(-0.5 * GU * GU) / math.sqrt(2 * math.pi)
GW = GW / GW.sum()


def kappa_for(margs, w):
    """Cohen's kappa for two judges with marginals `margs` sharing latent u~N(0,1)
    plus independent judge noise N(0,1), signal weight w."""
    cs = [-norm_ppf(p) * math.sqrt(w * w + 1.0) for p in margs]
    s1 = norm_cdf(w * GU - cs[0])
    s2 = norm_cdf(w * GU - cs[1])
    p11 = float(np.sum(GW * s1 * s2))
    p00 = float(np.sum(GW * (1 - s1) * (1 - s2)))
    po = p11 + p00
    pe = margs[0] * margs[1] + (1 - margs[0]) * (1 - margs[1])
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def calibrate_w():
    best, bw = None, None
    for w in np.round(np.arange(0.20, 3.01, 0.05), 2):
        ks = [kappa_for(m, w) for m in
              ([0.205, 0.199], [0.0975, 0.143], [0.401, 0.359])]
        dev = abs(float(np.mean(ks)) - TARGET_KAPPA)
        if best is None or dev < best:
            best, bw = dev, float(w)
    return bw


def thresholds(marg, w):
    """Per-response latent thresholds: flag = (w*u + eta > c)."""
    return -norm_ppf(marg) * math.sqrt(w * w + 1.0)


def simulate_dataset(w, marg_ds, marg_cl, seed):
    """Returns per-probe per-domain flag counts for (S1: DS, S2: DS&CL),
    plus model-wise S1 rates per domain."""
    rng = np.random.default_rng(seed)
    n_per_dom = N_PROBES * N_RESP
    rows = []
    for d in ("REG-H", "REG-S", "UNREG"):
        u = rng.standard_normal(n_per_dom)
        e1 = rng.standard_normal(n_per_dom)
        e2 = rng.standard_normal(n_per_dom)
        f1 = (w * u + e1) > thresholds(marg_ds[d], w)
        f2 = (w * u + e2) > thresholds(marg_cl[d], w)
        probe = np.repeat(np.arange(N_PROBES), N_RESP)
        model = np.arange(n_per_dom) % N_MODELS
        rows.append((d, probe, model, f1, f2))
    return rows


def summarize(rows, w):
    """Per-probe (pos, tot) for S1 and S2; model-wise margins; point margins."""
    out = {}
    for rule in ("S1", "S2"):
        per_probe = {}
        model_pos = {m: {d: [0, 0] for d in ("REG-H", "REG-S", "UNREG")} for m in range(N_MODELS)}
        point = {d: [0, 0] for d in ("REG-H", "REG-S", "UNREG")}
        for d, probe, model, f1, f2 in rows:
            flag = f1 if rule == "S1" else (f1 & f2)
            idx_p = per_probe.setdefault(d, np.zeros((N_PROBES, 2), dtype=int))
            np.add.at(idx_p, probe, np.stack([flag, np.ones_like(flag)], axis=1).astype(int))
            for m in range(N_MODELS):
                sel = model == m
                model_pos[m][d][0] += int(flag[sel].sum())
                model_pos[m][d][1] += int(sel.sum())
            point[d][0] += int(flag.sum())
            point[d][1] += int(len(flag))
        out[rule] = (per_probe, model_pos, point)
    return out


def margin_from(probe_counts, idx_by_dom):
    rates = {}
    for d in ("REG-H", "REG-S", "UNREG"):
        sel = probe_counts[d][idx_by_dom[d]]
        rates[d] = 100.0 * sel[:, 0].sum() / sel[:, 1].sum()
    return rates["UNREG"] - max(rates["REG-H"], rates["REG-S"])


def bootstrap_verdict(summ, rng):
    """2,000-draw probe bootstrap per rule; returns (survives, ci_excl0) per rule."""
    res = {}
    for rule in ("S1", "S2"):
        per_probe, model_pos, point = summ[rule]
        idx = {d: rng.integers(0, N_PROBES, size=(N_BOOT, N_PROBES)) for d in per_probe}
        draws = np.empty(N_BOOT)
        for i in range(N_BOOT):
            draws[i] = margin_from(per_probe, {d: idx[d][i] for d in per_probe})
        lo, hi = np.percentile(draws, 2.5), np.percentile(draws, 97.5)
        ci_excl = (lo > 0) or (hi < 0)
        point_m = {d: 100.0 * point[d][0] / point[d][1] for d in point}
        pm = point_m["UNREG"] - max(point_m["REG-H"], point_m["REG-S"])
        model_margins = []
        for m in range(N_MODELS):
            r = {d: 100.0 * model_pos[m][d][0] / model_pos[m][d][1] for d in point}
            model_margins.append(r["UNREG"] - max(r["REG-H"], r["REG-S"]))
        n_pos = sum(mm > 0 for mm in model_margins)
        survives = ci_excl and (n_pos >= 7)
        res[rule] = (bool(survives), bool(ci_excl), round(pm, 2))
    return res


def run_panelA(w):
    rng_master = np.random.default_rng(SEED_W)
    out = {}
    for delta in PANEL_A_DELTAS:
        for rule in ("S1", "S2"):
            out[(rule, delta)] = [0, 0]  # survives, ci_excl
    for rep in range(REPS):
        for delta in PANEL_A_DELTAS:
            marg = {"REG-H": BASE["REG-H"], "REG-S": BASE["REG-S"],
                    "UNREG": BASE["REG-H"] + delta / 100.0}
            rows = simulate_dataset(w, marg, marg, seed=int(rng_master.integers(0, 2**31)))
            summ = summarize(rows, w)
            # deduplicate bootstrap per dataset: verdict for both rules from same draws
            rng_b = np.random.default_rng(int(rng_master.integers(0, 2**31)))
            res = bootstrap_verdict(summ, rng_b)
            for rule in ("S1", "S2"):
                out[(rule, delta)][0] += res[rule][0]
                out[(rule, delta)][1] += res[rule][1]
    return {k: (v[0] / REPS, v[1] / REPS) for k, v in out.items()}


def run_bias_panel(w, shared):
    rng_master = np.random.default_rng(SEED_W + (2 if shared else 1))
    out = {}
    for delta in BIAS_DELTAS:
        rules = ("S1", "S2") if not shared else ("S2",)
        for rule in rules:
            out[(rule, delta)] = 0
    for rep in range(REPS):
        for delta in BIAS_DELTAS:
            marg_base = {"REG-H": EQUAL_RATE, "REG-S": EQUAL_RATE, "UNREG": EQUAL_RATE}
            marg_ds = dict(marg_base)
            marg_cl = dict(marg_base)
            marg_ds["UNREG"] = min(0.999, marg_base["UNREG"] + delta / 100.0)
            if shared:
                marg_cl["UNREG"] = marg_ds["UNREG"]
            rows = simulate_dataset(w, marg_ds, marg_cl, seed=int(rng_master.integers(0, 2**31)))
            summ = summarize(rows, w)
            rng_b = np.random.default_rng(int(rng_master.integers(0, 2**31)))
            res = bootstrap_verdict(summ, rng_b)
            for rule in (("S1", "S2") if not shared else ("S2",)):
                out[(rule, delta)] += res[rule][0]
    return {k: v / REPS for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    failures = []

    def check(name, got, want, tol):
        ok = abs(got - want) <= tol
        if args.verify:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, locked {want}")
            if not ok:
                failures.append(name)

    w = calibrate_w()
    ks = [kappa_for(m, w) for m in ([0.205, 0.199], [0.0975, 0.143], [0.401, 0.359])]
    print(f"== Calibrated signal weight w = {w} -> sim kappa per domain "
          f"{[round(k, 3) for k in ks]} (target {TARGET_KAPPA} +/- {KAPPA_TOL})")
    if EXPECTED["w"] is not None:
        check("w", w, EXPECTED["w"], 0.001)
        check("kappa_sim", round(float(np.mean(ks)), 3), EXPECTED["kappa_sim"], 0.01)

    print(f"\n== Panel A: P(survives) vs true margin (REPS={REPS}, boot={N_BOOT}) ==")
    resA = run_panelA(w)
    for delta in PANEL_A_DELTAS:
        s1 = resA[("S1", delta)]
        s2 = resA[("S2", delta)]
        print(f"  Delta={delta:5.1f}  S1: P(surv)={s1[0]:.3f} P(ci>0)={s1[1]:.3f}   "
              f"S2: P(surv)={s2[0]:.3f} P(ci>0)={s2[1]:.3f}")

    print(f"\n== Panel B: one-sided DeepSeek UNREG bias (true margin 0) ==")
    resB = run_bias_panel(w, shared=False)
    for delta in BIAS_DELTAS:
        print(f"  delta={delta:2d}  S1 false-survival={resB[('S1', delta)]:.3f}  "
              f"S2 false-survival={resB[('S2', delta)]:.3f}")

    print(f"\n== Panel C: shared rubric bias (true margin 0, S2 rule) ==")
    resC = run_bias_panel(w, shared=True)
    for delta in BIAS_DELTAS:
        print(f"  delta={delta:2d}  S2 false-survival={resC[('S2', delta)]:.3f}")

    if args.verify and EXPECTED["panelA"]:
        for k, v in EXPECTED["panelA"].items():
            check(f"panelA {k}", round(resA[k][0], 3), v, 0.06)
        for k, v in EXPECTED["panelB"].items():
            check(f"panelB {k}", round(resB[k], 3), v, 0.06)
        for k, v in EXPECTED["panelC"].items():
            check(f"panelC {k}", round(resC[k], 3), v, 0.06)
        print(f"\n{'ALL CALIBRATION CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
