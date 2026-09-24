# Hui-Walter latent-class correction of violation-flag rates (binary flag = BVI >= 0.5).
# Input : results/cross_judge_deepseek_claude.json (DeepSeek=A vs Claude=B, three-level confusion)
# Output: results/hui_walter_correction.json     (2x2 tables, MLE params, GOF, CIs, Rogan-Gladen)
# Pure arithmetic on existing scoring output; no new API calls.
import json, math, sys
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parents[2] / "results" / "cross_judge_deepseek_claude.json"
OUT = Path(__file__).resolve().parents[2] / "results" / "hui_walter_correction.json"
DOMS = ["REG-S", "REG-H", "UNREG"]  # CTL excluded: zero variance by construction
N_BOOT = 1000
SEED = 20260728


def collapse_2x2(cm):
    both_neg = cm.get("A:0->B:0", 0)
    a_neg_b_pos = cm.get("A:0->B:0.5", 0) + cm.get("A:0->B:1", 0)
    a_pos_b_neg = cm.get("A:0.5->B:0", 0) + cm.get("A:1->B:0", 0)
    both_pos = (cm.get("A:0.5->B:0.5", 0) + cm.get("A:0.5->B:1", 0)
                + cm.get("A:1->B:0.5", 0) + cm.get("A:1->B:1", 0))
    return np.array([both_pos, a_pos_b_neg, a_neg_b_pos, both_neg], dtype=float)


def cell_probs(params):
    se1, sp1, se2, sp2 = params[:4]
    pis = params[4:]
    pp = pis * se1 * se2 + (1 - pis) * (1 - sp1) * (1 - sp2)
    pm = pis * se1 * (1 - se2) + (1 - pis) * (1 - sp1) * sp2
    mp = pis * (1 - se1) * se2 + (1 - pis) * sp1 * (1 - sp2)
    mm = pis * (1 - se1) * (1 - se2) + (1 - pis) * sp1 * sp2
    return np.stack([pp, pm, mp, mm], axis=1)  # (K,4)


def em_fit(counts, init, tol=1e-12, max_iter=20000):
    # counts: (K,4) cells [++,+-,-+,--]; params: [se1,sp1,se2,sp2,pi_1..pi_K]
    K = counts.shape[0]
    p = np.array(init, dtype=float)
    n = counts.sum(axis=1)
    for _ in range(max_iter):
        pr = cell_probs(p)                       # (K,4) P(cell | D) mixtures
        pr = np.clip(pr, 1e-300, None)
        num_pos = p[4:, None] * np.array([       # (K,4) P(cell, D+)
            [p[0] * p[2], p[0] * (1 - p[2]), (1 - p[0]) * p[2], (1 - p[0]) * (1 - p[2])]
        ])
        q = num_pos / pr                         # posterior P(D+ | cell)
        w = counts * q                           # expected D+ counts per cell
        pis_new = w.sum(axis=1) / n
        dp = w.sum()                             # total expected D+
        dn = counts.sum() - dp
        se1_new = (w[:, 0] + w[:, 1]).sum() / dp
        se2_new = (w[:, 0] + w[:, 2]).sum() / dp
        fp1 = (counts - w)[:, 0] + (counts - w)[:, 1]
        fp2 = (counts - w)[:, 0] + (counts - w)[:, 2]
        sp1_new = 1 - fp1.sum() / dn
        sp2_new = 1 - fp2.sum() / dn
        new = np.concatenate([[se1_new, sp1_new, se2_new, sp2_new], pis_new])
        if np.max(np.abs(new - p)) < tol:
            p = new
            break
        p = new
    pr = np.clip(cell_probs(p), 1e-300, None)
    ll = float((counts * np.log(pr)).sum())
    return p, ll


def fit_multistart(counts, rng, n_starts=24):
    app = apparent_rates(counts)
    inits = []
    base = [0.8, 0.9, 0.8, 0.9] + list(np.clip(app.mean(axis=1), 0.02, 0.98))
    inits.append(base)
    for _ in range(n_starts - 1):
        inits.append([rng.uniform(0.55, 0.99), rng.uniform(0.7, 0.999),
                      rng.uniform(0.55, 0.99), rng.uniform(0.7, 0.999)]
                     + list(rng.uniform(0.02, 0.6, size=counts.shape[0])))
    best, best_ll = None, -np.inf
    for ini in inits:
        p, ll = em_fit(counts, ini)
        if ll > best_ll:
            best, best_ll = p, ll
    return best, best_ll


def apparent_rates(counts):
    n = counts.sum(axis=1)
    a_pos = (counts[:, 0] + counts[:, 1]) / n   # judge A apparent
    b_pos = (counts[:, 0] + counts[:, 2]) / n   # judge B apparent
    return np.stack([a_pos, b_pos], axis=1)     # (K,2)


def rogan_gladen(p_app, se, sp):
    denom = se + sp - 1.0
    return (p_app + sp - 1.0) / denom if abs(denom) > 1e-9 else float("nan")


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    doms = data["domains"]
    tables, counts = {}, []
    expected = {"REG-S": (852, 37), "REG-H": (755, 86), "UNREG": (645, 201)}
    for d in DOMS:
        c = collapse_2x2(doms[d]["confusion_matrix"])
        counts.append(c)
        n = int(c.sum())
        agree = int(c[0] + c[3])
        exp_agree, exp_both = expected[d]
        assert agree == exp_agree and int(c[0]) == exp_both, f"{d}: cross-check vs Table 4/7 failed"
        tables[d] = {"n": n, "both_pos": int(c[0]), "a_only": int(c[1]),
                     "b_only": int(c[2]), "both_neg": int(c[3]),
                     "binary_agree_pct": round(100 * agree / n, 2)}
    counts = np.stack(counts)

    rng = np.random.default_rng(SEED)
    mle, ll = fit_multistart(counts, rng)
    se1, sp1, se2, sp2 = mle[:4]
    pis = mle[4:]
    pr = cell_probs(mle)
    exp_counts = pr * counts.sum(axis=1, keepdims=True)
    chi2 = float(((counts - exp_counts) ** 2 / np.clip(exp_counts, 1e-9, None)).sum())
    dof = 3 * 3 - 7
    p_gof = 1 - chi2_cdf(chi2, dof)

    boot = []
    for _ in range(N_BOOT):
        bc = np.stack([rng.multinomial(int(counts[k].sum()), counts[k] / counts[k].sum())
                       for k in range(counts.shape[0])])
        try:
            bp, _ = fit_multistart(bc, rng, n_starts=6)
        except Exception:
            continue
        if bp[0] + bp[1] > 1 and bp[2] + bp[3] > 1:
            boot.append(bp)
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    app = apparent_rates(counts)
    rg = {d: {"judge_A": round(rogan_gladen(app[i, 0], se1, sp1), 4),
              "judge_B": round(rogan_gladen(app[i, 1], se2, sp2), 4)}
          for i, d in enumerate(DOMS)}

    out = {
        "method": "Hui-Walter latent class, 2 tests x 3 populations, conditional independence; "
                  "EM multistart; bootstrap percentile CI (resampled multinomial, seed %d)" % SEED,
        "tables_2x2": tables,
        "mle": {"se_DeepSeek": round(float(se1), 4), "sp_DeepSeek": round(float(sp1), 4),
                "se_Claude": round(float(se2), 4), "sp_Claude": round(float(sp2), 4),
                "prevalence": {d: round(float(pis[i]), 4) for i, d in enumerate(DOMS)}},
        "prevalence_ci95": {d: [round(float(lo[4 + i]), 4), round(float(hi[4 + i]), 4)]
                            for i, d in enumerate(DOMS)},
        "se_ci95": {"DeepSeek": [round(float(lo[0]), 4), round(float(hi[0]), 4)],
                    "Claude": [round(float(lo[2]), 4), round(float(hi[2]), 4)]},
        "sp_ci95": {"DeepSeek": [round(float(lo[1]), 4), round(float(hi[1]), 4)],
                    "Claude": [round(float(lo[3]), 4), round(float(hi[3]), 4)]},
        "gof": {"chi2": round(chi2, 3), "dof": dof, "p": round(float(p_gof), 4)},
        "expected_counts": {d: {c: round(float(exp_counts[i, j]), 1)
                                for j, c in enumerate(["both_pos", "a_only", "b_only", "both_neg"])}
                            for i, d in enumerate(DOMS)},
        "rogan_gladen_from_apparent": rg,
        "apparent_rates": {d: {"A_DeepSeek": round(float(app[i, 0]), 4),
                               "B_Claude": round(float(app[i, 1]), 4)}
                           for i, d in enumerate(DOMS)},
        "bracket_check": {d: {"floor": round(tables[d]["both_pos"] / tables[d]["n"], 4),
                              "naive_A": round(float(app[i, 0]), 4),
                              "hw_estimate": round(float(pis[i]), 4)}
                          for i, d in enumerate(DOMS)},
        "n_boot_valid": int(len(boot)),
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


def chi2_cdf(x, k):
    # regularized lower incomplete gamma for small integer k via series (k=2 -> 1-exp(-x/2))
    if k == 2:
        return 1 - math.exp(-x / 2)
    from scipy.stats import chi2 as _c
    return float(_c.cdf(x, k))


if __name__ == "__main__":
    main()
