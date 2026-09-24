# Sensitivity analysis (Appendix B): exclude DeepSeek-V4-Pro as SUBJECT model,
# recompute domain BVI=1 rates and role-line effect from R3 scoring data.
# Usage: python code/analysis/sensitivity_excl_deepseek.py
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R3 = os.path.join(HERE, "..", "..", "data", "scores", "r3_judge_deepseek.json")

recs = json.load(open(R3, encoding="utf-8"))
valid = [r for r in recs if not r.get("parse_error") and r.get("BVI") is not None]

SUBJECT = "deepseek-v4-pro"
excl = [r for r in valid if r["model_id"] != SUBJECT]
self_scored = [r for r in valid if r["model_id"] == SUBJECT]

print(f"valid R3 trials: {len(valid)}")
print(f"self-scored (DeepSeek-as-subject) trials excluded: {len(self_scored)}")
print(f"remaining: {len(excl)}\n")

def domain_rates(rs):
    out = {}
    for dom in ["CTL", "REG-S", "REG-H", "UNREG"]:
        sub = [r for r in rs if r["domain"] == dom]
        v1 = sum(1 for r in sub if r["BVI"] == 1)
        out[dom] = (len(sub), v1, 100 * v1 / len(sub) if sub else 0.0)
    return out

def role_rates(rs):
    out = {}
    for rl in ["no-role", "with-role", "boundary-role"]:
        sub = [r for r in rs if r["domain"] != "CTL" and r["role_line"] == rl]
        v1 = sum(1 for r in sub if r["BVI"] == 1)
        out[rl] = (len(sub), v1, 100 * v1 / len(sub) if sub else 0.0)
    return out

a, e = domain_rates(valid), domain_rates(excl)
print("=== Domain BVI=1 rates: all 8 models vs excluding DeepSeek-as-subject ===")
print(f"{'domain':7s} {'all n':>7s} {'all %':>7s} {'excl n':>7s} {'excl %':>7s}")
for dom in ["CTL", "REG-S", "REG-H", "UNREG"]:
    print(f"{dom:7s} {a[dom][0]:7d} {a[dom][2]:6.1f}% {e[dom][0]:7d} {e[dom][2]:6.1f}%")

print("\n=== Per-model: highest-violation domain ===")
for m in sorted({r["model_id"] for r in valid}):
    rr = domain_rates([r for r in valid if r["model_id"] == m])
    mx = max(["REG-S", "REG-H", "UNREG"], key=lambda d: rr[d][2])
    print(f"{m:28s} REG-S={rr['REG-S'][2]:5.1f}% REG-H={rr['REG-H'][2]:5.1f}% "
          f"UNREG={rr['UNREG'][2]:5.1f}%  max={mx}")

ra, re_ = role_rates(valid), role_rates(excl)
print("\n=== Role-line effect (non-CTL): BVI=1 rate ===")
for rl in ["no-role", "with-role", "boundary-role"]:
    print(f"{rl:15s} all: n={ra[rl][0]:5d} {ra[rl][2]:5.1f}%   "
          f"excl: n={re_[rl][0]:5d} {re_[rl][2]:5.1f}%")
red_a = 100 * (ra["no-role"][2] - ra["boundary-role"][2]) / ra["no-role"][2]
red_e = 100 * (re_["no-role"][2] - re_["boundary-role"][2]) / re_["no-role"][2]
print(f"\nrelative reduction no-role -> boundary-role: all={red_a:.1f}%  excl={red_e:.1f}%")
