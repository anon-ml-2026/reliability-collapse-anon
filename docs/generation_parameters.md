# Decoding and Scoring Parameters

This note records every generation and scoring temperature used by the study,
matching Methods §3.1 of Paper A (V17). Numeric values only; the original run
configuration files also contain provider API keys and are **not** part of this
release.

## Subject-model generation (R3)

Source: `experiments/paper2/runs/r3/config.yaml` on the collection machine;
the run driver is `code/collection/r3_run.py` (shipped here).

| Parameter   | Value |
|-------------|-------|
| temperature | 1.0   |
| seed        | 42    |
| max_tokens  | 4096  |

One response is sampled per trial. The same values apply to all eight
providers (a single experiment-level configuration, not per-provider
defaults).

## Subject-model generation (R4, regeneration)

R4 generation predates the July 30 analysis plan (which pre-registered the
subsequent judging, not generation). Its run artifacts are archived with the
R4 collection logs; no separate config is included in this release.

## AI judging

| Judge run | Temperature |
|-----------|-------------|
| DeepSeek-V4-Pro (primary), Claude-Haiku-4.5, Qwen3.6-Plus | 0 |
| Kimi-K3 run 1 / run 2 (pre-registered) | 1.0 (deviation disclosed in the plan) |
| Post-hoc control pilot scoring (DeepSeek pipeline) | 0 |

## Post-hoc control pilot (generation)

Temperature 1.0, one draw per (probe, framing, model, condition) variant;
see `pilot_posthoc_control/README.md`.
