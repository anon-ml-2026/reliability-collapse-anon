# Reliability Collapses Where Substitution Concentrates — Anonymous Repository

This repository accompanies an anonymous ARR submission. It contains all
materials promised in the paper's *Data and Code Availability* section:

1. All probes (60 base + 180 variants) — `docs/`
2. The D1/D2/D3/PSI scoring rubric and judge prompts, including the
   UNREG-anchored ablation variant — `docs/`, `code/scoring/`
3. All R3/R4 model responses — `data/responses/`
4. All four judges' scores (including the Kimi double-run) — `data/scores/`
5. The 36-item Anchor Set with anonymized annotations from all five raters —
   `data/human/`
6. The 30- and 14-item professional pilot sets with anonymized annotations —
   `data/human/`
7. The pre-registered validation-suite plan (2026-07-30) — `preregistration/`
8. All analysis scripts — `code/analysis/`, plus their outputs in `results/`

## Layout

```
paper/                 anonymous review PDF + compilable LaTeX source
preregistration/       validation-suite plan with decision thresholds
docs/                  probe sets, rubric, PSI synthesis rules, verification notes
data/responses/        r3_responses.jsonl (4,320 trials), r4_responses.jsonl (3,239)
data/scores/           one JSON per judge x round (see table below)
data/human/            anonymized human annotations + annotation manual
code/collection/       response-generation driver (R3)
code/scoring/          judge scoring runners (prompts embedded)
code/analysis/         one script per reported analysis
code/figures/          figure generation (matplotlib)
results/               output JSON of every analysis script
```

## Data files

| File | Content |
|---|---|
| `r3_judge_deepseek.json` | DeepSeek-V4-Pro judge, R3 (primary; 4,201 valid) |
| `r3_judge_claude.json` | Claude-Haiku-4.5 judge, R3 (4,136 valid) |
| `r3_judge_qwen.json` | Qwen3.6-Plus judge, R3 (4,316 valid) |
| `r3_ablation_unreg_anchored_*.json` | both T=0 judges, UNREG-anchored rubric |
| `r4_judge_deepseek.json` | DeepSeek judge, R4 test-retest (3,228 valid) |
| `r4_judge_kimi_run1/run2.json` | Kimi-K3 judge, R4, T=1 double-run |
| `r4_judge_qwen.json` | Qwen judge, R4 (3,231 valid) |

Each score record carries `trial_id` (join key), D1/D2/D3 codes, PSI (`BVI`
field, values 0 / 0.5 / 1), marker flags, and verbatim evidence excerpts.
`*.meta.json` files record run date, model, and validity counts.

## Reproducing the numbers

Analysis scripts join score files on `trial_id`; the binary flag is
PSI >= 0.5. All scripts locate the repository root from their own path
(`Path(__file__).resolve().parents[2]`), so they run out of the box from
any working directory — no configuration needed.

| Paper item | Script | Output |
|---|---|---|
| Table 2 (8 configurations) | `cross_judge_analysis.py`, `r4_qwen_test_retest.py`, `kimi_t1_noise_ceiling.py` | `results/cross_judge_*.json`, `results/test_retest_qwen_and_r4_cross.json` |
| Test-retest + 0.5-band stability | `r3_r4_test_retest.py`, `r3_r4_compute_stats.py` | `results/test_retest_stats.json` |
| PABAK decomposition | `cross_judge_analysis.py` | `results/cross_judge_deepseek_claude.json` |
| Ablation (sec. 4.2) | `ablation_unreg_anchor.py` | `results/ablation_unreg_anchor.json` |
| Prevalence matching | `prevalence_matched*.py` | `results/prevalence_matched_*.json` |
| Noise ceiling | `kimi_t1_noise_ceiling.py` | `results/kimi_t1_noise_ceiling.json` |
| Claude-subject bridge | `claude_band_bridge.py` | `results/claude_band_bridge.json` |
| Human anchor | `human_vs_r3_analysis.py` | `results/human_anchor_analysis.json` |
| Human extended tables (sec. 5.3) | `human_extended_tables.py` | `results/human_extended_tables.json` |
| Professional/lay panels (sec. 5.3) | `panel_pooling.py` | `results/panel_pooling.json` |
| Hui-Walter | `hui_walter_correction.py` | `results/hui_walter_correction.json` |
| PSI rule ablation | `psi_ablation.py` | `results/psi_rule_ablation.json` |
| Figures | `code/figures/generate_figures*.py` | `paper/latex/figures/` |

`verify_all_revisions.py` cross-checks the statistics quoted in the text
against these outputs.

## Scoring pipeline

`code/scoring/run_scoring_generic.py` scores any response file with any
configured provider (DeepSeek / Qwen / Kimi) under either rubric variant
(`--rubric-variant original|unreg-anchored`); API keys are read from
environment variables, never from the repository. The judge system prompt
(original and UNREG-anchored variants) is embedded in
`run_r3_scoring_qwen.py` / `run_scoring_generic.py` and documented in
`docs/scoring_rubric_v2.md`.

## Ethics

Human annotations were collected from consenting, compensated adults
judging synthetic materials; no personal data was collected. Rater
identifiers are anonymized to the labels used in the paper (A1-A5, E1, E2).
All probes and responses are synthetic; no patient or user data appears
anywhere in this repository.

## License

Code: MIT (see `LICENSE`). Data and documentation: CC-BY-4.0.
