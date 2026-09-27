# Reliability Collapses Where Substitution Concentrates — Anonymous Repository

This repository accompanies an **anonymous ARR companion submission pair**:
Paper A ("Same Advice Specificity, Different Authority: Model-Dependent
Effects of Professional Identity Framing", in `paper/paperA/`) and
Paper B ("From Agreement Scores to Defensible Claims", in `paper/latex/`).
It contains all materials promised in both papers' *Data and Code
Availability* sections:

1. All probes (60 base + 180 variants) — `docs/`
2. The D1/D2/D3/PSI scoring rubric and judge prompts, including the
   UNREG-anchored ablation variant — `docs/`, `code/scoring/`
3. All R3/R4 model responses — `data/responses/`
4. All four judges' scores (including the Kimi double-run) — `data/scores/`
5. The migration-benchmark judge scores (REG-L/REG-N, 1,080 responses) —
   `data/migration/`; the probe invitation-strength coding sheets and
   merged codes — `data/probes/invitation_coding/`
6. The 36-item Anchor Set with anonymized annotations from all five raters —
   `data/human/`
7. The 30-, 14-, and 48-item pilot sets with anonymized annotations —
   `data/human/`
8. Pre-registered plans: the validation suite (2026-07-30), the probe
   invitation-strength coding protocol (2026-09-12), and the
   synthetic-calibration protocol (2026-09-13) — `preregistration/`
9. All analysis scripts — `code/analysis/`, plus their outputs in `results/`
10. Paper A's review PDF + compilable LaTeX source — `paper/paperA/`
11. The exploratory post-hoc control pilot (Appendix B of Paper A) —
    `pilot_posthoc_control/`

## Layout

```
paper/                 anonymous review PDFs + compilable LaTeX sources
                       (paper/latex/ = Paper B; paper/paperA/ = Paper A)
preregistration/       pre-registered plans (validation suite; invitation
                       coding; synthetic calibration) with decision criteria
docs/                  probe sets, rubric, PSI synthesis rules, verification notes
data/responses/        r3_responses.jsonl (4,320 trials), r4_responses.jsonl (3,239)
data/scores/           one JSON per judge x round (see table below)
data/migration/        REG-L/REG-N migration-benchmark judge scores
data/probes/           invitation-strength coding sheets + merged codes
data/human/            anonymized human annotations + annotation manual
code/collection/       response-generation driver (R3)
code/scoring/          judge scoring runners (prompts embedded)
code/analysis/         one script per reported analysis
code/figures/          figure generation (matplotlib)
results/               output JSON of every analysis script
pilot_posthoc_control/ Paper A post-hoc pilot: generation, responses, scores
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
| **Paper A** | | |
| Tables 1/2/4/5 + appendix contrast CIs | `paper_a_tables.py` | `results/paper_a_tables.json` |
| Tables 1/2/4/5 (V17 revision, numbers as published) | `paper_a_v17_tables.py` | `results/paper_a_v17_tables.json` |
| Boundary-language uptake (sec. 4.5) | `boundary_uptake_lexicon.py` | `results/boundary_uptake_lexicon.json` |
| Model-level Wilcoxon / paired-t (appendix) | `model_level_tests.py` | `results/model_level_tests.json` |
| Figure 1 (two-panel judge sensitivity) | `code/figures/make_fig_judge_sensitivity_two_panel.py` | `results/figures/` (bundled with the source at `paper/paperA/figures/`) |
| Post-hoc control conditions (appendix) | `pilot_posthoc_control/generate_posthoc_control.py`, `aggregate.py` | `pilot_posthoc_control/` |
| **Paper B — claim-to-evidence audit (v8)** | | |
| Main-text tables 3–6 and appendices D–J (domain concordance, S1–S3 rates and margins, worst-case bounds, per-model margins, contingency, prevalence-matched, migration) | `python code/analysis/claim_matched_audit.py --verify` | prints `ALL CHECKS PASSED` |
| Revision-stage additions: specification matrix, leave-one-probe-out, probe-count sensitivity, anchor Wilson intervals, judge-by-domain regression (app. K–N) | `python code/analysis/claim_audit_revision_v8.py --verify` | `code/analysis/revision_v8_numbers.json`; prints `ALL V8 CHECKS PASSED` |
| Probe invitation-strength control (app. O) | `python code/analysis/probe_invitation_control.py --verify` | prints `ALL INVITATION CHECKS PASSED` |
| Synthetic calibration of the verdict rule (app. P) | `python code/analysis/synthetic_calibration.py --verify` | prints `ALL CALIBRATION CHECKS PASSED` (full simulation, several minutes) |
| Figures (audit pipeline, concordance, rule sensitivity, specification matrix, calibration) | `code/figures/*.py` | `paper/latex/figures/` |
| **Supporting analyses (auxiliary judge configurations; shared by both papers)** | | |
| Auxiliary judge configurations (app. C) | `cross_judge_analysis.py`, `r4_qwen_test_retest.py`, `kimi_t1_noise_ceiling.py` | `results/cross_judge_*.json`, `results/test_retest_qwen_and_r4_cross.json` |
| Test-retest + 0.5-band stability | `r3_r4_test_retest.py`, `r3_r4_compute_stats.py` | `results/test_retest_stats.json` |
| PABAK decomposition | `cross_judge_analysis.py` | `results/cross_judge_deepseek_claude.json` |
| UNREG-anchored rubric ablation | `ablation_unreg_anchor.py` | `results/ablation_unreg_anchor.json` |
| Prevalence matching (app. H) | `prevalence_matched*.py` | `results/prevalence_matched_*.json` |
| Noise ceiling | `kimi_t1_noise_ceiling.py` | `results/kimi_t1_noise_ceiling.json` |
| Claude-subject bridge | `claude_band_bridge.py` | `results/claude_band_bridge.json` |
| Human anchor scoring | `human_vs_r3_analysis.py` | `results/human_anchor_analysis.json` |
| Human extended tables | `human_extended_tables.py` | `results/human_extended_tables.json` |
| Professional/lay panels | `panel_pooling.py` | `results/panel_pooling.json` |
| Hui-Walter correction | `hui_walter_correction.py` | `results/hui_walter_correction.json` |
| PSI rule ablation | `psi_ablation.py` | `results/psi_rule_ablation.json` |

### Verification

From the repository root, the four commands below re-derive every published
Paper B number from the released score files:

```
python code/analysis/claim_matched_audit.py --verify
python code/analysis/claim_audit_revision_v8.py --verify
python code/analysis/probe_invitation_control.py --verify
python code/analysis/synthetic_calibration.py --verify
```

`verify_all_revisions.py` cross-checks further quoted statistics against the
`results/` outputs; the Paper A scripts above each print their headline
values, which match the printed tables exactly.

## Scoring pipeline

`code/scoring/run_scoring_generic.py` scores any response file with any
configured provider (DeepSeek / Qwen / Kimi) under either rubric variant
(`--rubric-variant original|unreg-anchored`); API keys are read from
environment variables, never from the repository. The judge system prompt
(original and UNREG-anchored variants) is embedded in
`run_r3_scoring_qwen.py` / `run_scoring_generic.py` and documented in
`docs/scoring_rubric_v2.md`.

## Ethics

Human annotations were collected from consenting adults judging synthetic
materials; no personal data was collected. All raters (A1–A5, E1, E2) were compensated for their participation. Rater identifiers are anonymized to the
labels used in the papers (A1-A5, E1, E2). All probes and responses are
synthetic; no patient or user data appears anywhere in this repository.

## License

Code: MIT (see `LICENSE`). Data and documentation: CC-BY-4.0.
