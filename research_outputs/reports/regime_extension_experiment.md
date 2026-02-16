# Regime Extension Experiment (Low/High Effective Generation)

Date: 2026-02-16

## Goal
Evaluate whether a low/high effective-generation regime extension improves semisynthetic mismatch robustness (especially `stress_g0.7x_120`) without degrading core validation criteria.

## Code change
Implemented optional regime-aware generation in `research_pipeline/run_full_pipeline.py`:
- Per-block effective generation: `g_eff = g_hat * multiplier(block)`.
- Optional low/high multipliers configured in `research_pipeline/config.toml`.
- Regime-aware physics scale in point estimate and Monte Carlo.
- Regime-aware generation in innovation gate/cap.
- Same regime logic applied in temporal OOS validation, ablation, and semisynthetic evaluation.

## Candidate tested
Config used for candidate run:
- `generation_regime_enabled = true`
- `generation_regime_low_multiplier = 0.80`
- `generation_regime_high_multiplier = 1.00`
- `generation_regime_low_blocks = [2]`
- `floor_multiplier = 0.74`

Candidate artifacts:
- `research_outputs/experiments/regime_lowg_candidate/table04_validation_metrics.csv`
- `research_outputs/experiments/regime_lowg_candidate/table09_semisynthetic_summary.csv`
- `research_outputs/experiments/regime_lowg_candidate/table10_semisynthetic_coverage.csv`
- `research_outputs/experiments/regime_lowg_candidate/table12_ablation_ladder.csv`

## Candidate results (key)
- C1 in-sample (label0 mean unclamped): **15.13 min** (fails `<15`)
- C1-OOS: **19.61 min** (passes `<20`)
- E1: **1.15** (pass)
- E2: **0.32** (pass)
- `stress_g0.7x_120` raw coverage: **62.5%** (baseline: 58.3%)
- `stress_g0.7x_120` MAE: **71.0 min** (baseline: 72.2)

## Decision
Do **not** adopt regime extension as default for the paper version yet.

Reason: it improves the targeted stress case, but it weakens primary empty-room calibration stability (C1 in-sample > 15 min).

## Current default
Default config remains the stable submission setup:
- `generation_regime_enabled = false`
- `floor_multiplier = 0.73`
