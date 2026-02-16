# Innovation Rescue Experiment (Floor-Clipped Low-g Rescue)

## Configuration
- `innovation_rescue_enabled = true`
- `innovation_rescue_z_hi = 2.33`
- `innovation_rescue_g_quantile = 0.05`
- `innovation_rescue_min_pairs = 30`
- `innovation_rescue_min_pos_frac = 0.60`

## Outcome Summary
Compared against the stable default pipeline (`innovation_rescue_enabled = false`):

- Semisynthetic stress `g×0.7` improved:
  - MAE: `72.2 -> 61.5` min
  - Raw Cov80: `58.3% -> 87.5%`
- But empty-room specificity degraded substantially:
  - C1 (in-sample): `14.96 -> 32.61` min
  - C1-OOS: `19.73 -> 54.71` min
  - C2-OOS: `78.9% -> 49.1%`
- Independent criteria also degraded:
  - E2: `0.217 -> 0.147`
  - E4: `44.3% -> 30.8%`

## Decision
The rescue path is kept in code as an optional research extension, but remains disabled in production/submission configuration because it violates core empty-room validation criteria.

## Artifacts
- `research_outputs/experiments/innovation_rescue_enabled_run/tables/table04_validation_metrics.csv`
- `research_outputs/experiments/innovation_rescue_enabled_run/tables/table09_semisynthetic_summary.csv`
- `research_outputs/experiments/innovation_rescue_enabled_run/tables/table10_semisynthetic_coverage.csv`
- `research_outputs/experiments/innovation_rescue_enabled_run/reports/pipeline_log.txt`
