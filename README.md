# CO2-Based Occupancy Duration Inference (Spring 2025)

Code + outputs repository for the paper workflow on dormitory CO2-based occupancy duration estimation.

## What This Repo Contains
- `research_pipeline/`: primary paper pipeline code.
- `research_outputs/`: generated outputs (tables, figures, reports, estimator outputs, compiled paper files).
- `_archived_not_used_in_paper/workflow/`: archived exploratory workflows not used in the final paper.
- `_archived_not_used_in_paper/tmp_ensemble_cfg/`: archived config variants.
- `assets/gifs/`: animated previews for key results.

## What Is Excluded
- Raw CO2 source folders.
- Raw occupancy input files.
- Calibration raw data files.

## Pipeline Flow
```mermaid
flowchart TD
    A[Minute CO2 + Block Presence Labels] --> B[Sensor-wise AR1 Physics Fit]
    B --> C[Block Feature Construction]
    C --> D[Empty-Floor Calibration]
    D --> E[Block Minutes Estimation]
    E --> F[Monte Carlo Uncertainty<br/>Parameter + Structural Mismatch Priors]
    F --> G[Validation + Sensitivity + Baseline Comparators]
    G --> H[Tables / Figures / Paper Outputs]
```

## Validation Framework
```mermaid
flowchart LR
    C1["C1 & C2 Calibration Diagnostics (in-sample)"] --> OOS["C1-OOS & C2-OOS Temporal Split"]
    OOS --> E1[E1/E2 Cross-Sensor Generalization]
    E1 --> E2[E3 Parameter Sensitivity]
    E2 --> E3[E4 Uncertainty Informativeness]
    E3 --> S1[Semisynthetic Coverage by Scenario]
```

## Repository Layout
```mermaid
graph TD
    R[repo root]
    R --> RP[research_pipeline]
    R --> RO[research_outputs]
    R --> AR[_archived_not_used_in_paper]
    R --> AS[assets]

    RO --> ROT[tables]
    RO --> ROF[figures]
    RO --> ROV[validation]
    RO --> ROR[reports]
    RO --> ROP[paper]
    RO --> ROE[estimator]
```

## End-to-End GIF (Simple Storyboard)
This GIF explains the full workflow from data to final results in plain steps.

![End to End Pipeline](assets/gifs/end_to_end_pipeline.gif)

## Key Results Snapshot (Latest Run)
Source: `research_outputs/tables/table04_validation_metrics.csv`

| Metric | Value | Criterion | Pass |
|---|---:|---:|:---:|
| C1 in-sample mean unclamped (label=0) | 12.82 min | < 15 | YES |
| C2 in-sample % under 5 min | 82.8% | > 60% | YES |
| C1-OOS mean unclamped (label=0) | 19.91 min | < 20 | YES |
| C2-OOS % under 5 min | 82.5% | > 50% | YES |
| C1-OOS median unclamped | 0.00 min | descriptive | N/A (descriptive) |
| C1-OOS 5% trimmed mean | 12.94 min | descriptive | N/A (descriptive) |
| C1-OOS tail contribution (top 5%) | 52.2% | descriptive | N/A (descriptive) |
| E1 LOO mean relative MAE | 1.22 | < 1.5 | YES |
| E2 LOO mean Spearman rho | 0.334 | > 0.2 | YES |
| E3 max phi sensitivity change | 48.4% | < 200% | YES |
| E4 high-confidence blocks | 47.9% | > 40% | YES |

`N/A (descriptive)` means the metric is reported for interpretation (robustness/tail behavior), not used as a pass/fail criterion.

## Submission Readiness (Building and Environment)
- Main manuscript precheck: `submission/building_and_environment_precheck.md`
- Current manuscript formatting profile:
  - Abstract: 246 words
  - Keywords: 6
  - Figures in main manuscript: 9
  - Tables in main manuscript: 4
  - Total figures + tables: 13

Descriptive analyses not essential for the core contribution remain in `research_outputs/` and are referenced from the manuscript as reproducible outputs.

## Semisynthetic CI Coverage (Latest Run)
Source: `research_outputs/tables/table10_semisynthetic_coverage.csv`

| Group | Coverage (80% CI) | Mean CI Width | MAE |
|---|---:|---:|---:|
| Continuous | 78.3% | 103.3 min | 34.5 min |
| Fragmented | 91.7% | 143.5 min | 53.8 min |
| Stress | 80.0% | 147.0 min | 45.9 min |
| Overall | 81.3% | 128.6 min | 42.6 min |

## Baseline Comparator Note
Source: `research_outputs/tables/table07_baseline_comparators.csv`

Baseline comparison is computed on a comparable room-level subset (`n_label0=177`, `n_label1=260`) for method-to-method comparability.

## New Robustness Artifacts
- Detectability diagnostics table: `research_outputs/tables/table11_detectability_thresholds.csv`
- Ablation ladder table: `research_outputs/tables/table12_ablation_ladder.csv`
- Detectability figure: `research_outputs/figures/fig15_detectability_thresholds.pdf`
- Ablation figure: `research_outputs/figures/fig16_ablation_ladder.pdf`

## Reproducing
```bash
.venv/bin/python research_pipeline/run_full_pipeline.py
```

Main generated artifacts:
- Validation table: `research_outputs/tables/table04_validation_metrics.csv`
- Baseline comparison: `research_outputs/tables/table07_baseline_comparators.csv`
- Semisynthetic summary: `research_outputs/tables/table09_semisynthetic_summary.csv`
- Detectability summary: `research_outputs/tables/table11_detectability_thresholds.csv`
- Ablation ladder: `research_outputs/tables/table12_ablation_ladder.csv`
- Paper: `research_outputs/paper/main.pdf`
