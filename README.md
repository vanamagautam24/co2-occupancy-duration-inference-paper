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
    E --> F[Monte Carlo Uncertainty]
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

## Additional Result Previews
- Validation + Baselines: `assets/gifs/validation_and_baselines.gif`
- Distribution / Uncertainty / Semisynthetic: `assets/gifs/results_overview.gif`

## Key Results Snapshot (Latest Run)
Source: `research_outputs/tables/table04_validation_metrics.csv`

| Metric | Value | Criterion | Pass |
|---|---:|---:|:---:|
| C1 in-sample mean unclamped (label=0) | 11.19 min | < 15 | YES |
| C2 in-sample % under 5 min | 80.3% | > 60% | YES |
| C1-OOS mean unclamped (label=0) | 19.54 min | < 20 | YES |
| C2-OOS % under 5 min | 77.2% | > 50% | YES |
| C1-OOS median unclamped | 0.00 min | report | NA |
| C1-OOS 5% trimmed mean | 12.08 min | report | NA |
| C1-OOS tail contribution (top 5%) | 53.3% | report | NA |
| E1 LOO mean relative MAE | 1.40 | < 1.5 | YES |
| E2 LOO mean Spearman rho | 0.314 | > 0.2 | YES |
| E3 max phi sensitivity change | 76.5% | < 200% | YES |
| E4 high-confidence blocks | 49.5% | > 40% | YES |

## Semisynthetic CI Coverage (Latest Run)
Source: `research_outputs/tables/table10_semisynthetic_coverage.csv`

| Group | Coverage (80% CI) | Mean CI Width | MAE |
|---|---:|---:|---:|
| Continuous | 56.8% | 57.3 min | 44.5 min |
| Fragmented | 51.9% | 83.4 min | 54.9 min |
| Stress | 53.1% | 75.4 min | 45.3 min |
| Overall | 54.4% | 69.4 min | 46.6 min |

## Baseline Comparator Note
Source: `research_outputs/tables/table07_baseline_comparators.csv`

Baseline comparison is computed on a comparable room-level subset (`n_label0=177`, `n_label1=260`) for method-to-method comparability.

## Reproducing
```bash
.venv/bin/python research_pipeline/run_full_pipeline.py
```

Main generated artifacts:
- Validation table: `research_outputs/tables/table04_validation_metrics.csv`
- Baseline comparison: `research_outputs/tables/table07_baseline_comparators.csv`
- Semisynthetic summary: `research_outputs/tables/table09_semisynthetic_summary.csv`
- Paper: `research_outputs/paper/main.pdf`
