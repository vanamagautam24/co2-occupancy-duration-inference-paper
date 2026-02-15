# CO2 Occupancy Inference: Data, Problem Statement, and Current Findings

## 1) Problem Statement

We have block-level occupancy labels and minute-level CO2 data.

- Occupancy label format:
  - `value = 0`: subject not in room for that 4-hour block
  - `value = 1`: subject in room at some unknown duration within that 4-hour block
- Block starts are hourly keys:
  - `2` means 2:00-6:00
  - `6` means 6:00-10:00
  - `10`, `14`, `18`, `22` follow the same 4-hour logic

The core question:

Estimate **occupied minutes within each 4-hour block** using only CO2 dynamics, especially when label=`1` does not tell duration.

## 2) Data Inventory

### A) CO2 raw data

- Path pattern: `CO2/CO2_YYYY-MM-DD/*.csv`
- Valid files used: `138`
- Unique sensors in CO2 pool: `78`
- Minute-level rows after cleaning/aggregation: `1,398,359`
- Date coverage in CO2 pool: `2025-02-03` to `2025-04-30`

Required columns used from each raw CSV:
- `Datetime`, `CO2`, `Temp`, `RH`, `Dew`

### B) Occupancy labels

- File: `gf2-y2-dailyoccupancy-updated.csv`
- Rows used after parsing/filtering: `131,574`
- Unique subjects: `183`
- Date coverage: `2024-09-03` to `2025-05-13`
- Label counts:
  - `0`: `47,204`
  - `1`: `84,370`

### C) Subject to sensor mapping (crosswalk)

- Source workbook provided: `InfectionsDatabase_2024_2025.xlsx`
- Extracted spring crosswalk file: `data/subject_sensor_map_from_infections_spring2025.csv`
- Crosswalk rows: `26`
- Crosswalk subjects: `26`
- Crosswalk sensors: `13`

Mapping QC (spring scope):
- mapped: `24` subjects
- no date overlap with CO2: `2` subjects
  - subject `437` and `438` mapped to sensor `20233656`, but occupancy dates do not overlap this sensor's CO2 date window

QC file:
- `outputs_spring2025_infections/mapping_qc.csv`

## 3) Spring Analysis Cohort Actually Modeled

After applying subject-sensor mapping plus deployment/date overlap:

- Mapped occupancy block rows: `791`
- Subjects modeled: `24`
- Sensors modeled: `12`
- Date coverage: `2025-02-04` to `2025-04-13`
- Block labels in modeled subset:
  - `present=0`: `252`
  - `present=1`: `539`

Room-block composition in mapped data:
- one subject record, zero present: `45`
- one subject record, one present: `58`
- two subject records, zero present: `39`
- two subject records, one present: `129`
- two subject records, two present: `176`

Interpretation:
- Shared-room structure is important.
- Subject-level `0` is not always room-empty in shared settings.

## 4) Methods Implemented

### Method 1: Physics robust estimator (primary)

Script:
- `workflow/run_physics_robust_estimator.py`

Idea:
- Fit sensor baseline and decay from known-empty periods.
- Infer minute-level occupancy intensity from innovation term.
- Aggregate to block occupied minutes.
- Use Monte Carlo parameter sampling for uncertainty intervals.

Primary outputs:
- `outputs_spring2025_physics_robust/block_estimates_physics_robust.csv`
- `outputs_spring2025_physics_robust/daily_estimates_physics_robust.csv`
- `outputs_spring2025_physics_robust/individual_summary_physics_robust.csv`
- `outputs_spring2025_physics_robust/summary_physics_robust.json`

### Method 2: Room decomposition estimator (shared-room sensitivity)

Script:
- `workflow/run_room_decomposition_estimator.py`

Idea:
- Infer room-level person-minutes first.
- Allocate person-minutes to subjects in shared blocks with uncertainty-aware sampling.

Outputs:
- `outputs_spring2025_room_decomposition/block_estimates_room_decomposition.csv`
- `outputs_spring2025_room_decomposition/daily_estimates_room_decomposition.csv`
- `outputs_spring2025_room_decomposition/individual_summary_room_decomposition.csv`
- `outputs_spring2025_room_decomposition/summary_room_decomposition.json`

### Method 3: Consensus estimator (optional sensitivity)

Script:
- `workflow/run_consensus_estimator.py`

Idea:
- Combine Method 1 and Method 2 with uncertainty-weighting and model-disagreement penalty.

Outputs:
- `outputs_spring2025_consensus/block_estimates_consensus.csv`
- `outputs_spring2025_consensus/daily_estimates_consensus.csv`
- `outputs_spring2025_consensus/individual_summary_consensus.csv`
- `outputs_spring2025_consensus/summary_consensus.json`

## 5) What Was Found from the Dataset

### A) Primary model (physics robust) summary

From `outputs_spring2025_physics_robust/summary_physics_robust.json`:

- subject-block rows: `774`
- subject-day rows: `144`
- subjects: `24`
- sensors: `12`
- `present=1` mean estimated minutes: `44.20` per 4-hour block
- `present=1` median estimated minutes: `47.22`
- `present=0` mean estimated minutes: `0.00` (hard-constrained)
- mean uncertainty width (p90-p10) for `present=1`: `34.41` minutes
- stability counts (all rows):
  - high: `548`
  - medium: `76`
  - low: `150`

Additional primary findings:
- `present=1` blocks: `535`
- high-confidence `present=1` blocks (`uncertainty <= 30`): `309` (57.8%)
- daily estimated minutes:
  - mean: `164.22`
  - median: `165.07`
- mean daily minutes by block start hour (`present=1`):
  - 2:00 block: `50.06`
  - 6:00 block: `43.38`
  - 10:00 block: `39.56`
  - 14:00 block: `39.25`
  - 18:00 block: `38.80`
  - 22:00 block: `48.20`

### B) High-confidence subset (primary model)

Conservative outputs created:
- `outputs_spring2025_physics_robust/block_estimates_physics_robust_high_confidence.csv`
- `outputs_spring2025_physics_robust/daily_estimates_physics_robust_high_confidence.csv`
- `outputs_spring2025_physics_robust/individual_summary_physics_robust_high_confidence.csv`

Counts:
- high-confidence subject-block rows: `309`
- high-confidence subject-day rows: `55`
- subjects with high-confidence days: `17`

### C) Sensitivity comparison

Room decomposition summary:
- `present=1` mean: `73.22`
- mean uncertainty width: `62.96`

Consensus summary:
- `present=1` mean: `43.60`
- mean uncertainty width: `57.01`
- mean disagreement between methods (label=1): `49.69` minutes

Interpretation:
- Different plausible formulations produce materially different point estimates in many blocks.
- This is expected without minute-level ground truth and with shared-room ambiguity.

## 6) Robustness Assessment

Short answer:

- Robust enough for **population-level trends and relative comparisons**.
- Not yet robust enough for **strong per-block exact-duration claims** in every case.

Why this is still usable:
- Hard constraints are respected (`present=0 -> 0` minutes).
- Uncertainty is explicitly quantified per block.
- Per-subject and per-day outputs are available.
- A high-confidence subset is available for conservative reporting.

Why caution is still needed:
- No minute-level ground-truth occupancy for direct calibration.
- Shared-room identifiability remains a core source of uncertainty.
- Method disagreement is non-trivial in a substantial fraction of blocks.

## 7) Recommended Reporting Strategy for a Paper Draft

1. Use `physics_robust` as the primary analysis.
2. Report high-confidence subset results as conservative estimates.
3. Use room-decomposition and consensus as sensitivity analyses.
4. Avoid claiming exact minute truth for each block.
5. Frame results as best-estimate inference with uncertainty intervals.

## 8) Reproducibility Commands

Primary model:

```bash
python3 workflow/run_physics_robust_estimator.py \
  --config workflow/config.default.toml \
  --subject-sensor-map data/subject_sensor_map_from_infections_spring2025.csv \
  --occupancy-file gf2-y2-dailyoccupancy-updated.csv \
  --output-dir outputs_spring2025_physics_robust \
  --n-samples 120 \
  --seed 42
```

Shared-room sensitivity:

```bash
python3 workflow/run_room_decomposition_estimator.py \
  --config workflow/config.default.toml \
  --subject-sensor-map data/subject_sensor_map_from_infections_spring2025.csv \
  --occupancy-file gf2-y2-dailyoccupancy-updated.csv \
  --output-dir outputs_spring2025_room_decomposition \
  --n-samples 120 \
  --seed 42
```

Consensus:

```bash
python3 workflow/run_consensus_estimator.py \
  --physics-robust-dir outputs_spring2025_physics_robust \
  --room-decomp-dir outputs_spring2025_room_decomposition \
  --output-dir outputs_spring2025_consensus
```

## 9) Methods Worth Trying Next (for your research round)

These are high-value candidates to improve robustness:

1. Hierarchical Bayesian CO2 mass-balance model:
   - latent ventilation + latent occupancy intensity
   - partial pooling across sensors/subjects
2. Hidden semi-Markov model (HSMM):
   - explicit occupancy bout durations inside each 4-hour block
3. Weakly supervised contrastive sequence model:
   - learn occupancy signatures from labeled-empty and labeled-present blocks
4. Joint roommate-coupled inference:
   - constrain shared-room subjects with coupled latent states
5. Calibration transfer model across sensors:
   - sensor-specific drift/bias correction before inference

Note on quantum:
- There is currently no clear evidence that quantum computing will outperform these classical methods for this dataset scale and model structure.
- If explored, quantum should be treated as an auxiliary optimization benchmark, not the primary inference engine.

