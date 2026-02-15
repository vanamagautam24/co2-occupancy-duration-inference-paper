# CO2 Occupancy Duration Workflow (Research/Production Grade)

This workflow estimates **occupied minutes** inside each 4-hour occupancy block using minute-level CO2 time series.

It is designed for your exact setup:
- block label `present=0`: occupant not in room for that block
- block label `present=1`: occupant in room for an unknown duration inside that block
- block index by hour (`2` = 2:00 start, `6` = 6:00 start, etc.)

## Why this is the right approach (and not quantum-first)

For this problem, a constrained physical/statistical model is stronger than quantum computing:
- you need interpretable durations and uncertainty per block
- you have strong structure (mass-balance dynamics + hard `0/1` block constraints)
- the data size and model class are ideal for classical optimization/inference

Quantum methods are not expected to improve this pipeline today unless a very specific combinatorial subproblem is proven to bottleneck and benefit from QUBO mapping.

## Data Contract

### 1) CO2 data (already in this repo)
- path pattern: `CO2/CO2_YYYY-MM-DD/*.csv`
- columns required: `Datetime, CO2, Temp, RH, Dew`

### 2) Occupancy block labels (current file)
- file used: `gf2-y2-dailyoccupancy-updated.csv`
- required columns: `subject_id,date,block,value`
- expected blocks: `2,6,10,14,18,22` (4-hour windows)
- values: `0` (not in room) / `1` (in room, unknown duration)

### 3) Subject-to-sensor crosswalk (required)

You must provide a CSV mapping occupancy `subject_id` to CO2 `sensor_id`.

Minimum required columns:
- `subject_id`
- `sensor_id`

Optional columns:
- `map_start_date` (or `start_date`) to bound mapping validity
- `map_end_date` (or `end_date`) to bound mapping validity
- `room_type` (`single` or `shared`)

If you have `InfectionsDatabase_2024_2025.xlsx`, you can auto-extract a draft crosswalk:

```bash
python3 workflow/extract_spring_crosswalk_from_infections.py \
  --infile InfectionsDatabase_2024_2025.xlsx \
  --outfile data/subject_sensor_map_from_infections_spring2025.csv
```

## Model Summary

For each sensor, minute-level CO2 is modeled as:

`E_t = phi * E_(t-1) + g * u_t + noise`

where:
- `E_t`: excess CO2 above sensor baseline
- `phi`: decay factor estimated from known unoccupied (`present=0`) periods
- `g`: effective per-minute generation scale (estimated from occupied-labeled blocks, with room-type defaults)
- `u_t` in [0, 1]: latent occupancy intensity per minute

We estimate:
- `u_t = clip((E_t - phi * E_(t-1)) / g, 0, 1)`
- block occupied minutes = sum of `u_t` within the block

Hard constraints:
- if `present=0`, estimated occupied minutes forced to 0
- if `present=1`, estimate is >= configurable minimum when enough data exists

## Outputs

Running the pipeline writes:
- `outputs/mapped_blocks.parquet`
- `outputs/sensor_parameters.parquet`
- `outputs/minute_estimates.parquet`
- `outputs/block_estimates.parquet`
- `outputs/summary.json`
- `outputs/graphs/*.svg`

Additional robust/sensitivity workflows now included:
- `workflow/run_physics_robust_estimator.py`
  - sensor-level parameter uncertainty + block-level Monte Carlo intervals
  - outputs per-block, per-day, and per-individual summaries
- `workflow/run_room_decomposition_estimator.py`
  - room-level CO2 inversion, then subject allocation in shared blocks
  - useful as a sensitivity analysis for shared-room identifiability
- `workflow/run_consensus_estimator.py`
  - combines robust + room-decomposition outputs with disagreement penalties

## Run

1) Install deps:

```bash
pip install -r workflow/requirements.txt
```

2) Run:

```bash
python3 workflow/run_co2_occupancy_analysis.py \
  --config workflow/config.default.toml \
  --subject-sensor-map data/subject_sensor_map.csv
```

You can override any path directly:

```bash
python3 workflow/run_co2_occupancy_analysis.py \
  --config workflow/config.default.toml \
  --occupancy-file gf2-y2-dailyoccupancy-updated.csv \
  --subject-sensor-map data/subject_sensor_map.csv \
  --output-dir outputs
```

Use `workflow/subject_sensor_crosswalk_template.csv` as a template.

## Recommended Run Order (Paper Draft)

1) Primary estimator (most stable uncertainty):

```bash
python3 workflow/run_physics_robust_estimator.py \
  --config workflow/config.default.toml \
  --subject-sensor-map data/subject_sensor_map_from_infections_spring2025.csv \
  --occupancy-file gf2-y2-dailyoccupancy-updated.csv \
  --output-dir outputs_spring2025_physics_robust
```

2) Shared-room sensitivity estimator:

```bash
python3 workflow/run_room_decomposition_estimator.py \
  --config workflow/config.default.toml \
  --subject-sensor-map data/subject_sensor_map_from_infections_spring2025.csv \
  --occupancy-file gf2-y2-dailyoccupancy-updated.csv \
  --output-dir outputs_spring2025_room_decomposition
```

3) Optional consensus layer:

```bash
python3 workflow/run_consensus_estimator.py \
  --physics-robust-dir outputs_spring2025_physics_robust \
  --room-decomp-dir outputs_spring2025_room_decomposition \
  --output-dir outputs_spring2025_consensus
```

All three produce per-individual, per-day CSV outputs.

For conservative paper tables, you can also use:
- `outputs_spring2025_physics_robust/block_estimates_physics_robust_high_confidence.csv`
- `outputs_spring2025_physics_robust/daily_estimates_physics_robust_high_confidence.csv`
- `outputs_spring2025_physics_robust/individual_summary_physics_robust_high_confidence.csv`

## Validation Guidance for Paper Work

- Hold out a subset of rooms/days with known occupancy duration (if available) for evaluation.
- Compare against simple baselines:
  - `0 minutes` for all `present=1` blocks
  - fixed duration prior per block (for example 120 min)
  - threshold-only CO2 rise method
- Report:
  - MAE/RMSE on occupied minutes
  - calibration of uncertainty/quality flags
  - stratified metrics for `single` vs `shared`
