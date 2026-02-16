# Methods: CO2-Based Occupancy Duration Estimation

## Study Design

Indoor CO2 concentrations were measured using deployable sensors placed in university dormitory rooms.
Each sensor recorded CO2 concentration at approximately 1-minute intervals. Occupancy was assessed
via self-reported 4-hour block labels (present/absent) for each subject.

## CO2 Occupancy Model

We modeled indoor CO2 dynamics using a first-order autoregressive (AR(1)) process. The excess CO2
concentration above the ambient baseline at time t follows:

    excess(t) = phi * excess(t-1) + g * u(t) + epsilon(t)

where phi is the minute-level decay rate reflecting room air exchange, g is the CO2 generation rate
per person per minute of presence, u(t) is the occupancy intensity (fraction of minute t the subject
was present, 0 to 1), and epsilon(t) is measurement noise.

## Parameter Estimation

**Baseline**: For each sensor, the ambient CO2 baseline was estimated as the 1st
percentile of all CO2 readings across the sensor's entire monitoring period. This captures
true outdoor/ambient air levels (~400-420 ppm) rather than residual CO2 from prior occupancy
during nominally empty periods. A floor of 380 ppm was enforced for physical plausibility.

**Decay rate (phi)**: Estimated via ordinary least squares regression on consecutive minute pairs
during known-empty periods where excess CO2 was decaying. Across 12 sensors, the fitted
decay rate ranged from 0.9697 to 0.9950 (median 0.9894), corresponding
to effective air exchange rates of 0.3 to 1.8 per hour.

**Generation rate (g)**: Estimated as the maximum of three candidates: (1) a steady-state estimate
from the 75th percentile of occupied-period excess CO2, (2) the 80th
percentile of positive innovations during occupied periods, and (3) a floor of 1.0 ppm/min.
Values ranged from 5.4 to 60.0 ppm/min (median 19.0).
The estimate is clipped to [1.0, 60.0] ppm/min.

## Occupancy Duration Estimation

The primary point estimate is level-based: carryover-adjusted block excess is calibrated
against the empty-room floor, normalized by the finite-window scale, and mapped to minutes:

    E_cal = max(E_adj - F_empty, 0)
    M_level = clip((E_cal / S) * 240, 0, 240)

To suppress false positives in empty blocks, an innovation significance gate and cap are then applied:

    S_b = sum_t [ innovation(t) - mu_empty ]
    innovation(t) = excess(t) - phi * excess(t-1)
    M_innov = clip(S_b / g, 0, 240)
    if S_b <= z * sigma_empty * sqrt(m_b): M_hat = 0
    else: M_hat = min(M_level, M_innov + 15)

where z = 1.28 and m_b is the number of valid consecutive innovation pairs in the block.
The optional floor-clipped innovation rescue was disabled in this run.


## Uncertainty Quantification

Parameter uncertainty was propagated via Monte Carlo simulation (N=200).
For each simulation, phi and g were drawn from normal distributions centered on their
point estimates with standard errors derived from the fitting residuals. In addition, structural
mismatch terms were sampled (enabled):
generation-rate multiplier, additive phi drift, baseline-excess shift, and innovation-noise
scale inflation. Block-level estimates were summarized as posterior median (p50) with 80%
credible intervals (p10, p90). For semisynthetic stress evaluation, we additionally report
split-conformal calibrated intervals (evaluation split only) to assess empirical coverage
under modeled mismatch families. Conformal padding uses the finite-sample split-conformal
order statistic with nominal alpha = 0.20.

## CO2 Smoothing

Raw CO2 measurements were smoothed using a 5-minute rolling median
to reduce sensor noise while preserving occupancy-driven transitions.
