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

For each minute during a labeled-present block, the occupancy intensity was computed as:

    u_hat(t) = clip(max(innovation(t), 0) / g, 0, 1)

where innovation(t) = excess(t) - phi * excess(t-1). The estimated occupied minutes within each
4-hour block was the sum of minute-level intensities.

## Uncertainty Quantification

Parameter uncertainty was propagated via Monte Carlo simulation (N=200).
For each simulation, phi and g were drawn from normal distributions centered on their
point estimates with standard errors derived from the fitting residuals. Block-level estimates
were summarized as posterior median (p50) with 80% credible intervals (p10, p90).

## CO2 Smoothing

Raw CO2 measurements were smoothed using a 5-minute rolling median
to reduce sensor noise while preserving occupancy-driven transitions.
