#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

import run_co2_occupancy_analysis as core


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Physics-based robust occupancy duration estimator with uncertainty."
    )
    p.add_argument("--config", default="workflow/config.default.toml")
    p.add_argument("--subject-sensor-map", required=True)
    p.add_argument("--occupancy-file", default=None)
    p.add_argument("--output-dir", default="outputs_spring2025_physics_robust")
    p.add_argument("--n-samples", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _fit_sensor_params(df_sensor: pl.DataFrame, model: core.ModelConfig) -> dict:
    # Baseline from known-empty blocks
    bdf = df_sensor.filter(pl.col("present") == 0)
    if bdf.height >= 20:
        baseline = float(bdf.select(pl.col("co2_smooth").quantile(model.baseline_quantile)).item())
    else:
        baseline = float(df_sensor.select(pl.col("co2_smooth").quantile(0.05)).item())

    arr = (
        df_sensor.with_columns(
            [
                (pl.col("co2_smooth") - baseline).clip(lower_bound=0).alias("excess"),
                pl.col("co2_smooth").shift(1).alias("co2_prev"),
                pl.col("timestamp_min").shift(1).alias("ts_prev"),
            ]
        )
        .with_columns(
            [
                ((pl.col("timestamp_min") - pl.col("ts_prev")).dt.total_minutes()).alias("dt_min"),
                pl.col("excess").shift(1).alias("excess_prev"),
            ]
        )
    )

    decay = arr.filter(
        (pl.col("present") == 0)
        & (pl.col("dt_min") == 1)
        & pl.col("excess_prev").is_not_null()
        & (pl.col("excess_prev") > 0)
    )

    if decay.height >= 30:
        x = decay["excess_prev"].to_numpy()
        y = decay["excess"].to_numpy()
        den = float(np.sum(x * x))
        if den > 1e-6:
            phi_hat = float(np.sum(x * y) / den)
            resid = y - phi_hat * x
            n = max(2, len(x))
            sigma2 = float(np.sum(resid * resid) / (n - 1))
            se_phi = float(np.sqrt(max(1e-12, sigma2 / den)))
        else:
            phi_hat = model.phi_default
            se_phi = 0.02
    else:
        phi_hat = model.phi_default
        se_phi = 0.03

    phi_hat = float(np.clip(phi_hat, model.phi_min, model.phi_max))
    se_phi = float(np.clip(se_phi, 0.005, 0.08))

    arr = arr.with_columns(
        (pl.col("excess") - pl.lit(phi_hat) * pl.col("excess_prev")).alias("innovation")
    )

    # Innovation noise from known-empty blocks
    noise_df = arr.filter((pl.col("present") == 0) & (pl.col("dt_min") == 1))
    if noise_df.height >= 20:
        noise = noise_df["innovation"].to_numpy()
        sigma_noise = float(np.std(noise, ddof=1))
    else:
        sigma_noise = 35.0
    sigma_noise = float(np.clip(sigma_noise, 5.0, 200.0))

    room_vals = (
        df_sensor.select("room_type")
        .drop_nulls()
        .to_series()
        .to_list()
    )
    room_mode = room_vals[0] if room_vals else None

    occ_df = arr.filter((pl.col("present") == 1) & (pl.col("dt_min") == 1))
    if occ_df.height >= 20:
        pos_innov = occ_df.select(pl.col("innovation").clip(lower_bound=0).alias("p"))["p"].to_numpy()
        # robust generation scale from upper-middle quantile
        g_hat = float(np.quantile(pos_innov, model.generation_quantile))
        # robust spread via MAD
        med = float(np.median(pos_innov))
        mad = float(np.median(np.abs(pos_innov - med)))
        sigma_g = 1.4826 * mad
        se_g = float(np.clip(sigma_g / np.sqrt(max(5.0, len(pos_innov))), 0.5, 8.0))
    else:
        if room_mode == "single":
            g_hat = model.generation_single
        elif room_mode == "shared":
            g_hat = model.generation_shared
        else:
            g_hat = model.generation_default
        se_g = 4.0

    g_hat = float(np.clip(g_hat, model.generation_min, model.generation_max))
    se_g = float(np.clip(se_g, 0.5, 10.0))

    n_decay = int(decay.height)
    n_occ = int(occ_df.height)

    return {
        "baseline": baseline,
        "phi_hat": phi_hat,
        "phi_se": se_phi,
        "g_hat": g_hat,
        "g_se": se_g,
        "sigma_noise": sigma_noise,
        "n_decay_points": n_decay,
        "n_occ_points": n_occ,
        "room_type_mode": room_mode,
        "arr": arr,
    }


def _estimate_point_minutes(arr: pl.DataFrame, phi: float, g: float) -> pl.DataFrame:
    minute = (
        arr.with_columns(
            [
                (pl.col("excess") - pl.lit(phi) * pl.col("excess_prev")).alias("innovation"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("present") == 0)
                .then(pl.lit(0.0))
                .when(pl.col("dt_min") == 1)
                .then((pl.col("innovation").clip(lower_bound=0) / pl.lit(g)).clip(0.0, 1.0))
                .otherwise(pl.lit(0.0))
                .alias("occ_intensity"),
            ]
        )
    )
    return minute


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = core.load_config(Path(args.config))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graphs").mkdir(exist_ok=True)

    occupancy_file = Path(args.occupancy_file).resolve() if args.occupancy_file else cfg.paths.occupancy_csv
    subject_sensor_map = Path(args.subject_sensor_map).resolve()

    co2_files = core.discover_co2_files(cfg.paths.co2_root)
    co2_minutes = core.load_co2_minutes(co2_files)
    occupancy = core.load_occupancy(occupancy_file)
    building_key = core.load_building_key(cfg.paths.building_key_csv)
    map_df = core.load_subject_sensor_map(subject_sensor_map)
    mapped_blocks = core.map_occupancy_to_sensor(occupancy, map_df, building_key, co2_minutes)
    if mapped_blocks.height == 0:
        raise RuntimeError("No mapped blocks after applying subject-sensor mapping.")

    occ_minute = core.expand_blocks_to_minutes(mapped_blocks, cfg.model.block_hours)
    joined = (
        co2_minutes.join(occ_minute, on=["sensor_id", "timestamp_min"], how="inner")
        .sort(["sensor_id", "timestamp_min"])
        .with_columns(
            pl.col("co2_ppm")
            .rolling_median(window_size=cfg.model.smoothing_window_minutes)
            .over("sensor_id")
            .alias("co2_smooth")
        )
        .with_columns(pl.col("co2_smooth").fill_null(pl.col("co2_ppm")).alias("co2_smooth"))
    )

    sensors = joined.select("sensor_id").unique().to_series().to_list()
    sensor_param_rows = []
    block_outputs = []

    for sensor_id in sensors:
        sdf = joined.filter(pl.col("sensor_id") == sensor_id).sort("timestamp_min")
        if sdf.height < 20:
            continue
        fit = _fit_sensor_params(sdf, cfg.model)
        arr = fit["arr"]

        # Point estimate (posterior median proxy)
        point_minute = _estimate_point_minutes(arr, fit["phi_hat"], fit["g_hat"])

        # Block-level point aggregation
        block_point = (
            point_minute.group_by(
                [
                    "subject_id",
                    "sensor_id",
                    "block_date",
                    "block",
                    "block_start_ts",
                    "block_end_ts",
                    "present",
                    "group_id_deploy",
                    "Address",
                    "building_code",
                    "hallname",
                    "vent_cat",
                    "room_type",
                ]
            )
            .agg(
                [
                    pl.len().alias("data_minutes"),
                    pl.col("occ_intensity").sum().alias("minutes_point"),
                    pl.col("co2_smooth").mean().alias("mean_co2_ppm"),
                    pl.col("co2_smooth").max().alias("peak_co2_ppm"),
                ]
            )
        )

        # Monte Carlo uncertainty on each block for this sensor
        bkeys = block_point.select(
            [
                "subject_id",
                "sensor_id",
                "block_date",
                "block",
                "block_start_ts",
                "block_end_ts",
                "present",
            ]
        )
        key_tuples = [tuple(r) for r in bkeys.rows()]
        key_to_idx = {k: i for i, k in enumerate(key_tuples)}

        arr2 = arr.select(
            [
                "subject_id",
                "sensor_id",
                "block_date",
                "block",
                "block_start_ts",
                "block_end_ts",
                "present",
                "dt_min",
                "excess",
                "excess_prev",
            ]
        )
        block_idx = np.array(
            [
                key_to_idx[
                    (
                        r[0],
                        r[1],
                        r[2],
                        r[3],
                        r[4],
                        r[5],
                        r[6],
                    )
                ]
                for r in arr2.rows()
            ],
            dtype=np.int64,
        )
        present = np.array(arr2["present"], dtype=np.int8)
        dt1 = np.array(arr2["dt_min"].fill_null(0), dtype=np.float64) == 1.0
        excess = np.array(arr2["excess"].fill_null(0), dtype=np.float64)
        ex_prev = np.array(arr2["excess_prev"].fill_null(0), dtype=np.float64)

        n_blocks = len(key_tuples)
        samples = np.zeros((args.n_samples, n_blocks), dtype=np.float64)

        for s in range(args.n_samples):
            phi_s = float(
                np.clip(
                    rng.normal(fit["phi_hat"], fit["phi_se"]),
                    cfg.model.phi_min,
                    cfg.model.phi_max,
                )
            )
            g_s = float(
                np.clip(
                    rng.normal(fit["g_hat"], fit["g_se"]),
                    cfg.model.generation_min,
                    cfg.model.generation_max,
                )
            )
            innovation = excess - phi_s * ex_prev
            u = np.zeros_like(innovation)
            mask = (present == 1) & dt1
            u[mask] = np.clip(np.maximum(innovation[mask], 0.0) / max(g_s, 1e-6), 0.0, 1.0)
            # present==0 remains zero by construction
            block_sum = np.bincount(block_idx, weights=u, minlength=n_blocks)
            samples[s, :] = np.clip(block_sum, 0.0, 240.0)

        p10 = np.quantile(samples, 0.10, axis=0)
        p50 = np.quantile(samples, 0.50, axis=0)
        p90 = np.quantile(samples, 0.90, axis=0)

        block_mc = block_point.with_columns(
            [
                pl.Series("minutes_p10", p10),
                pl.Series("minutes_p50", p50),
                pl.Series("minutes_p90", p90),
            ]
        ).with_columns(
            [
                pl.when(pl.col("present") == 0).then(pl.lit(0.0)).otherwise(pl.col("minutes_p50")).alias(
                    "estimated_occupied_minutes_best"
                ),
                pl.when(pl.col("present") == 0).then(pl.lit(0.0)).otherwise(pl.col("minutes_p10")).alias(
                    "estimated_occupied_minutes_p10"
                ),
                pl.when(pl.col("present") == 0).then(pl.lit(0.0)).otherwise(pl.col("minutes_p90")).alias(
                    "estimated_occupied_minutes_p90"
                ),
                (pl.col("minutes_p90") - pl.col("minutes_p10")).alias("uncertainty_p90_p10"),
            ]
        ).with_columns(
            [
                (pl.col("estimated_occupied_minutes_best") / 240.0).alias("estimated_occupied_fraction_best"),
                pl.when(pl.col("uncertainty_p90_p10") <= 30)
                .then(pl.lit("high"))
                .when(pl.col("uncertainty_p90_p10") <= 60)
                .then(pl.lit("medium"))
                .otherwise(pl.lit("low"))
                .alias("stability_band"),
            ]
        ).with_columns(
            [
                pl.lit(fit["phi_hat"]).alias("phi_hat"),
                pl.lit(fit["phi_se"]).alias("phi_se"),
                pl.lit(fit["g_hat"]).alias("generation_hat"),
                pl.lit(fit["g_se"]).alias("generation_se"),
                pl.lit(fit["sigma_noise"]).alias("innovation_sigma"),
                pl.lit(fit["n_decay_points"]).alias("n_decay_points"),
                pl.lit(fit["n_occ_points"]).alias("n_occ_points"),
            ]
        )

        block_outputs.append(block_mc)
        sensor_param_rows.append(
            {
                "sensor_id": int(sensor_id),
                "baseline": fit["baseline"],
                "phi_hat": fit["phi_hat"],
                "phi_se": fit["phi_se"],
                "generation_hat": fit["g_hat"],
                "generation_se": fit["g_se"],
                "innovation_sigma": fit["sigma_noise"],
                "n_decay_points": fit["n_decay_points"],
                "n_occ_points": fit["n_occ_points"],
                "room_type_mode": fit["room_type_mode"],
            }
        )

    if not block_outputs:
        raise RuntimeError("No sensor outputs produced.")

    block_best = pl.concat(block_outputs, how="vertical").sort(["sensor_id", "block_date", "block"])
    sensor_params = pl.DataFrame(sensor_param_rows).sort("sensor_id")

    # Individual-day outputs (requested explicitly)
    daily = (
        block_best.group_by(["subject_id", "block_date"])
        .agg(
            [
                pl.col("estimated_occupied_minutes_best").sum().alias("daily_estimated_minutes_best"),
                pl.col("estimated_occupied_minutes_p10").sum().alias("daily_estimated_minutes_p10"),
                pl.col("estimated_occupied_minutes_p90").sum().alias("daily_estimated_minutes_p90"),
                pl.col("uncertainty_p90_p10").mean().alias("daily_mean_block_uncertainty"),
                pl.col("present").sum().alias("n_blocks_label1"),
            ]
        )
        .with_columns((pl.col("daily_estimated_minutes_best") / 1440.0).alias("daily_fraction_of_day"))
        .sort(["subject_id", "block_date"])
    )

    individual = (
        daily.group_by("subject_id")
        .agg(
            [
                pl.len().alias("n_days"),
                pl.col("daily_estimated_minutes_best").mean().alias("mean_daily_minutes_best"),
                pl.col("daily_estimated_minutes_best").median().alias("median_daily_minutes_best"),
                pl.col("daily_estimated_minutes_p10").mean().alias("mean_daily_minutes_p10"),
                pl.col("daily_estimated_minutes_p90").mean().alias("mean_daily_minutes_p90"),
                pl.col("daily_mean_block_uncertainty").mean().alias("mean_block_uncertainty"),
            ]
        )
        .sort("mean_daily_minutes_best", descending=True)
    )

    block_best.write_parquet(output_dir / "block_estimates_physics_robust.parquet")
    block_best.write_csv(output_dir / "block_estimates_physics_robust.csv")
    sensor_params.write_parquet(output_dir / "sensor_parameters_physics_robust.parquet")
    sensor_params.write_csv(output_dir / "sensor_parameters_physics_robust.csv")
    daily.write_parquet(output_dir / "daily_estimates_physics_robust.parquet")
    daily.write_csv(output_dir / "daily_estimates_physics_robust.csv")
    individual.write_csv(output_dir / "individual_summary_physics_robust.csv")

    # Minimal graph: uncertainty histogram
    unc = block_best.select("uncertainty_p90_p10").to_series().to_numpy()
    bins = np.array([0, 15, 30, 45, 60, 90, 120, 180, 240], dtype=float)
    hist, _ = np.histogram(unc, bins=bins)
    ymax = max(1, int(hist.max()))
    w, h = 1200, 700
    left, right, top, bottom = 90, 40, 70, 90
    pw, ph = w - left - right, h - top - bottom
    bw = pw / len(hist)
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='white'/>",
        "<text x='90' y='36' font-size='28' font-family='sans-serif'>Block Uncertainty (p90-p10, minutes)</text>",
        f"<line x1='{left}' y1='{top+ph}' x2='{left+pw}' y2='{top+ph}' stroke='black'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+ph}' stroke='black'/>",
    ]
    for i, c in enumerate(hist):
        x = left + i * bw
        bh = (c / ymax) * ph
        parts.append(
            f"<rect x='{x+4:.1f}' y='{top+ph-bh:.1f}' width='{bw-8:.1f}' height='{bh:.1f}' fill='#457B9D'/>"
        )
        parts.append(
            f"<text x='{x+bw/2-12:.1f}' y='{top+ph+24}' font-size='12' font-family='sans-serif'>{int(bins[i])}-{int(bins[i+1])}</text>"
        )
    parts.append("</svg>")
    (output_dir / "graphs" / "uncertainty_histogram.svg").write_text("".join(parts), encoding="utf-8")

    summary = {
        "rows_block": int(block_best.height),
        "rows_daily": int(daily.height),
        "subjects": int(block_best.select(pl.col("subject_id").n_unique()).item()),
        "sensors": int(block_best.select(pl.col("sensor_id").n_unique()).item()),
        "label1_mean_minutes_best": float(
            block_best.filter(pl.col("present") == 1)
            .select(pl.col("estimated_occupied_minutes_best").mean())
            .item()
        ),
        "label1_median_minutes_best": float(
            block_best.filter(pl.col("present") == 1)
            .select(pl.col("estimated_occupied_minutes_best").median())
            .item()
        ),
        "label0_mean_minutes_best": float(
            block_best.filter(pl.col("present") == 0)
            .select(pl.col("estimated_occupied_minutes_best").mean())
            .item()
        ),
        "label1_mean_uncertainty_width": float(
            block_best.filter(pl.col("present") == 1)
            .select(pl.col("uncertainty_p90_p10").mean())
            .item()
        ),
        "stability_counts": {
            row[0]: int(row[1]) for row in block_best.group_by("stability_band").len().rows()
        },
    }
    (output_dir / "summary_physics_robust.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
