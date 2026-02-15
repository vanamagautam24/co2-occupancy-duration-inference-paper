#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

import run_co2_occupancy_analysis as core


BLOCK_KEY_COLS = [
    "sensor_id",
    "block_date",
    "block",
    "block_start_ts",
    "block_end_ts",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Room-level CO2 physics estimator with subject-level decomposition for shared rooms."
        )
    )
    p.add_argument("--config", default="workflow/config.default.toml")
    p.add_argument("--subject-sensor-map", required=True)
    p.add_argument("--occupancy-file", default=None)
    p.add_argument("--output-dir", default="outputs_spring2025_room_decomposition")
    p.add_argument("--n-samples", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--share-alpha",
        type=float,
        default=24.0,
        help="Concentration for subject share sampling in shared blocks (larger -> tighter shares).",
    )
    return p.parse_args()


def _fit_sensor_params(df_sensor: pl.DataFrame, model: core.ModelConfig) -> dict:
    empty = df_sensor.filter(pl.col("room_empty_all_mapped") == 1)
    if empty.height >= 20:
        baseline = float(
            empty.select(pl.col("co2_smooth").quantile(model.baseline_quantile)).item()
        )
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
        (pl.col("room_empty_all_mapped") == 1)
        & (pl.col("dt_min") == 1)
        & pl.col("excess_prev").is_not_null()
        & (pl.col("excess_prev") > 0)
    )

    if decay.height >= 30:
        x = decay["excess_prev"].to_numpy()
        y = decay["excess"].to_numpy()
        den = float(np.sum(x * x))
        if den > 1e-6:
            phi_ls = float(np.sum(x * y) / den)
            resid = y - phi_ls * x
            sigma2 = float(np.sum(resid * resid) / max(1, len(x) - 1))
            se_phi = float(np.sqrt(max(1e-12, sigma2 / den)))
        else:
            phi_ls = model.phi_default
            se_phi = 0.02
        # Light shrinkage to default avoids overfitting near-1 decay in noisy sensors.
        lam = 180.0 / (180.0 + float(len(x)))
        phi_hat = float(lam * model.phi_default + (1.0 - lam) * phi_ls)
    else:
        phi_hat = model.phi_default
        se_phi = 0.03

    phi_cap = min(model.phi_max, 0.998)
    phi_hat = float(np.clip(phi_hat, model.phi_min, phi_cap))
    se_phi = float(np.clip(se_phi, 0.005, 0.08))

    arr = arr.with_columns(
        (pl.col("excess") - pl.lit(phi_hat) * pl.col("excess_prev")).alias("innovation")
    )

    # Prefer one-subject blocks to calibrate per-person generation.
    solo = arr.filter(
        (pl.col("n_present_subjects") == 1)
        & (pl.col("dt_min") == 1)
        & (pl.col("innovation") > 0)
    )
    occ_any = arr.filter(
        (pl.col("room_present_any") == 1)
        & (pl.col("dt_min") == 1)
        & (pl.col("innovation") > 0)
    )
    gen_df = solo if solo.height >= 20 else occ_any
    n_solo = int(solo.height)

    if gen_df.height >= 20:
        vals = gen_df["innovation"].to_numpy()
        g_hat = float(np.quantile(vals, model.generation_quantile))
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        sigma_g = 1.4826 * mad
        se_g = float(np.clip(sigma_g / np.sqrt(max(5.0, len(vals))), 0.5, 8.0))
    else:
        room_vals = (
            df_sensor.select("room_type")
            .drop_nulls()
            .to_series()
            .to_list()
        )
        room_mode = room_vals[0] if room_vals else None
        if room_mode == "single":
            g_hat = model.generation_single
        elif room_mode == "shared":
            g_hat = model.generation_shared
        else:
            g_hat = model.generation_default
        se_g = 4.0

    g_hat = float(np.clip(g_hat, model.generation_min, model.generation_max))
    se_g = float(np.clip(se_g, 0.5, 10.0))

    max_occ = int(
        df_sensor.select(pl.col("n_subject_records").max()).item()
        if df_sensor.height
        else 1
    )
    max_occ = int(np.clip(max_occ, 1, 4))

    return {
        "baseline": baseline,
        "phi_hat": phi_hat,
        "phi_se": se_phi,
        "g_hat": g_hat,
        "g_se": se_g,
        "n_decay_points": int(decay.height),
        "n_solo_points": n_solo,
        "max_occ": max_occ,
        "arr": arr,
    }


def _estimate_room_equiv(arr: pl.DataFrame, phi: float, g: float) -> pl.DataFrame:
    out = (
        arr.with_columns(
            (pl.col("excess") - pl.lit(phi) * pl.col("excess_prev")).alias("innovation")
        )
        .with_columns(
            [
                pl.when(pl.col("dt_min") == 1)
                .then((pl.col("innovation").clip(lower_bound=0) / pl.lit(g)).clip(0.0, 10.0))
                .otherwise(pl.lit(0.0))
                .alias("occ_equiv_raw"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("occ_equiv_raw") > pl.col("n_subject_records").cast(pl.Float64))
                .then(pl.col("n_subject_records").cast(pl.Float64))
                .otherwise(pl.col("occ_equiv_raw"))
                .alias("occ_equiv"),
            ]
        )
    )
    return out


def _get_prior_weights(
    subj_blocks: pl.DataFrame,
    room_point: pl.DataFrame,
) -> dict[int, float]:
    joined = subj_blocks.join(
        room_point.select(BLOCK_KEY_COLS + ["room_person_minutes_point", "n_present_subjects"]),
        on=BLOCK_KEY_COLS,
        how="left",
    )
    solo = joined.filter(
        (pl.col("present") == 1)
        & (pl.col("n_present_subjects") == 1)
        & pl.col("room_person_minutes_point").is_not_null()
    )
    subjects = subj_blocks.select("subject_id").unique().to_series().to_list()
    if not subjects:
        return {}

    if solo.height == 0:
        return {int(s): 1.0 for s in subjects}

    w = (
        solo.group_by("subject_id")
        .agg((pl.col("room_person_minutes_point").mean() + 1.0).alias("w"))
        .with_columns(pl.col("w").clip(lower_bound=0.2))
    )
    w_map = {int(r[0]): float(r[1]) for r in w.rows()}
    out = {}
    for s in subjects:
        out[int(s)] = float(w_map.get(int(s), 1.0))
    return out


def _prepare_allocation(
    subj_blocks: pl.DataFrame,
    room_point: pl.DataFrame,
    prior_weights: dict[int, float],
    share_alpha: float,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    key_rows = room_point.select(BLOCK_KEY_COLS).rows()
    key_to_idx = {tuple(r): i for i, r in enumerate(key_rows)}

    row_keys = subj_blocks.select(BLOCK_KEY_COLS).rows()
    row_block_idx = np.array([key_to_idx[tuple(k)] for k in row_keys], dtype=np.int64)
    row_present = np.array(subj_blocks["present"], dtype=np.int8)
    row_subject = np.array(subj_blocks["subject_id"], dtype=np.int64)

    n_blocks = len(key_rows)
    active_rows_by_block: list[np.ndarray] = []
    alpha_by_block: list[np.ndarray] = []
    for b in range(n_blocks):
        rows = np.where((row_block_idx == b) & (row_present == 1))[0]
        active_rows_by_block.append(rows)
        if len(rows) == 0:
            alpha_by_block.append(np.zeros(0, dtype=float))
            continue
        pri = np.array([prior_weights.get(int(row_subject[r]), 1.0) for r in rows], dtype=float)
        pri = np.clip(pri, 0.2, None)
        alpha = np.clip(pri / np.mean(pri) * share_alpha, 0.5, 120.0)
        alpha_by_block.append(alpha)

    return row_block_idx, active_rows_by_block, alpha_by_block, row_present, row_subject


def _allocate_subject_samples(
    rng: np.random.Generator,
    room_samples: np.ndarray,
    row_block_idx: np.ndarray,
    active_rows_by_block: list[np.ndarray],
    alpha_by_block: list[np.ndarray],
) -> np.ndarray:
    n_samples, _ = room_samples.shape
    n_rows = len(row_block_idx)
    out = np.zeros((n_samples, n_rows), dtype=np.float64)

    for b, rows in enumerate(active_rows_by_block):
        if len(rows) == 0:
            continue

        pm = room_samples[:, b]
        if len(rows) == 1:
            out[:, rows[0]] = np.clip(pm, 0.0, 240.0)
            continue

        alpha = alpha_by_block[b]
        if len(rows) == 2:
            share = rng.beta(alpha[0], alpha[1], size=n_samples)
            a = np.clip(pm * share, 0.0, 240.0)
            bval = np.clip(pm * (1.0 - share), 0.0, 240.0)
            out[:, rows[0]] = a
            out[:, rows[1]] = bval
            continue

        for s in range(n_samples):
            gam = rng.gamma(shape=alpha, scale=1.0)
            shares = gam / np.sum(gam)
            vals = np.clip(pm[s] * shares, 0.0, 240.0)
            out[s, rows] = vals

    return out


def _save_uncertainty_hist(block_est: pl.DataFrame, out_path: Path) -> None:
    unc = block_est.select("uncertainty_p90_p10").to_series().to_numpy()
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
        "<text x='90' y='36' font-size='28' font-family='sans-serif'>Subject-Block Uncertainty (p90-p10, minutes)</text>",
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
    out_path.write_text("".join(parts), encoding="utf-8")


def _save_individual_bar(individual: pl.DataFrame, out_path: Path) -> None:
    if individual.height == 0:
        return
    labels = [str(v) for v in individual["subject_id"].to_list()]
    vals = np.array(individual["mean_daily_minutes_best"].to_list(), dtype=float)
    vmax = float(max(1.0, np.nanmax(vals)))

    w, h = 1400, 760
    left, right, top, bottom = 90, 40, 70, 130
    pw, ph = w - left - right, h - top - bottom
    bw = pw / max(1, len(vals))

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='white'/>",
        "<text x='90' y='36' font-size='28' font-family='sans-serif'>Mean Estimated Daily Minutes by Subject</text>",
        f"<line x1='{left}' y1='{top+ph}' x2='{left+pw}' y2='{top+ph}' stroke='black'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+ph}' stroke='black'/>",
    ]

    for i in range(6):
        yv = vmax * i / 5.0
        y = top + ph - (yv / vmax) * ph
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left+pw}' y2='{y:.1f}' stroke='#e8e8e8'/>")
        parts.append(f"<text x='42' y='{y+4:.1f}' font-size='12' font-family='sans-serif'>{int(round(yv))}</text>")

    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = left + i * bw
        bh = (v / vmax) * ph
        parts.append(
            f"<rect x='{x+2:.1f}' y='{top+ph-bh:.1f}' width='{bw-4:.1f}' height='{bh:.1f}' fill='#2A9D8F'/>"
        )
        parts.append(
            f"<text x='{x+bw/2-10:.1f}' y='{top+ph+18}' font-size='10' font-family='sans-serif' transform='rotate(65,{x+bw/2-10:.1f},{top+ph+18})'>{lab}</text>"
        )

    parts.append("</svg>")
    out_path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = core.load_config(Path(args.config))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graphs").mkdir(exist_ok=True)

    occupancy_file = (
        Path(args.occupancy_file).resolve() if args.occupancy_file else cfg.paths.occupancy_csv
    )
    subject_sensor_map = Path(args.subject_sensor_map).resolve()

    co2_files = core.discover_co2_files(cfg.paths.co2_root)
    co2_minutes = core.load_co2_minutes(co2_files)
    occupancy = core.load_occupancy(occupancy_file)
    building_key = core.load_building_key(cfg.paths.building_key_csv)
    map_df = core.load_subject_sensor_map(subject_sensor_map)
    mapped_blocks = core.map_occupancy_to_sensor(occupancy, map_df, building_key, co2_minutes)
    if mapped_blocks.height == 0:
        raise RuntimeError("No mapped blocks after applying subject-sensor mapping.")

    subject_blocks = (
        mapped_blocks.with_columns(
            [
                pl.col("date").alias("block_date"),
                (
                    pl.col("date").cast(pl.Datetime)
                    + pl.duration(hours=pl.col("block").cast(pl.Int64))
                ).alias("block_start_ts"),
            ]
        )
        .with_columns(
            (pl.col("block_start_ts") + pl.duration(hours=cfg.model.block_hours)).alias(
                "block_end_ts"
            )
        )
        .select(
            [
                "subject_id",
                "sensor_id",
                "block_date",
                "block",
                "block_start_ts",
                "block_end_ts",
                "present",
                "room_type",
                "group_id_deploy",
                "Address",
                "building_code",
                "hallname",
                "vent_cat",
            ]
        )
        .sort(["sensor_id", "block_date", "block", "subject_id"])
    )

    room_blocks = (
        subject_blocks.group_by(
            BLOCK_KEY_COLS
            + [
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
                pl.col("present").sum().alias("n_present_subjects"),
                pl.len().alias("n_subject_records"),
            ]
        )
        .with_columns(
            [
                (pl.col("n_present_subjects") > 0).cast(pl.Int8).alias("room_present_any"),
                (pl.col("n_present_subjects") == 0).cast(pl.Int8).alias("room_empty_all_mapped"),
            ]
        )
        .sort(["sensor_id", "block_date", "block"])
    )

    room_minute = (
        room_blocks.with_columns(
            pl.datetime_ranges(
                pl.col("block_start_ts"),
                pl.col("block_end_ts"),
                interval="1m",
                closed="left",
            ).alias("timestamp_min")
        )
        .explode("timestamp_min")
        .sort(["sensor_id", "timestamp_min"])
    )

    joined = (
        co2_minutes.join(room_minute, on=["sensor_id", "timestamp_min"], how="inner")
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
    room_block_outputs = []
    subject_block_outputs = []

    for sensor_id in sensors:
        sdf = joined.filter(pl.col("sensor_id") == sensor_id).sort("timestamp_min")
        if sdf.height < 20:
            continue
        fit = _fit_sensor_params(sdf, cfg.model)
        arr = fit["arr"]

        point_minute = _estimate_room_equiv(arr, fit["phi_hat"], fit["g_hat"])

        room_point = (
            point_minute.group_by(
                BLOCK_KEY_COLS
                + [
                    "group_id_deploy",
                    "Address",
                    "building_code",
                    "hallname",
                    "vent_cat",
                    "room_type",
                    "n_present_subjects",
                    "n_subject_records",
                ]
            )
            .agg(
                [
                    pl.len().alias("data_minutes"),
                    pl.col("occ_equiv").sum().alias("room_person_minutes_point"),
                    pl.col("co2_smooth").mean().alias("mean_co2_ppm"),
                    pl.col("co2_smooth").max().alias("peak_co2_ppm"),
                ]
            )
            .sort(["block_date", "block"])
        )
        if room_point.height == 0:
            continue

        bkeys = room_point.select(BLOCK_KEY_COLS)
        key_tuples = [tuple(r) for r in bkeys.rows()]
        key_to_idx = {k: i for i, k in enumerate(key_tuples)}

        arr2 = arr.select(
            BLOCK_KEY_COLS + ["dt_min", "excess", "excess_prev", "n_subject_records"]
        )
        block_idx = np.array(
            [key_to_idx[tuple(r[:5])] for r in arr2.rows()],
            dtype=np.int64,
        )
        dt1 = np.array(arr2["dt_min"].fill_null(0), dtype=np.float64) == 1.0
        excess = np.array(arr2["excess"].fill_null(0), dtype=np.float64)
        ex_prev = np.array(arr2["excess_prev"].fill_null(0), dtype=np.float64)
        max_occ_row = np.array(arr2["n_subject_records"], dtype=np.float64)

        n_blocks = len(key_tuples)
        room_samples = np.zeros((args.n_samples, n_blocks), dtype=np.float64)
        max_block_cap = (
            np.array(room_point["n_subject_records"], dtype=np.float64) * 240.0
        )

        for s in range(args.n_samples):
            phi_s = float(
                np.clip(
                    rng.normal(fit["phi_hat"], fit["phi_se"]),
                    cfg.model.phi_min,
                    min(cfg.model.phi_max, 0.998),
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
            occ = np.zeros_like(innovation)
            occ[dt1] = np.maximum(innovation[dt1], 0.0) / max(g_s, 1e-6)
            occ = np.minimum(np.clip(occ, 0.0, 10.0), max_occ_row)
            room_sum = np.bincount(block_idx, weights=occ, minlength=n_blocks)
            room_samples[s, :] = np.minimum(room_sum, max_block_cap)

        rp10 = np.quantile(room_samples, 0.10, axis=0)
        rp50 = np.quantile(room_samples, 0.50, axis=0)
        rp90 = np.quantile(room_samples, 0.90, axis=0)

        room_mc = room_point.with_columns(
            [
                pl.Series("room_person_minutes_p10", rp10),
                pl.Series("room_person_minutes_p50", rp50),
                pl.Series("room_person_minutes_p90", rp90),
            ]
        ).with_columns(
            (pl.col("room_person_minutes_p90") - pl.col("room_person_minutes_p10")).alias(
                "room_uncertainty_p90_p10"
            )
        )
        room_block_outputs.append(room_mc)

        subj_sensor = (
            subject_blocks.filter(pl.col("sensor_id") == sensor_id)
            .join(
                room_mc.select(BLOCK_KEY_COLS + ["n_present_subjects", "n_subject_records"]),
                on=BLOCK_KEY_COLS,
                how="inner",
            )
            .sort(["block_date", "block", "subject_id"])
        )

        prior_w = _get_prior_weights(subj_sensor, room_mc)
        (
            row_block_idx,
            active_rows_by_block,
            alpha_by_block,
            row_present,
            row_subject,
        ) = _prepare_allocation(subj_sensor, room_mc, prior_w, args.share_alpha)

        subj_samples = _allocate_subject_samples(
            rng, room_samples, row_block_idx, active_rows_by_block, alpha_by_block
        )
        sp10 = np.quantile(subj_samples, 0.10, axis=0)
        sp50 = np.quantile(subj_samples, 0.50, axis=0)
        sp90 = np.quantile(subj_samples, 0.90, axis=0)

        # Respect known subject-absence blocks from survey labels.
        absent_mask = row_present == 0
        sp10[absent_mask] = 0.0
        sp50[absent_mask] = 0.0
        sp90[absent_mask] = 0.0

        room_p50_by_row = rp50[row_block_idx]
        room_p10_by_row = rp10[row_block_idx]
        room_p90_by_row = rp90[row_block_idx]
        n_present_by_row = np.array(subj_sensor["n_present_subjects"], dtype=np.int64)

        sub_out = subj_sensor.with_columns(
            [
                pl.Series("room_person_minutes_best", room_p50_by_row),
                pl.Series("room_person_minutes_p10", room_p10_by_row),
                pl.Series("room_person_minutes_p90", room_p90_by_row),
                pl.Series("estimated_occupied_minutes_p10", sp10),
                pl.Series("estimated_occupied_minutes_best", sp50),
                pl.Series("estimated_occupied_minutes_p90", sp90),
                pl.Series("n_present_subjects_block", n_present_by_row),
            ]
        ).with_columns(
            [
                (pl.col("estimated_occupied_minutes_p90") - pl.col("estimated_occupied_minutes_p10")).alias(
                    "uncertainty_p90_p10"
                ),
                (pl.col("estimated_occupied_minutes_best") / 240.0).alias(
                    "estimated_occupied_fraction_best"
                ),
            ]
        ).with_columns(
            [
                pl.when(pl.col("uncertainty_p90_p10") <= 30)
                .then(pl.lit("high"))
                .when(pl.col("uncertainty_p90_p10") <= 60)
                .then(pl.lit("medium"))
                .otherwise(pl.lit("low"))
                .alias("stability_band"),
                pl.lit(fit["phi_hat"]).alias("phi_hat"),
                pl.lit(fit["phi_se"]).alias("phi_se"),
                pl.lit(fit["g_hat"]).alias("generation_hat"),
                pl.lit(fit["g_se"]).alias("generation_se"),
                pl.lit(fit["n_decay_points"]).alias("n_decay_points"),
                pl.lit(fit["n_solo_points"]).alias("n_solo_points"),
            ]
        )

        subject_block_outputs.append(sub_out)
        sensor_param_rows.append(
            {
                "sensor_id": int(sensor_id),
                "baseline": fit["baseline"],
                "phi_hat": fit["phi_hat"],
                "phi_se": fit["phi_se"],
                "generation_hat": fit["g_hat"],
                "generation_se": fit["g_se"],
                "n_decay_points": fit["n_decay_points"],
                "n_solo_points": fit["n_solo_points"],
                "max_occ": fit["max_occ"],
                "subject_weight_priors": json.dumps(prior_w, sort_keys=True),
            }
        )

    if not subject_block_outputs:
        raise RuntimeError("No subject-level outputs produced.")

    room_blocks_out = pl.concat(room_block_outputs, how="vertical").sort(
        ["sensor_id", "block_date", "block"]
    )
    block_best = pl.concat(subject_block_outputs, how="vertical").sort(
        ["sensor_id", "block_date", "block", "subject_id"]
    )
    sensor_params = pl.DataFrame(sensor_param_rows).sort("sensor_id")

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

    room_blocks_out.write_parquet(output_dir / "room_block_estimates.parquet")
    room_blocks_out.write_csv(output_dir / "room_block_estimates.csv")
    block_best.write_parquet(output_dir / "block_estimates_room_decomposition.parquet")
    block_best.write_csv(output_dir / "block_estimates_room_decomposition.csv")
    daily.write_parquet(output_dir / "daily_estimates_room_decomposition.parquet")
    daily.write_csv(output_dir / "daily_estimates_room_decomposition.csv")
    individual.write_csv(output_dir / "individual_summary_room_decomposition.csv")
    sensor_params.write_parquet(output_dir / "sensor_parameters_room_decomposition.parquet")
    sensor_params.write_csv(output_dir / "sensor_parameters_room_decomposition.csv")

    _save_uncertainty_hist(block_best, output_dir / "graphs" / "uncertainty_histogram.svg")
    _save_individual_bar(individual, output_dir / "graphs" / "individual_mean_daily.svg")

    label1 = block_best.filter(pl.col("present") == 1)
    label0 = block_best.filter(pl.col("present") == 0)
    summary = {
        "rows_subject_block": int(block_best.height),
        "rows_room_block": int(room_blocks_out.height),
        "rows_daily": int(daily.height),
        "subjects": int(block_best.select(pl.col("subject_id").n_unique()).item()),
        "sensors": int(block_best.select(pl.col("sensor_id").n_unique()).item()),
        "label1_mean_minutes_best": float(
            label1.select(pl.col("estimated_occupied_minutes_best").mean()).item()
        )
        if label1.height
        else None,
        "label1_median_minutes_best": float(
            label1.select(pl.col("estimated_occupied_minutes_best").median()).item()
        )
        if label1.height
        else None,
        "label0_mean_minutes_best": float(
            label0.select(pl.col("estimated_occupied_minutes_best").mean()).item()
        )
        if label0.height
        else None,
        "label1_mean_uncertainty_width": float(
            label1.select(pl.col("uncertainty_p90_p10").mean()).item()
        )
        if label1.height
        else None,
        "shared_block_rows_label1": int(
            label1.filter(pl.col("n_present_subjects_block") >= 2).height
        ),
        "stability_counts": {
            row[0]: int(row[1]) for row in block_best.group_by("stability_band").len().rows()
        },
    }
    (output_dir / "summary_room_decomposition.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
