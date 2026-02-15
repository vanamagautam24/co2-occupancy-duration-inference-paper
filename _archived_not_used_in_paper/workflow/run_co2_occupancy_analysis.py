#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class PathsConfig:
    co2_root: Path
    building_key_csv: Path
    occupancy_csv: Path
    subject_sensor_map_csv: Path | None
    output_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    block_hours: int
    smoothing_window_minutes: int
    baseline_quantile: float
    phi_default: float
    phi_min: float
    phi_max: float
    generation_quantile: float
    generation_default: float
    generation_single: float
    generation_shared: float
    generation_min: float
    generation_max: float
    min_data_minutes_per_block: int
    min_minutes_if_present: float


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    model: ModelConfig


def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p).resolve()


def load_config(config_path: Path) -> Config:
    root = config_path.resolve().parent.parent
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    paths_cfg = raw["paths"]
    model_cfg = raw["model"]

    paths = PathsConfig(
        co2_root=_resolve_path(root, paths_cfg["co2_root"]),
        building_key_csv=_resolve_path(root, paths_cfg["building_key_csv"]),
        occupancy_csv=_resolve_path(root, paths_cfg["occupancy_csv"]),
        subject_sensor_map_csv=(
            _resolve_path(root, str(paths_cfg.get("subject_sensor_map_csv", "")).strip())
            if str(paths_cfg.get("subject_sensor_map_csv", "")).strip()
            else None
        ),
        output_dir=_resolve_path(root, paths_cfg["output_dir"]),
    )
    model = ModelConfig(
        block_hours=int(model_cfg["block_hours"]),
        smoothing_window_minutes=int(model_cfg["smoothing_window_minutes"]),
        baseline_quantile=float(model_cfg["baseline_quantile"]),
        phi_default=float(model_cfg["phi_default"]),
        phi_min=float(model_cfg["phi_min"]),
        phi_max=float(model_cfg["phi_max"]),
        generation_quantile=float(model_cfg["generation_quantile"]),
        generation_default=float(model_cfg["generation_default"]),
        generation_single=float(model_cfg["generation_single"]),
        generation_shared=float(model_cfg["generation_shared"]),
        generation_min=float(model_cfg["generation_min"]),
        generation_max=float(model_cfg["generation_max"]),
        min_data_minutes_per_block=int(model_cfg["min_data_minutes_per_block"]),
        min_minutes_if_present=float(model_cfg["min_minutes_if_present"]),
    )
    return Config(paths=paths, model=model)


def discover_co2_files(co2_root: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(co2_root.glob("CO2_20*/*.csv")):
        if "unuseful" in str(p).lower():
            continue
        files.append(p)
    return files


def load_co2_minutes(co2_files: list[Path]) -> pl.DataFrame:
    lfs: list[pl.LazyFrame] = []
    for p in co2_files:
        sensor_id = int(p.stem)
        download_batch = p.parent.name
        lf = (
            pl.scan_csv(p, has_header=True, ignore_errors=True, infer_schema_length=2000)
            .select(
                [
                    pl.lit(sensor_id).cast(pl.Int64).alias("sensor_id"),
                    pl.lit(download_batch).alias("download_batch"),
                    pl.col("Datetime").cast(pl.Utf8).alias("datetime_raw"),
                    pl.col("CO2").cast(pl.Float64).alias("co2_ppm"),
                    pl.col("Temp").cast(pl.Float64).alias("temp_c"),
                    pl.col("RH").cast(pl.Float64).alias("rh_pct"),
                    pl.col("Dew").cast(pl.Float64).alias("dew_c"),
                ]
            )
        )
        lfs.append(lf)

    if not lfs:
        raise RuntimeError("No CO2 files found.")

    df = (
        pl.concat(lfs, how="vertical")
        .with_columns(
            pl.col("datetime_raw")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
            .alias("timestamp")
        )
        .filter(pl.col("timestamp").is_not_null() & pl.col("co2_ppm").is_not_null())
        .with_columns(
            [
                pl.col("timestamp").dt.truncate("1m").alias("timestamp_min"),
                pl.col("timestamp").dt.date().alias("date"),
            ]
        )
        .group_by(["sensor_id", "timestamp_min", "date"])
        .agg(
            [
                pl.mean("co2_ppm").alias("co2_ppm"),
                pl.mean("temp_c").alias("temp_c"),
                pl.mean("rh_pct").alias("rh_pct"),
                pl.mean("dew_c").alias("dew_c"),
                pl.len().alias("rows_per_minute"),
            ]
        )
        .sort(["sensor_id", "timestamp_min"])
        .collect()
    )
    return df


def load_building_key(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=2000)
    id_col = df.columns[0]
    out = (
        df.with_columns(
            [
                pl.col(id_col).cast(pl.Int64).alias("deployment_row_id"),
                pl.col("mon_id").cast(pl.Int64).alias("sensor_id"),
                pl.col("deploy_date")
                .cast(pl.Utf8)
                .str.to_date("%m/%d/%Y", strict=False)
                .alias("deploy_date"),
                pl.col("return_date_redcap")
                .cast(pl.Utf8)
                .str.to_date("%m/%d/%Y", strict=False)
                .alias("return_date"),
            ]
        )
        .select(
            [
                "deployment_row_id",
                "sensor_id",
                "group_id_deploy",
                "deploy_date",
                "return_date",
                "Address",
                "building_code",
                "hallname",
                "vent_cat",
            ]
        )
        .filter(
            pl.col("sensor_id").is_not_null()
            & pl.col("deploy_date").is_not_null()
        )
    )
    return out


def load_occupancy(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=2000)
    cols = {c.lower(): c for c in df.columns}
    required = ["subject_id", "date", "block", "value"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise RuntimeError(f"Occupancy file missing required columns: {missing}")

    out = (
        df.rename(
            {
                cols["subject_id"]: "subject_id",
                cols["date"]: "date",
                cols["block"]: "block",
                cols["value"]: "present",
            }
        )
        .with_columns(
            [
                pl.col("subject_id").cast(pl.Int64),
                pl.col("date").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False),
                pl.col("block").cast(pl.Int64),
                pl.col("present").cast(pl.Float64).round(0).cast(pl.Int8),
            ]
        )
        .filter(
            pl.col("subject_id").is_not_null()
            & pl.col("date").is_not_null()
            & pl.col("block").is_not_null()
            & pl.col("present").is_not_null()
        )
        .filter(pl.col("present").is_in([0, 1]))
    )
    if "room_type" in cols:
        out = out.with_columns(
            pl.col(cols["room_type"]).cast(pl.Utf8).str.to_lowercase().alias("room_type")
        )
    else:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("room_type"))
    return out


def _parse_date_flex(col: pl.Expr) -> pl.Expr:
    return pl.coalesce(
        [
            col.cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False),
            col.cast(pl.Utf8).str.to_date("%m/%d/%Y", strict=False),
            col.cast(pl.Utf8).str.to_date("%m/%d/%y", strict=False),
        ]
    )


def load_subject_sensor_map(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=2000)
    cols = {c.lower(): c for c in df.columns}

    subject_col = cols.get("subject_id")
    sensor_col = cols.get("sensor_id") or cols.get("mon_id")
    if subject_col is None or sensor_col is None:
        raise RuntimeError(
            "Subject-sensor map must contain at least `subject_id` and `sensor_id` columns."
        )

    start_col = None
    for c in ["map_start_date", "start_date", "deploy_date", "from_date"]:
        if c in cols:
            start_col = cols[c]
            break
    end_col = None
    for c in ["map_end_date", "end_date", "return_date", "to_date"]:
        if c in cols:
            end_col = cols[c]
            break
    room_col = cols.get("room_type")

    rename_map = {
        subject_col: "subject_id",
        sensor_col: "sensor_id",
    }
    if start_col is not None:
        rename_map[start_col] = "map_start_raw"
    if end_col is not None:
        rename_map[end_col] = "map_end_raw"
    if room_col is not None:
        rename_map[room_col] = "room_type_map"

    out = (
        df.rename(rename_map)
        .with_columns(
            [
                pl.col("subject_id").cast(pl.Int64),
                pl.col("sensor_id").cast(pl.Int64),
            ]
        )
        .filter(pl.col("subject_id").is_not_null() & pl.col("sensor_id").is_not_null())
    )

    if "map_start_raw" in out.columns:
        out = out.with_columns(_parse_date_flex(pl.col("map_start_raw")).alias("map_start_date"))
    else:
        out = out.with_columns(pl.lit(None).cast(pl.Date).alias("map_start_date"))

    if "map_end_raw" in out.columns:
        out = out.with_columns(_parse_date_flex(pl.col("map_end_raw")).alias("map_end_date"))
    else:
        out = out.with_columns(pl.lit(None).cast(pl.Date).alias("map_end_date"))

    if "room_type_map" in out.columns:
        out = out.with_columns(pl.col("room_type_map").cast(pl.Utf8).str.to_lowercase())
    else:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("room_type_map"))

    out = out.select(
        [
            "subject_id",
            "sensor_id",
            "map_start_date",
            "map_end_date",
            "room_type_map",
        ]
    )
    out = out.unique(subset=["subject_id", "sensor_id", "map_start_date", "map_end_date"])
    return out


def map_occupancy_to_sensor(
    occupancy: pl.DataFrame,
    subject_sensor_map: pl.DataFrame,
    building_key: pl.DataFrame,
    co2_minutes: pl.DataFrame,
) -> pl.DataFrame:
    co2_coverage = co2_minutes.group_by("sensor_id").agg(
        [
            pl.col("date").min().alias("co2_start"),
            pl.col("date").max().alias("co2_end"),
        ]
    )
    mapped = (
        occupancy.join(subject_sensor_map, on="subject_id", how="inner")
        .with_columns(pl.coalesce([pl.col("room_type"), pl.col("room_type_map")]).alias("room_type"))
        .drop("room_type_map")
        .filter(
            (pl.col("map_start_date").is_null() | (pl.col("date") >= pl.col("map_start_date")))
            & (pl.col("map_end_date").is_null() | (pl.col("date") <= pl.col("map_end_date")))
        )
        .join(building_key, on="sensor_id", how="left")
        .join(co2_coverage, on="sensor_id", how="inner")
        .filter(
            (pl.col("deploy_date").is_null() | (pl.col("date") >= pl.col("deploy_date")))
            & (pl.col("return_date").is_null() | (pl.col("date") <= pl.col("return_date")))
            & (pl.col("date") >= pl.col("co2_start"))
            & (pl.col("date") <= pl.col("co2_end"))
        )
        .select(
            [
                "subject_id",
                "sensor_id",
                "date",
                "block",
                "present",
                "room_type",
                "group_id_deploy",
                "deploy_date",
                "return_date",
                "Address",
                "building_code",
                "hallname",
                "vent_cat",
                "co2_start",
                "co2_end",
            ]
        )
        .sort(["sensor_id", "date", "block"])
    )
    mapped = mapped.unique(subset=["subject_id", "sensor_id", "date", "block"], keep="first")
    return mapped


def expand_blocks_to_minutes(mapped_blocks: pl.DataFrame, block_hours: int) -> pl.DataFrame:
    out = (
        mapped_blocks.with_columns(
            [
                (
                    pl.col("date").cast(pl.Datetime)
                    + pl.duration(hours=pl.col("block").cast(pl.Int64))
                ).alias("block_start_ts"),
            ]
        )
        .with_columns(
            [
                (pl.col("block_start_ts") + pl.duration(hours=block_hours)).alias("block_end_ts"),
                pl.datetime_ranges(
                    pl.col("block_start_ts"),
                    pl.col("block_start_ts") + pl.duration(hours=block_hours),
                    interval="1m",
                    closed="left",
                ).alias("timestamp_min"),
            ]
        )
        .explode("timestamp_min")
        .select(
            [
                "subject_id",
                "sensor_id",
                pl.col("date").alias("block_date"),
                "block",
                "present",
                "room_type",
                "group_id_deploy",
                "Address",
                "building_code",
                "hallname",
                "vent_cat",
                "block_start_ts",
                "block_end_ts",
                "timestamp_min",
            ]
        )
    )
    return out


def _default_generation_expr(model: ModelConfig) -> pl.Expr:
    return (
        pl.when(pl.col("room_type_mode").is_not_null() & (pl.col("room_type_mode") == "single"))
        .then(pl.lit(model.generation_single))
        .when(pl.col("room_type_mode").is_not_null() & (pl.col("room_type_mode") == "shared"))
        .then(pl.lit(model.generation_shared))
        .otherwise(pl.lit(model.generation_default))
    )


def infer_occupancy_from_co2(
    co2_minutes: pl.DataFrame, occupancy_minute: pl.DataFrame, model: ModelConfig
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    joined = (
        co2_minutes.join(occupancy_minute, on=["sensor_id", "timestamp_min"], how="inner")
        .sort(["sensor_id", "timestamp_min"])
        .with_columns(
            [
                pl.col("co2_ppm")
                .rolling_median(window_size=model.smoothing_window_minutes)
                .over("sensor_id")
                .alias("co2_smooth"),
            ]
        )
        .with_columns(pl.col("co2_smooth").fill_null(pl.col("co2_ppm")).alias("co2_smooth"))
    )

    baseline_by_sensor = joined.filter(pl.col("present") == 0).group_by("sensor_id").agg(
        pl.col("co2_smooth").quantile(model.baseline_quantile).alias("co2_baseline")
    )
    if baseline_by_sensor.height == 0:
        global_baseline = float(joined.select(pl.col("co2_smooth").quantile(0.05)).item())
    else:
        global_baseline = float(
            baseline_by_sensor.select(pl.col("co2_baseline").quantile(0.5)).item()
        )

    joined = (
        joined.join(baseline_by_sensor, on="sensor_id", how="left")
        .with_columns(pl.col("co2_baseline").fill_null(global_baseline).alias("co2_baseline"))
        .with_columns((pl.col("co2_smooth") - pl.col("co2_baseline")).clip(lower_bound=0).alias("excess_ppm"))
        .with_columns(
            [
                pl.col("timestamp_min").shift(1).over("sensor_id").alias("prev_ts"),
                pl.col("excess_ppm").shift(1).over("sensor_id").alias("excess_prev"),
            ]
        )
        .with_columns(
            ((pl.col("timestamp_min") - pl.col("prev_ts")).dt.total_minutes()).alias("dt_min")
        )
    )

    decay_pairs = joined.filter(
        (pl.col("present") == 0) & (pl.col("dt_min") == 1) & pl.col("excess_prev").is_not_null()
    )
    phi_stats = (
        decay_pairs.group_by("sensor_id")
        .agg(
            [
                (pl.col("excess_ppm") * pl.col("excess_prev")).sum().alias("num"),
                (pl.col("excess_prev") * pl.col("excess_prev")).sum().alias("den"),
                pl.len().alias("n_decay_points"),
            ]
        )
        .with_columns(
            pl.when(pl.col("den") > 1e-6)
            .then(pl.col("num") / pl.col("den"))
            .otherwise(pl.lit(model.phi_default))
            .alias("phi_raw")
        )
        .with_columns(pl.col("phi_raw").clip(model.phi_min, model.phi_max).alias("phi"))
        .select(["sensor_id", "phi", "n_decay_points"])
    )

    room_mode = joined.group_by("sensor_id").agg(
        pl.col("room_type").drop_nulls().mode().first().cast(pl.Utf8).alias("room_type_mode")
    )

    joined = (
        joined.join(phi_stats, on="sensor_id", how="left")
        .with_columns(pl.col("phi").fill_null(model.phi_default).alias("phi"))
        .join(room_mode, on="sensor_id", how="left")
        .with_columns(
            [
                (pl.col("excess_ppm") - pl.col("phi") * pl.col("excess_prev")).alias("innovation_ppm"),
            ]
        )
        .with_columns(pl.col("innovation_ppm").clip(lower_bound=0).alias("innovation_pos_ppm"))
    )

    gen_stats = (
        joined.filter((pl.col("present") == 1) & (pl.col("dt_min") == 1))
        .group_by("sensor_id")
        .agg(
            [
                pl.col("innovation_pos_ppm")
                .quantile(model.generation_quantile)
                .alias("generation_raw"),
                pl.len().alias("n_occ_points"),
            ]
        )
    )

    params = (
        joined.select(["sensor_id", "co2_baseline", "room_type_mode"])
        .unique(subset=["sensor_id"])
        .join(phi_stats, on="sensor_id", how="left")
        .join(gen_stats, on="sensor_id", how="left")
        .with_columns(
            [
                pl.col("phi").fill_null(model.phi_default).alias("phi"),
                pl.when(pl.col("generation_raw").is_not_null() & (pl.col("generation_raw") > 0))
                .then(pl.col("generation_raw"))
                .otherwise(_default_generation_expr(model))
                .alias("generation_per_min"),
            ]
        )
        .with_columns(
            pl.col("generation_per_min")
            .clip(model.generation_min, model.generation_max)
            .alias("generation_per_min")
        )
        .with_columns(pl.col("n_decay_points").fill_null(0), pl.col("n_occ_points").fill_null(0))
    )

    minute_est = (
        joined.join(params.select(["sensor_id", "generation_per_min"]), on="sensor_id", how="left")
        .with_columns(
            [
                pl.when(pl.col("present") == 0)
                .then(pl.lit(0.0))
                .when(pl.col("dt_min") == 1)
                .then((pl.col("innovation_pos_ppm") / pl.col("generation_per_min")).clip(0.0, 1.0))
                .otherwise(pl.lit(0.0))
                .alias("occ_intensity"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("occ_intensity") >= 0.5)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("occ_binary_est"),
            ]
        )
    )

    block_est = (
        minute_est.group_by(
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
                pl.col("occ_intensity").sum().alias("estimated_minutes_raw"),
                pl.col("co2_smooth").mean().alias("mean_co2_ppm"),
                pl.col("co2_smooth").max().alias("peak_co2_ppm"),
                pl.col("innovation_pos_ppm").mean().alias("mean_innovation_ppm"),
            ]
        )
        .join(params.select(["sensor_id", "phi", "generation_per_min", "n_decay_points"]), on="sensor_id", how="left")
        .with_columns(
            [
                pl.when(pl.col("present") == 0)
                .then(pl.lit(0.0))
                .when(
                    (pl.col("present") == 1)
                    & (pl.col("data_minutes") >= model.min_data_minutes_per_block)
                    & (pl.col("estimated_minutes_raw") < model.min_minutes_if_present)
                )
                .then(pl.lit(model.min_minutes_if_present))
                .otherwise(pl.col("estimated_minutes_raw"))
                .clip(0.0, float(model.block_hours * 60))
                .alias("estimated_occupied_minutes"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("estimated_occupied_minutes") / float(model.block_hours * 60)
                ).alias("estimated_occupied_fraction"),
                (pl.lit(model.block_hours * 60) - pl.col("data_minutes")).alias("missing_minutes"),
                (
                    (pl.col("data_minutes") / float(model.block_hours * 60)).clip(0.0, 1.0)
                    * pl.when(pl.col("n_decay_points") >= 100)
                    .then(pl.lit(1.0))
                    .when(pl.col("n_decay_points") >= 30)
                    .then(pl.lit(0.7))
                    .otherwise(pl.lit(0.4))
                ).alias("quality_score"),
            ]
        )
        .with_columns(
            pl.when(pl.col("quality_score") >= 0.8)
            .then(pl.lit("high"))
            .when(pl.col("quality_score") >= 0.55)
            .then(pl.lit("medium"))
            .otherwise(pl.lit("low"))
            .alias("quality_band")
        )
        .sort(["sensor_id", "block_date", "block"])
    )

    return params.sort("sensor_id"), minute_est.sort(["sensor_id", "timestamp_min"]), block_est


def _svg_header(width: int, height: int) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>"
    )


def _svg_footer() -> str:
    return "</svg>"


def save_histogram_svg(block_est: pl.DataFrame, out_path: Path) -> None:
    present1 = (
        block_est.filter(pl.col("present") == 1)
        .select("estimated_occupied_minutes")
        .to_series()
        .to_numpy()
    )
    present0 = (
        block_est.filter(pl.col("present") == 0)
        .select("estimated_occupied_minutes")
        .to_series()
        .to_numpy()
    )
    bins = np.arange(0, 241, 15)
    h1, _ = np.histogram(present1, bins=bins)
    h0, _ = np.histogram(present0, bins=bins)
    ymax = int(max(h1.max() if len(h1) else 1, h0.max() if len(h0) else 1))
    ymax = max(1, ymax)

    w, h = 1400, 820
    left, right, top, bottom = 90, 40, 70, 90
    plot_w, plot_h = w - left - right, h - top - bottom
    bw = plot_w / len(h1)

    parts = [_svg_header(w, h)]
    parts.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    parts.append(
        "<text x='90' y='36' font-size='28' font-family='sans-serif'>Estimated Occupied Minutes Per 4h Block</text>"
    )
    parts.append(f"<line x1='{left}' y1='{top+plot_h}' x2='{left+plot_w}' y2='{top+plot_h}' stroke='black'/>")
    parts.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}' stroke='black'/>")

    for i in range(0, 17, 2):
        x = left + i * bw
        label = int(bins[i])
        parts.append(f"<text x='{x:.1f}' y='{top+plot_h+24}' font-size='12' font-family='sans-serif'>{label}</text>")
    for i in range(6):
        yv = ymax * i / 5.0
        y = top + plot_h - (yv / ymax) * plot_h
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left+plot_w}' y2='{y:.1f}' stroke='#e8e8e8'/>")
        parts.append(f"<text x='40' y='{y+4:.1f}' font-size='12' font-family='sans-serif'>{int(round(yv))}</text>")

    for i, (c1, c0) in enumerate(zip(h1, h0)):
        x = left + i * bw
        h1_px = (c1 / ymax) * plot_h if ymax > 0 else 0
        h0_px = (c0 / ymax) * plot_h if ymax > 0 else 0
        parts.append(
            f"<rect x='{x+1:.1f}' y='{top+plot_h-h1_px:.1f}' width='{bw/2-2:.1f}' height='{h1_px:.1f}' fill='#2A9D8F'/>"
        )
        parts.append(
            f"<rect x='{x+bw/2+1:.1f}' y='{top+plot_h-h0_px:.1f}' width='{bw/2-2:.1f}' height='{h0_px:.1f}' fill='#E76F51'/>"
        )

    parts.append(
        "<rect x='980' y='72' width='16' height='16' fill='#2A9D8F'/><text x='1004' y='85' font-size='14' font-family='sans-serif'>label=1 blocks</text>"
    )
    parts.append(
        "<rect x='1168' y='72' width='16' height='16' fill='#E76F51'/><text x='1192' y='85' font-size='14' font-family='sans-serif'>label=0 blocks</text>"
    )
    parts.append("<text x='610' y='790' font-size='14' font-family='sans-serif'>Estimated occupied minutes</text>")
    parts.append("<text x='14' y='420' transform='rotate(-90,14,420)' font-size='14' font-family='sans-serif'>Block count</text>")
    parts.append(_svg_footer())
    out_path.write_text("".join(parts), encoding="utf-8")


def save_block_hour_svg(block_est: pl.DataFrame, out_path: Path) -> None:
    by_block = (
        block_est.filter(pl.col("present") == 1)
        .group_by("block")
        .agg(pl.col("estimated_occupied_minutes").mean().alias("avg_minutes"))
        .sort("block")
    )
    blocks = by_block.select("block").to_series().to_list()
    vals = by_block.select("avg_minutes").to_series().to_list()
    if not blocks:
        return
    vmax = max(vals) if vals else 1.0
    vmax = max(vmax, 1.0)

    w, h = 1200, 700
    left, right, top, bottom = 90, 40, 70, 90
    plot_w, plot_h = w - left - right, h - top - bottom
    bw = plot_w / len(blocks)

    parts = [_svg_header(w, h)]
    parts.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    parts.append(
        "<text x='90' y='36' font-size='28' font-family='sans-serif'>Average Estimated Occupied Minutes by Block Start Hour (label=1)</text>"
    )
    parts.append(f"<line x1='{left}' y1='{top+plot_h}' x2='{left+plot_w}' y2='{top+plot_h}' stroke='black'/>")
    parts.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}' stroke='black'/>")
    for i in range(6):
        yv = vmax * i / 5.0
        y = top + plot_h - (yv / vmax) * plot_h
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left+plot_w}' y2='{y:.1f}' stroke='#e8e8e8'/>")
        parts.append(f"<text x='42' y='{y+4:.1f}' font-size='12' font-family='sans-serif'>{int(round(yv))}</text>")

    for i, (b, v) in enumerate(zip(blocks, vals)):
        x = left + i * bw
        bh = (v / vmax) * plot_h
        parts.append(
            f"<rect x='{x+16:.1f}' y='{top+plot_h-bh:.1f}' width='{bw-32:.1f}' height='{bh:.1f}' fill='#457B9D'/>"
        )
        parts.append(
            f"<text x='{x+bw/2-14:.1f}' y='{top+plot_h+24}' font-size='13' font-family='sans-serif'>{int(b)}:00</text>"
        )
        parts.append(
            f"<text x='{x+bw/2-8:.1f}' y='{top+plot_h-bh-8:.1f}' font-size='12' font-family='sans-serif'>{int(round(v))}</text>"
        )

    parts.append("<text x='530' y='670' font-size='14' font-family='sans-serif'>Block start hour</text>")
    parts.append("<text x='14' y='360' transform='rotate(-90,14,360)' font-size='14' font-family='sans-serif'>Average occupied minutes</text>")
    parts.append(_svg_footer())
    out_path.write_text("".join(parts), encoding="utf-8")


def save_example_timeseries_svg(minute_est: pl.DataFrame, out_path: Path) -> None:
    key = (
        minute_est.group_by(["sensor_id", "date"])
        .agg(
            [
                pl.len().alias("n"),
                pl.col("present").n_unique().alias("n_labels"),
                pl.col("co2_smooth").max().alias("peak"),
            ]
        )
        .filter((pl.col("n") >= 180) & (pl.col("n_labels") >= 2))
        .sort(["peak", "n"], descending=True)
        .head(1)
    )
    if key.height == 0:
        return
    sensor_id, day = key.select(["sensor_id", "date"]).row(0)
    ex = (
        minute_est.filter((pl.col("sensor_id") == sensor_id) & (pl.col("date") == day))
        .sort("timestamp_min")
        .select(["timestamp_min", "co2_smooth", "occ_intensity", "present"])
    )
    if ex.height < 2:
        return
    y_co2 = ex["co2_smooth"].to_numpy()
    y_occ = ex["occ_intensity"].to_numpy()
    x = np.arange(len(y_co2))
    co2_min = float(np.nanmin(y_co2))
    co2_max = float(np.nanmax(y_co2))
    if not math.isfinite(co2_min) or not math.isfinite(co2_max) or co2_max <= co2_min:
        return

    w, h = 1500, 760
    left, right, top, bottom = 100, 60, 80, 90
    plot_w, plot_h = w - left - right, h - top - bottom

    def x_px(i: int) -> float:
        return left + (i / max(1, len(x) - 1)) * plot_w

    def y_px_co2(v: float) -> float:
        return top + plot_h - ((v - co2_min) / (co2_max - co2_min)) * plot_h

    def y_px_occ(v: float) -> float:
        return top + plot_h - v * plot_h

    co2_points = " ".join(f"{x_px(i):.2f},{y_px_co2(v):.2f}" for i, v in enumerate(y_co2))
    occ_points = " ".join(f"{x_px(i):.2f},{y_px_occ(v):.2f}" for i, v in enumerate(y_occ))

    parts = [_svg_header(w, h)]
    parts.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    parts.append(
        f"<text x='100' y='38' font-size='28' font-family='sans-serif'>Example Sensor-Day: sensor {sensor_id}, {day}</text>"
    )
    parts.append(
        "<text x='100' y='62' font-size='15' font-family='sans-serif'>Blue: CO2 (ppm), Orange: estimated occupancy intensity (0-1)</text>"
    )
    parts.append(f"<line x1='{left}' y1='{top+plot_h}' x2='{left+plot_w}' y2='{top+plot_h}' stroke='black'/>")
    parts.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}' stroke='black'/>")
    parts.append(f"<line x1='{left+plot_w}' y1='{top}' x2='{left+plot_w}' y2='{top+plot_h}' stroke='black'/>")

    for i in range(6):
        frac = i / 5.0
        y = top + plot_h - frac * plot_h
        co2_val = co2_min + frac * (co2_max - co2_min)
        parts.append(f"<line x1='{left}' y1='{y:.2f}' x2='{left+plot_w}' y2='{y:.2f}' stroke='#efefef'/>")
        parts.append(f"<text x='35' y='{y+4:.1f}' font-size='12' font-family='sans-serif'>{int(round(co2_val))}</text>")
        parts.append(f"<text x='{left+plot_w+10}' y='{y+4:.1f}' font-size='12' font-family='sans-serif'>{frac:.1f}</text>")

    parts.append(f"<polyline fill='none' stroke='#1D3557' stroke-width='2' points='{co2_points}'/>")
    parts.append(f"<polyline fill='none' stroke='#F4A261' stroke-width='2' points='{occ_points}'/>")
    parts.append(
        f"<text x='{left+plot_w/2-70:.0f}' y='{h-24}' font-size='14' font-family='sans-serif'>Minutes within day</text>"
    )
    parts.append("<text x='14' y='430' transform='rotate(-90,14,430)' font-size='14' font-family='sans-serif'>CO2 ppm</text>")
    parts.append(
        f"<text x='{w-20}' y='430' transform='rotate(-90,{w-20},430)' font-size='14' font-family='sans-serif'>Occ intensity</text>"
    )
    parts.append(_svg_footer())
    out_path.write_text("".join(parts), encoding="utf-8")


def save_summary_json(
    out_path: Path,
    co2_minutes: pl.DataFrame,
    mapped_blocks: pl.DataFrame,
    minute_est: pl.DataFrame,
    block_est: pl.DataFrame,
    params: pl.DataFrame,
) -> None:
    label1 = block_est.filter(pl.col("present") == 1)
    label0 = block_est.filter(pl.col("present") == 0)
    summary: dict[str, Any] = {
        "co2_rows_minute_level": int(co2_minutes.height),
        "mapped_block_rows": int(mapped_blocks.height),
        "minute_rows_after_join": int(minute_est.height),
        "block_estimate_rows": int(block_est.height),
        "n_subjects_mapped": int(mapped_blocks.select(pl.col("subject_id").n_unique()).item()),
        "n_sensors_mapped": int(mapped_blocks.select(pl.col("sensor_id").n_unique()).item()),
        "date_min": str(mapped_blocks.select(pl.col("date").min()).item()),
        "date_max": str(mapped_blocks.select(pl.col("date").max()).item()),
        "phi_median": float(params.select(pl.col("phi").median()).item()),
        "generation_median": float(params.select(pl.col("generation_per_min").median()).item()),
        "label1_avg_est_minutes": float(label1.select(pl.col("estimated_occupied_minutes").mean()).item())
        if label1.height
        else None,
        "label0_avg_est_minutes": float(label0.select(pl.col("estimated_occupied_minutes").mean()).item())
        if label0.height
        else None,
        "quality_band_counts": {
            row[0]: int(row[1])
            for row in block_est.group_by("quality_band").len().rows()
        },
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer occupancy minutes from CO2 + block labels.")
    parser.add_argument(
        "--config",
        default="workflow/config.default.toml",
        help="Path to TOML config file.",
    )
    parser.add_argument(
        "--occupancy-file",
        default=None,
        help="Optional override path to occupancy CSV.",
    )
    parser.add_argument(
        "--subject-sensor-map",
        default=None,
        help="Optional override path to subject-sensor crosswalk CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override path for outputs.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    paths = cfg.paths
    if args.occupancy_file:
        paths = PathsConfig(
            co2_root=paths.co2_root,
            building_key_csv=paths.building_key_csv,
            occupancy_csv=_resolve_path(Path.cwd(), args.occupancy_file),
            subject_sensor_map_csv=paths.subject_sensor_map_csv,
            output_dir=paths.output_dir,
        )
    if args.subject_sensor_map:
        paths = PathsConfig(
            co2_root=paths.co2_root,
            building_key_csv=paths.building_key_csv,
            occupancy_csv=paths.occupancy_csv,
            subject_sensor_map_csv=_resolve_path(Path.cwd(), args.subject_sensor_map),
            output_dir=paths.output_dir,
        )
    if args.output_dir:
        paths = PathsConfig(
            co2_root=paths.co2_root,
            building_key_csv=paths.building_key_csv,
            occupancy_csv=paths.occupancy_csv,
            subject_sensor_map_csv=paths.subject_sensor_map_csv,
            output_dir=_resolve_path(Path.cwd(), args.output_dir),
        )

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = paths.output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    co2_files = discover_co2_files(paths.co2_root)
    co2_minutes = load_co2_minutes(co2_files)
    occupancy = load_occupancy(paths.occupancy_csv)
    if paths.subject_sensor_map_csv is None:
        raise RuntimeError(
            "No subject-sensor map provided. Set `paths.subject_sensor_map_csv` in config or pass "
            "`--subject-sensor-map` with columns subject_id,sensor_id (and optional date bounds)."
        )
    subject_sensor_map = load_subject_sensor_map(paths.subject_sensor_map_csv)
    building_key = load_building_key(paths.building_key_csv)
    mapped_blocks = map_occupancy_to_sensor(
        occupancy,
        subject_sensor_map,
        building_key,
        co2_minutes,
    )
    if mapped_blocks.height == 0:
        raise RuntimeError(
            "No mapped occupancy rows after applying subject-sensor mapping + date filters. "
            "Check mapping file and date bounds."
        )

    occ_minute = expand_blocks_to_minutes(mapped_blocks, cfg.model.block_hours)
    params, minute_est, block_est = infer_occupancy_from_co2(co2_minutes, occ_minute, cfg.model)

    mapped_blocks.write_parquet(paths.output_dir / "mapped_blocks.parquet")
    mapped_blocks.write_csv(paths.output_dir / "mapped_blocks.csv")
    params.write_parquet(paths.output_dir / "sensor_parameters.parquet")
    params.write_csv(paths.output_dir / "sensor_parameters.csv")
    minute_est.write_parquet(paths.output_dir / "minute_estimates.parquet")
    block_est.write_parquet(paths.output_dir / "block_estimates.parquet")
    block_est.write_csv(paths.output_dir / "block_estimates.csv")

    save_histogram_svg(block_est, graphs_dir / "hist_estimated_minutes.svg")
    save_block_hour_svg(block_est, graphs_dir / "avg_minutes_by_block.svg")
    save_example_timeseries_svg(minute_est, graphs_dir / "example_sensor_day.svg")

    save_summary_json(
        paths.output_dir / "summary.json",
        co2_minutes,
        mapped_blocks,
        minute_est,
        block_est,
        params,
    )

    top_blocks = block_est.sort("estimated_occupied_minutes", descending=True).head(20)
    top_blocks.write_csv(paths.output_dir / "top20_blocks_by_estimated_minutes.csv")


if __name__ == "__main__":
    run()
