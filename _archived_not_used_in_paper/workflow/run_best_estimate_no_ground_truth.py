#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import polars as pl


IDENTITY_COLS = [
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

META_COLS = [
    "data_minutes",
    "phi",
    "generation_per_min",
    "n_decay_points",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run robust no-ground-truth occupancy duration ensemble from CO2."
    )
    p.add_argument("--config", default="workflow/config.default.toml")
    p.add_argument("--subject-sensor-map", required=True)
    p.add_argument("--occupancy-file", default=None)
    p.add_argument("--output-dir", default="outputs_spring2025_best_estimate")
    return p.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def write_toml(data: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    for section in ["paths", "model"]:
        lines.append(f"[{section}]")
        sec = data[section]
        for k, v in sec.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            else:
                lines.append(f"{k} = {v}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_variant(
    config_path: Path,
    subject_sensor_map: Path,
    output_dir: Path,
    occupancy_file: Path | None,
) -> None:
    cmd = [
        "python3",
        "workflow/run_co2_occupancy_analysis.py",
        "--config",
        str(config_path),
        "--subject-sensor-map",
        str(subject_sensor_map),
        "--output-dir",
        str(output_dir),
    ]
    if occupancy_file is not None:
        cmd.extend(["--occupancy-file", str(occupancy_file)])
    subprocess.run(cmd, check=True)


def save_uncertainty_hist_svg(df: pl.DataFrame, out_path: Path) -> None:
    vals = df.select("uncertainty_p90_p10").to_series().to_list()
    if not vals:
        return
    bins = [0, 15, 30, 45, 60, 90, 120, 180, 240]
    counts = [0] * (len(bins) - 1)
    for v in vals:
        if v is None:
            continue
        x = float(v)
        for i in range(len(bins) - 1):
            if bins[i] <= x < bins[i + 1]:
                counts[i] += 1
                break
            if x >= bins[-1]:
                counts[-1] += 1
                break

    w, h = 1180, 700
    left, right, top, bottom = 80, 40, 70, 90
    plot_w = w - left - right
    plot_h = h - top - bottom
    ymax = max(1, max(counts))
    bw = plot_w / len(counts)

    parts: list[str] = []
    parts.append(
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
    )
    parts.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    parts.append(
        "<text x='80' y='35' font-size='28' font-family='sans-serif'>Uncertainty Width per Block (p90 - p10 minutes)</text>"
    )
    parts.append(
        f"<line x1='{left}' y1='{top+plot_h}' x2='{left+plot_w}' y2='{top+plot_h}' stroke='black'/>"
    )
    parts.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}' stroke='black'/>")

    for i, c in enumerate(counts):
        x = left + i * bw
        bh = (c / ymax) * plot_h
        parts.append(
            f"<rect x='{x+4:.1f}' y='{top+plot_h-bh:.1f}' width='{bw-8:.1f}' height='{bh:.1f}' fill='#2A9D8F'/>"
        )
        parts.append(
            f"<text x='{x+bw/2-8:.1f}' y='{top+plot_h+24}' font-size='12' font-family='sans-serif'>{bins[i]}-{bins[i+1]}</text>"
        )
    parts.append(
        "<text x='500' y='680' font-size='14' font-family='sans-serif'>Uncertainty bucket (minutes)</text>"
    )
    parts.append("</svg>")
    out_path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_cfg_path = Path(args.config).resolve()
    subject_sensor_map = Path(args.subject_sensor_map).resolve()
    occupancy_file = Path(args.occupancy_file).resolve() if args.occupancy_file else None
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_toml(base_cfg_path)

    variants: list[tuple[str, dict[str, Any]]] = [
        (
            "v_base",
            {},
        ),
        (
            "v_cons_low",
            {
                "smoothing_window_minutes": 3,
                "baseline_quantile": 0.05,
                "generation_quantile": 0.7,
                "min_minutes_if_present": 0.0,
            },
        ),
        (
            "v_cons_high",
            {
                "smoothing_window_minutes": 9,
                "baseline_quantile": 0.2,
                "generation_quantile": 0.9,
                "min_minutes_if_present": 10.0,
            },
        ),
        (
            "v_mid_1",
            {
                "smoothing_window_minutes": 5,
                "baseline_quantile": 0.08,
                "generation_quantile": 0.75,
                "min_minutes_if_present": 3.0,
            },
        ),
        (
            "v_mid_2",
            {
                "smoothing_window_minutes": 7,
                "baseline_quantile": 0.12,
                "generation_quantile": 0.85,
                "min_minutes_if_present": 7.0,
            },
        ),
    ]

    project_root = base_cfg_path.parent.parent
    cfg_dir = project_root / "tmp_ensemble_cfg"
    run_dir = out_root / "_runs"
    cfg_dir.mkdir(exist_ok=True)
    run_dir.mkdir(exist_ok=True)

    run_outputs: list[tuple[str, Path]] = []
    for name, overrides in variants:
        local = {
            "paths": dict(cfg["paths"]),
            "model": dict(cfg["model"]),
        }
        for k, v in overrides.items():
            local["model"][k] = v
        local["paths"]["output_dir"] = str((run_dir / name).as_posix())
        # this workflow requires explicit subject-sensor mapping from CLI
        local["paths"]["subject_sensor_map_csv"] = ""
        this_cfg = cfg_dir / f"{name}.toml"
        write_toml(local, this_cfg)
        run_variant(this_cfg, subject_sensor_map, run_dir / name, occupancy_file)
        run_outputs.append((name, run_dir / name / "block_estimates.parquet"))

    merged: pl.DataFrame | None = None
    for i, (name, path) in enumerate(run_outputs):
        df = pl.read_parquet(path)
        if i == 0:
            df = df.select(
                IDENTITY_COLS
                + META_COLS
                + [
                    pl.col("estimated_occupied_minutes").alias(f"minutes_{name}"),
                    pl.col("estimated_occupied_fraction").alias(f"fraction_{name}"),
                ]
            )
        else:
            df = df.select(
                IDENTITY_COLS
                + [
                    pl.col("estimated_occupied_minutes").alias(f"minutes_{name}"),
                    pl.col("estimated_occupied_fraction").alias(f"fraction_{name}"),
                ]
            )
        if i == 0:
            merged = df
        else:
            assert merged is not None
            merged = merged.join(df, on=IDENTITY_COLS, how="inner")

    if merged is None or merged.height == 0:
        raise RuntimeError("No merged ensemble rows were produced.")

    minute_cols = [f"minutes_{name}" for name, _ in run_outputs]
    frac_cols = [f"fraction_{name}" for name, _ in run_outputs]

    out = (
        merged.with_columns(
            [
                pl.concat_list(minute_cols).alias("minutes_ensemble"),
                pl.concat_list(frac_cols).alias("fraction_ensemble"),
            ]
        )
        .with_columns(
            [
                pl.col("minutes_ensemble").list.median().alias("estimated_occupied_minutes_best"),
                pl.col("minutes_ensemble").list.min().alias("minutes_p10"),
                pl.col("minutes_ensemble").list.max().alias("minutes_p90"),
                pl.col("minutes_ensemble").list.mean().alias("minutes_mean_ensemble"),
                pl.col("fraction_ensemble").list.median().alias("estimated_occupied_fraction_best"),
            ]
        )
        .with_columns((pl.col("minutes_p90") - pl.col("minutes_p10")).alias("uncertainty_p90_p10"))
        .with_columns(
            pl.when(pl.col("uncertainty_p90_p10") <= 30)
            .then(pl.lit("high"))
            .when(pl.col("uncertainty_p90_p10") <= 60)
            .then(pl.lit("medium"))
            .otherwise(pl.lit("low"))
            .alias("stability_band")
        )
        .sort(["sensor_id", "block_date", "block"])
    )

    out.select(
        IDENTITY_COLS
        + META_COLS
        + [
            "estimated_occupied_minutes_best",
            "estimated_occupied_fraction_best",
            "minutes_p10",
            "minutes_p90",
            "uncertainty_p90_p10",
            "stability_band",
            "minutes_mean_ensemble",
        ]
        + minute_cols
    ).write_parquet(out_root / "block_estimates_best.parquet")

    out.select(
        IDENTITY_COLS
        + META_COLS
        + [
            "estimated_occupied_minutes_best",
            "estimated_occupied_fraction_best",
            "minutes_p10",
            "minutes_p90",
            "uncertainty_p90_p10",
            "stability_band",
            "minutes_mean_ensemble",
        ]
        + minute_cols
    ).write_csv(out_root / "block_estimates_best.csv")

    summary = {
        "n_variants": len(run_outputs),
        "variants": [name for name, _ in run_outputs],
        "rows": int(out.height),
        "subjects": int(out.select(pl.col("subject_id").n_unique()).item()),
        "sensors": int(out.select(pl.col("sensor_id").n_unique()).item()),
        "label1_mean_best_minutes": float(
            out.filter(pl.col("present") == 1).select(pl.col("estimated_occupied_minutes_best").mean()).item()
        ),
        "label0_mean_best_minutes": float(
            out.filter(pl.col("present") == 0).select(pl.col("estimated_occupied_minutes_best").mean()).item()
        ),
        "label1_uncertainty_p90_p10_mean": float(
            out.filter(pl.col("present") == 1).select(pl.col("uncertainty_p90_p10").mean()).item()
        ),
        "stability_counts": {r[0]: int(r[1]) for r in out.group_by("stability_band").len().rows()},
    }
    (out_root / "ensemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    graphs_dir = out_root / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    save_uncertainty_hist_svg(
        out.select(["uncertainty_p90_p10"]),
        graphs_dir / "uncertainty_hist_p90_p10.svg",
    )


if __name__ == "__main__":
    main()
