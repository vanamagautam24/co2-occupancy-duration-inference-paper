#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


KEY_COLS = ["subject_id", "sensor_id", "block_date", "block", "present"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine multiple CO2 occupancy estimators into an uncertainty-aware consensus."
    )
    p.add_argument(
        "--physics-robust-dir",
        default="outputs_spring2025_physics_robust",
        help="Directory containing block_estimates_physics_robust.csv",
    )
    p.add_argument(
        "--room-decomp-dir",
        default="outputs_spring2025_room_decomposition",
        help="Directory containing block_estimates_room_decomposition.csv",
    )
    p.add_argument(
        "--output-dir",
        default="outputs_spring2025_consensus",
    )
    return p.parse_args()


def _save_hist(block_est: pl.DataFrame, out_path: Path) -> None:
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
        "<text x='90' y='36' font-size='28' font-family='sans-serif'>Consensus Uncertainty (p90-p10, minutes)</text>",
        f"<line x1='{left}' y1='{top+ph}' x2='{left+pw}' y2='{top+ph}' stroke='black'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+ph}' stroke='black'/>",
    ]
    for i, c in enumerate(hist):
        x = left + i * bw
        bh = (c / ymax) * ph
        parts.append(
            f"<rect x='{x+4:.1f}' y='{top+ph-bh:.1f}' width='{bw-8:.1f}' height='{bh:.1f}' fill='#1D3557'/>"
        )
        parts.append(
            f"<text x='{x+bw/2-12:.1f}' y='{top+ph+24}' font-size='12' font-family='sans-serif'>{int(bins[i])}-{int(bins[i+1])}</text>"
        )
    parts.append("</svg>")
    out_path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graphs").mkdir(exist_ok=True)

    robust = pl.read_csv(Path(args.physics_robust_dir) / "block_estimates_physics_robust.csv")
    room = pl.read_csv(Path(args.room_decomp_dir) / "block_estimates_room_decomposition.csv")

    # Keep one row per block key from room outputs for shared-room context.
    room_ctx = (
        room.select(
            KEY_COLS
            + [
                "n_present_subjects_block",
                "estimated_occupied_minutes_best",
                "estimated_occupied_minutes_p10",
                "estimated_occupied_minutes_p90",
                "uncertainty_p90_p10",
            ]
        )
        .rename(
            {
                "estimated_occupied_minutes_best": "room_best",
                "estimated_occupied_minutes_p10": "room_p10",
                "estimated_occupied_minutes_p90": "room_p90",
                "uncertainty_p90_p10": "room_unc",
            }
        )
    )
    robust_ctx = (
        robust.select(
            KEY_COLS
            + [
                "estimated_occupied_minutes_best",
                "estimated_occupied_minutes_p10",
                "estimated_occupied_minutes_p90",
                "uncertainty_p90_p10",
                "block_start_ts",
                "block_end_ts",
                "room_type",
                "group_id_deploy",
                "Address",
                "building_code",
                "hallname",
                "vent_cat",
            ]
        )
        .rename(
            {
                "estimated_occupied_minutes_best": "robust_best",
                "estimated_occupied_minutes_p10": "robust_p10",
                "estimated_occupied_minutes_p90": "robust_p90",
                "uncertainty_p90_p10": "robust_unc",
            }
        )
    )

    merged = robust_ctx.join(room_ctx, on=KEY_COLS, how="inner")
    if merged.height == 0:
        raise RuntimeError("No overlapping rows between robust and room decomposition outputs.")

    robust_best = merged["robust_best"].to_numpy()
    robust_unc = merged["robust_unc"].to_numpy()
    room_best = merged["room_best"].to_numpy()
    room_unc = merged["room_unc"].to_numpy()
    present = merged["present"].to_numpy()
    n_present_subj = merged["n_present_subjects_block"].fill_null(1).to_numpy()

    # Convert p90-p10 widths into variance proxies with floor to avoid overconfidence.
    sigma_r = np.maximum(robust_unc / 2.563, 6.0)
    sigma_d = np.maximum(room_unc / 2.563, 8.0)
    var_r = sigma_r * sigma_r
    var_d = sigma_d * sigma_d

    # Shared blocks receive mild boost for room-decomposition contribution.
    boost = np.where(n_present_subj >= 2, 1.25, 1.0)
    inv_r = 1.0 / var_r
    inv_d = boost / var_d

    mean = (inv_r * robust_best + inv_d * room_best) / (inv_r + inv_d)
    sigma_comb = np.sqrt(1.0 / (inv_r + inv_d))
    model_gap = np.abs(robust_best - room_best)
    sigma_total = np.sqrt(sigma_comb * sigma_comb + (0.5 * model_gap) ** 2)

    p10 = np.clip(mean - 1.2816 * sigma_total, 0.0, 240.0)
    p90 = np.clip(mean + 1.2816 * sigma_total, 0.0, 240.0)
    best = np.clip(mean, 0.0, 240.0)

    absent_mask = present == 0
    best[absent_mask] = 0.0
    p10[absent_mask] = 0.0
    p90[absent_mask] = 0.0

    out = merged.with_columns(
        [
            pl.Series("estimated_occupied_minutes_best", best),
            pl.Series("estimated_occupied_minutes_p10", p10),
            pl.Series("estimated_occupied_minutes_p90", p90),
            pl.Series("model_disagreement_minutes", model_gap),
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
        ]
    ).sort(["sensor_id", "block_date", "block", "subject_id"])

    daily = (
        out.group_by(["subject_id", "block_date"])
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

    out.write_parquet(out_dir / "block_estimates_consensus.parquet")
    out.write_csv(out_dir / "block_estimates_consensus.csv")
    daily.write_parquet(out_dir / "daily_estimates_consensus.parquet")
    daily.write_csv(out_dir / "daily_estimates_consensus.csv")
    individual.write_csv(out_dir / "individual_summary_consensus.csv")
    _save_hist(out, out_dir / "graphs" / "uncertainty_histogram.svg")

    label1 = out.filter(pl.col("present") == 1)
    label0 = out.filter(pl.col("present") == 0)
    summary = {
        "rows_subject_block": int(out.height),
        "rows_daily": int(daily.height),
        "subjects": int(out.select(pl.col("subject_id").n_unique()).item()),
        "sensors": int(out.select(pl.col("sensor_id").n_unique()).item()),
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
        "label1_model_disagreement_mean": float(
            label1.select(pl.col("model_disagreement_minutes").mean()).item()
        )
        if label1.height
        else None,
        "stability_counts": {
            row[0]: int(row[1]) for row in out.group_by("stability_band").len().rows()
        },
    }
    (out_dir / "summary_consensus.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
