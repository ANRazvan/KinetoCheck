from pathlib import Path

import pandas as pd


def main() -> None:
    base = Path("temporal_pyramid_stgat/outputs/video_assessment")
    rows = []

    for run_dir in sorted(base.glob("deep_squat_multiple_repetitions*__pyramid_stgat*exercise_0*")):
        if not run_dir.is_dir():
            continue

        summaries = list(run_dir.glob("*_summary.csv"))
        if not summaries:
            continue

        summary_path = summaries[0]
        df = pd.read_csv(summary_path)
        if df.empty:
            continue

        row = df.iloc[0]
        video_path = str(row.get("video_path", ""))
        checkpoint_path = str(row.get("checkpoint_path", ""))

        rows.append(
            {
                "video_name": Path(video_path).stem if video_path else summary_path.stem.replace("_summary", ""),
                "checkpoint_name": Path(checkpoint_path).stem if checkpoint_path else run_dir.name.split("__")[-1],
                "prediction_label": str(row.get("prediction_label", "")),
                "quality_score_mean": float(row.get("quality_score_mean", 0.0)),
                "confidence_mean": float(row.get("confidence_mean", 0.0)),
                "num_windows": int(row.get("num_windows", 0)),
                "frames_total": int(row.get("frames_total", 0)),
                "pose_backend": str(row.get("pose_backend", "")),
                "summary_csv": str(summary_path.resolve()),
            }
        )

    if not rows:
        raise RuntimeError("No summary CSV rows found for multi-repetition benchmark runs.")

    combined = pd.DataFrame(rows).sort_values(["video_name", "checkpoint_name"]).reset_index(drop=True)
    combined_path = base / "deep_squat_multi_reps_all_models_summary.csv"
    combined.to_csv(combined_path, index=False)

    leaderboard = (
        combined.groupby("checkpoint_name", as_index=False)
        .agg(
            videos_tested=("video_name", "count"),
            mean_quality_score=("quality_score_mean", "mean"),
            mean_confidence=("confidence_mean", "mean"),
        )
    )

    pred_map = (
        combined.sort_values(["checkpoint_name", "video_name"])
        .groupby("checkpoint_name")
        .apply(
            lambda g: "; ".join(
                f"{video}:{pred}"
                for video, pred in zip(g["video_name"], g["prediction_label"])
            )
        )
        .rename("predictions")
        .reset_index()
    )

    leaderboard = leaderboard.merge(pred_map, on="checkpoint_name", how="left")
    leaderboard = leaderboard.sort_values(
        ["mean_quality_score", "mean_confidence"], ascending=[False, True]
    ).reset_index(drop=True)

    leaderboard_path = base / "deep_squat_multi_reps_all_models_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False, float_format="%.6f")

    print(f"WROTE {combined_path.resolve()}")
    print(f"WROTE {leaderboard_path.resolve()}")
    print("\nTop leaderboard rows:")
    print(leaderboard.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
