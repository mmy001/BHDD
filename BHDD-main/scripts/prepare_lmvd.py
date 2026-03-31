from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_LM_PATTERNS = [
    r"^x_\d+$",
    r"^y_\d+$",
    r"^lm[_-]",
    r"^landmark",
]
DEFAULT_AU_PATTERNS = [r"^AU"]
DEFAULT_POSE_PATTERNS = [r"^pose_", r"^pose", r"^head_pose"]
DEFAULT_GAZE_PATTERNS = [r"^gaze_", r"^gaze"]
DEFAULT_IGNORE_PATTERNS = [
    r"^frame$",
    r"^timestamp$",
    r"^face_id$",
    r"^success$",
    r"^confidence$",
    r"^Unnamed:",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstructed preprocessing script for LMVD visual features. "
            "It converts per-sample CSV files into NumPy arrays expected by the public BHDD code."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing original LMVD per-sample visual CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/LMVD"),
        help="Root directory where processed LMVD feature folders will be created.",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for input files. Default: *.csv",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for CSV files under --input-dir.",
    )
    parser.add_argument(
        "--lm-patterns",
        nargs="*",
        default=DEFAULT_LM_PATTERNS,
        help=(
            "Regex patterns for LM columns. Defaults are suitable for common OpenFace-style landmark columns."
        ),
    )
    parser.add_argument(
        "--au-patterns",
        nargs="*",
        default=DEFAULT_AU_PATTERNS,
        help="Regex patterns for AU columns.",
    )
    parser.add_argument(
        "--pose-patterns",
        nargs="*",
        default=DEFAULT_POSE_PATTERNS,
        help="Regex patterns for pose columns.",
    )
    parser.add_argument(
        "--gaze-patterns",
        nargs="*",
        default=DEFAULT_GAZE_PATTERNS,
        help="Regex patterns for gaze columns.",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=DEFAULT_IGNORE_PATTERNS,
        help="Regex patterns for columns to ignore before feature grouping.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if a file does not produce both LM and AU+pose+gaze features.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect files and print a summary without writing .npy outputs.",
    )
    return parser.parse_args()


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def select_columns(columns: list[str], include_patterns: list[str], exclude_patterns: list[str]) -> list[str]:
    selected: list[str] = []
    for col in columns:
        if matches_any(col, exclude_patterns):
            continue
        if matches_any(col, include_patterns):
            selected.append(col)
    return selected


def build_output_dirs(output_root: Path) -> tuple[Path, Path]:
    lm_dir = output_root / "lm(915_136)"
    apg_dir = output_root / "au+pose+gaze(915_29)"
    lm_dir.mkdir(parents=True, exist_ok=True)
    apg_dir.mkdir(parents=True, exist_ok=True)
    return lm_dir, apg_dir


def iter_csv_files(input_dir: Path, file_pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted(input_dir.rglob(file_pattern))
    else:
        files = sorted(input_dir.glob(file_pattern))
    return [f for f in files if f.is_file()]


def main() -> None:
    args = parse_args()
    lm_dir, apg_dir = build_output_dirs(args.output_root)

    csv_files = iter_csv_files(args.input_dir, args.file_pattern, args.recursive)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir} with pattern {args.file_pattern!r}.")

    summary: dict[str, object] = {
        "input_dir": str(args.input_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "num_files": len(csv_files),
        "processed": 0,
        "skipped": [],
        "examples": [],
        "note": (
            "This is a reconstructed preprocessing utility for public reproducibility. "
            "Please verify column grouping against your local LMVD release if it differs from the default assumptions."
        ),
    }

    for idx, csv_path in enumerate(csv_files, start=1):
        df = pd.read_csv(csv_path)
        all_columns = [str(col) for col in df.columns]

        lm_columns = select_columns(all_columns, args.lm_patterns, args.ignore_patterns)
        au_columns = select_columns(all_columns, args.au_patterns, args.ignore_patterns)
        pose_columns = select_columns(all_columns, args.pose_patterns, args.ignore_patterns)
        gaze_columns = select_columns(all_columns, args.gaze_patterns, args.ignore_patterns)
        apg_columns = au_columns + pose_columns + gaze_columns

        if not lm_columns or not apg_columns:
            message = {
                "file": str(csv_path),
                "lm_columns": len(lm_columns),
                "au_columns": len(au_columns),
                "pose_columns": len(pose_columns),
                "gaze_columns": len(gaze_columns),
            }
            if args.strict:
                raise ValueError(f"Unable to extract required columns from {csv_path}: {message}")
            summary["skipped"].append(message)
            continue

        lm_array = df[lm_columns].to_numpy(dtype=np.float32)
        apg_array = df[apg_columns].to_numpy(dtype=np.float32)

        sample_id = csv_path.stem
        if not args.dry_run:
            np.save(lm_dir / f"{sample_id}.npy", lm_array)
            np.save(apg_dir / f"{sample_id}.npy", apg_array)

        summary["processed"] = int(summary["processed"]) + 1
        if len(summary["examples"]) < 5:
            summary["examples"].append(
                {
                    "file": csv_path.name,
                    "lm_shape": list(lm_array.shape),
                    "au_pose_gaze_shape": list(apg_array.shape),
                    "lm_columns_preview": lm_columns[:6],
                    "apg_columns_preview": apg_columns[:6],
                }
            )

        if idx % 50 == 0 or idx == len(csv_files):
            print(f"Processed {idx}/{len(csv_files)} files...")

    summary_path = args.output_root / "lmvd_preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Summary written to: {summary_path}")
    print(f"LM output directory: {lm_dir}")
    print(f"AU+pose+gaze output directory: {apg_dir}")


if __name__ == "__main__":
    main()
