from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstructed preprocessing script for an authorized D-Vlog release. "
            "It reorganizes acoustic/visual features into the directory structure expected by the public BHDD code."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory of the official D-Vlog release containing labels.csv and per-sample folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/dvlog"),
        help="Output root where Audio_feature, Video_feature, and label folders will be created.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of trying hard links first. Default behavior prefers hard links when possible.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately when a sample folder, feature file, or label entry is missing.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "audio": output_root / "Audio_feature",
        "video": output_root / "Video_feature",
        "label": output_root / "label",
    }
    for base in dirs.values():
        for split in ("train", "valid", "test"):
            (base / split).mkdir(parents=True, exist_ok=True)
    return dirs


def link_or_copy(src: Path, dst: Path, force_copy: bool) -> None:
    if dst.exists():
        dst.unlink()
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def normalize_label(label: str) -> int:
    label_l = label.strip().lower()
    if label_l == "depression":
        return 1
    if label_l == "normal":
        return 0
    raise ValueError(f"Unsupported label value: {label!r}")


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    labels_csv = input_root / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"Could not find labels.csv at: {labels_csv}")

    out_dirs = ensure_output_dirs(args.output_root)
    summary: dict[str, object] = {
        "input_root": str(input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "processed": 0,
        "missing": [],
        "invalid": [],
        "note": (
            "This is a reconstructed public preprocessing utility. "
            "Please verify the file naming and split fields against your authorized D-Vlog release."
        ),
    }

    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required_columns = {"index", "label", "fold"}
    missing_cols = required_columns - set(rows[0].keys()) if rows else required_columns
    if missing_cols:
        raise ValueError(f"labels.csv is missing required columns: {sorted(missing_cols)}")

    for row in rows:
        sample_id = str(row["index"]).strip()
        split = str(row["fold"]).strip()
        if split not in {"train", "valid", "test"}:
            message = {"sample": sample_id, "reason": f"invalid split: {split!r}"}
            if args.strict:
                raise ValueError(str(message))
            summary["invalid"].append(message)
            continue

        try:
            label_value = normalize_label(str(row["label"]))
        except ValueError as exc:
            message = {"sample": sample_id, "reason": str(exc)}
            if args.strict:
                raise
            summary["invalid"].append(message)
            continue

        sample_dir = input_root / sample_id
        acoustic_src = sample_dir / f"{sample_id}_acoustic.npy"
        visual_src = sample_dir / f"{sample_id}_visual.npy"

        missing = [str(p) for p in (sample_dir, acoustic_src, visual_src) if not p.exists()]
        if missing:
            message = {"sample": sample_id, "missing": missing}
            if args.strict:
                raise FileNotFoundError(str(message))
            summary["missing"].append(message)
            continue

        audio_dst = out_dirs["audio"] / split / f"{sample_id}.npy"
        video_dst = out_dirs["video"] / split / f"{sample_id}.npy"
        label_dst = out_dirs["label"] / split / f"{sample_id}_Depression.csv"

        link_or_copy(acoustic_src, audio_dst, args.copy)
        link_or_copy(visual_src, video_dst, args.copy)
        label_dst.write_text(f"{label_value}\n", encoding="utf-8")

        summary["processed"] = int(summary["processed"]) + 1

    summary_path = args.output_root / "dvlog_preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done.")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
