# LMVD data layout and preprocessing notes

This directory stores the LMVD features used by `model/BHDD/train.py`.

Expected structure:

```text
dataset/LMVD/
├── Audio_feature/
├── lm(915_136)/
├── au+pose+gaze(915_29)/
└── lmvd_preprocessing_summary.json
```

## What the released code expects

The public BHDD LMVD pipeline expects three feature sources:

- `Audio_feature/`: precomputed audio features
- `lm(915_136)/`: per-sample NumPy arrays containing only landmark features
- `au+pose+gaze(915_29)/`: per-sample NumPy arrays containing AU, pose, and gaze features concatenated together

Label files are supplied separately through the `--label-dir` argument of `model/BHDD/train.py`.

## Preprocessing note

In the original project workflow, the LMVD visual features were reorganized before training.
The raw LMVD visual features were provided as per-sample CSV files containing multiple visual feature groups.
For the BHDD codebase, these files were converted into NumPy arrays for easier and faster loading:

- landmark features were stored separately in `lm(915_136)/`
- AU, pose, and gaze features were concatenated and stored in `au+pose+gaze(915_29)/`

This step only changes feature organization and storage format for training efficiency. It does not change the original labels or dataset split.

## Reconstructed preprocessing utility

The exact original preprocessing script used in the first experiments was not preserved.
For reproducibility, this repository provides a **reconstructed preprocessing utility**:

```bash
python scripts/prepare_lmvd.py   --input-dir /path/to/original_lmvd_visual_csv   --output-root dataset/LMVD
```

The script attempts to infer column groups using common OpenFace-style naming conventions:

- LM: `x_*`, `y_*`, `lm_*`, or `landmark*`
- AU: `AU*`
- pose: `pose*` or `head_pose*`
- gaze: `gaze*`

If your local LMVD release uses different column names, adjust the regex patterns via command-line arguments such as `--lm-patterns`, `--au-patterns`, `--pose-patterns`, and `--gaze-patterns`.

## Important note for public release

Even if you store LMVD locally in this directory for training, be careful before redistributing processed feature files in a public repository.
When in doubt, share the code and preprocessing instructions, and let users obtain the dataset from the original source.
