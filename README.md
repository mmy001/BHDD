# BHDD-main

Official public code release for **BHDD**, a multimodal depression detection framework.

This repository contains two cleaned training pipelines:

- `model/BHDD`: LMVD version
- `model/BHDD-dvlog`: D-Vlog version

The original model logic is preserved as closely as possible, while this public release adds the following improvements:

- removed private absolute paths
- added command-line arguments for data and output locations
- standardized English comments and docstrings
- improved logging and experiment output structure
- added more robust label parsing while keeping compatibility with the original label format
- preserved the project folder structure for public release
- added dataset access and non-redistribution notes for safer sharing
- added reconstructed preprocessing utilities for LMVD and D-Vlog
- added a sanitized `environment.yml` for Conda-based setup

## Repository structure

```text
BHDD-main/
├── README.md
├── DATA_ACCESS.md
├── requirements.txt
├── environment.yml
├── environment.lock.yml
├── .gitignore
├── scripts/
│   ├── prepare_lmvd.py
│   └── prepare_dvlog.py
├── model/
│   ├── BHDD/
│   │   ├── dataloader.py
│   │   ├── model.py
│   │   └── train.py
│   └── BHDD-dvlog/
│       ├── dataloader.py
│       ├── model.py
│       └── train.py
├── dataset/
│   ├── README.md
│   ├── LMVD/
│   │   ├── README.md
│   │   ├── Audio_feature/
│   │   ├── lm(915_136)/
│   │   └── au+pose+gaze(915_29)/
│   └── dvlog/
│       ├── README.md
│       ├── Audio_feature/
│       │   ├── train/
│       │   ├── valid/
│       │   └── test/
│       ├── Video_feature/
│       │   ├── train/
│       │   ├── valid/
│       │   └── test/
│       └── label/
│           ├── train/
│           ├── valid/
│           └── test/
└── outputs/
```

## Environment setup

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate BHDD-main
```

### Option 2: closer snapshot of the original local Conda environment

```bash
conda env create -f environment.lock.yml
conda activate LMVD-main
```

### Option 3: pip

```bash
pip install -r requirements.txt
```

`environment.yml` is a cleaned public Conda environment for this repository. `environment.lock.yml` is a sanitized version of the original local environment snapshot you exported, with the machine-specific `prefix` entry removed for portability.

## Data availability and redistribution

This repository **does not include any dataset files by default**.

- **LMVD**: users should obtain the dataset from the official source. If you reorganize LMVD locally into the format expected by this repository, keep those files local unless redistribution is explicitly permitted by the dataset terms.
- **D-Vlog**: this repository does **not** redistribute D-Vlog. Users should request access from the original authors and place the approved files under `dataset/dvlog/` following the folder layout in this repository.

Please make sure your use of each dataset complies with the corresponding license, terms of use, and access policy.

Additional notes are provided in [`DATA_ACCESS.md`](DATA_ACCESS.md).

## LMVD preprocessing

The LMVD pipeline in this repository uses reorganized visual features rather than the raw per-sample CSV files directly. Specifically:

- landmark features are stored as `.npy` files under `dataset/LMVD/lm(915_136)/`
- AU, pose, and gaze features are concatenated and stored as `.npy` files under `dataset/LMVD/au+pose+gaze(915_29)/`

Because the exact original preprocessing script used in the early internal experiments was not preserved, this repository provides a **reconstructed preprocessing utility** for reproducibility:

```bash
python scripts/prepare_lmvd.py
  --input-dir /path/to/original_lmvd_visual_csv
  --output-root dataset/LMVD
```

See `dataset/LMVD/README.md` for details and customization notes.

## D-Vlog preprocessing

After obtaining authorized access to the official D-Vlog release, you can reorganize the files into the directory structure expected by this repository with:

```bash
python scripts/prepare_dvlog.py
  --input-root /path/to/official_dvlog
  --output-root dataset/dvlog
```

This reconstructed utility copies acoustic and visual features into split-specific subdirectories and converts string labels into per-sample binary label CSV files.

## Training on LMVD

Example:

```bash
python model/BHDD/train.py
  --lm-dir dataset/LMVD/lm(915_136)
  --au-pose-gaze-dir dataset/LMVD/au+pose+gaze(915_29)
  --audio-dir dataset/LMVD/Audio_feature
  --label-dir /path/to/lmvd_labels
  --output-dir outputs/lmvd
  --run-name bhdd_lmvd
```

## Training on D-Vlog

Example:

```bash
python model/BHDD-dvlog/train.py
  --video-root dataset/dvlog/Video_feature
  --audio-root dataset/dvlog/Audio_feature
  --label-root dataset/dvlog/label
  --train-split train
  --eval-split valid
  --output-dir outputs/dvlog
  --run-name bhdd_dvlog
```

If you want to follow an alternative evaluation protocol, you can switch `--eval-split` to `test`.

## Label format compatibility

Both data loaders support these label CSV formats:

1. a CSV whose **column name** is the label value
2. a CSV whose first cell is the label value
3. a CSV with a column named `label`

## Outputs

Each run creates a directory like:

```text
outputs/<dataset>/<run_name>/
├── checkpoints/
├── config.json
├── confusion_matrix.png
├── metrics_summary.json
├── sample_predictions.csv
└── training.log
```

## Notes for public release

- Do not commit raw data files, extracted features, labels, or trained checkpoints unless redistribution is explicitly permitted.
- The default `.gitignore` is configured to reduce the risk of accidentally uploading raw arrays, checkpoints, and run outputs.
- If you store LMVD locally in `dataset/LMVD/`, Git will still ignore the feature arrays by default.
- Before publishing the repository, it is recommended to add a `LICENSE` file and citation metadata.
