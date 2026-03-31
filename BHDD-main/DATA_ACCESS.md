# Data access and redistribution policy

This repository provides **code only**.

## LMVD

Users should obtain LMVD from the original source and place the files under:

```text
dataset/LMVD/
├── Audio_feature/
├── lm(915_136)/
└── au+pose+gaze(915_29)/
```

If you reorganize LMVD locally into NumPy feature files for this codebase, be careful before redistributing those processed files in a public repository. Public availability of an original download source does not automatically guarantee that third-party redistribution is allowed.

A reconstructed preprocessing script is provided at `scripts/prepare_lmvd.py`.

## D-Vlog

D-Vlog is **not redistributed** in this repository.

If access to D-Vlog requires author approval or another controlled access procedure, users should obtain the dataset directly from the original authors and place the files under:

```text
dataset/dvlog/
├── Audio_feature/
│   ├── train/
│   ├── valid/
│   └── test/
├── Video_feature/
│   ├── train/
│   ├── valid/
│   └── test/
└── label/
    ├── train/
    ├── valid/
    └── test/
```

A reconstructed organization script is provided at `scripts/prepare_dvlog.py` for users who have obtained authorized access.

## Important reminder

Before sharing a public code repository, verify the dataset license or access policy for:

- redistribution of raw data
- redistribution of extracted features
- redistribution of labels or metadata
- redistribution of trained checkpoints when they may embed restricted data characteristics

When in doubt, share only the code and detailed setup instructions.
