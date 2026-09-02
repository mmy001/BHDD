# Dataset placeholders

This directory preserves the expected folder layout for reproducibility.

It is intentionally released **without raw data, extracted features, or labels**.

## Folder overview

```text
dataset/
├── LMVD/
│   ├── Audio_feature/
│   ├── lm(915_136)/
│   └── au+pose+gaze(915_29)/
└── dvlog/
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

## Usage

- Put publicly obtainable LMVD files under `dataset/LMVD/`.
- Put D-Vlog files under `dataset/dvlog/` only after obtaining permission or access from the original source.
- Do not commit dataset contents back to a public repository unless redistribution is explicitly allowed.
