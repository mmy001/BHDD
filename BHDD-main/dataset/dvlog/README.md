# D-Vlog data layout and preprocessing notes

This repository does **not** redistribute D-Vlog.

If you have obtained authorized access from the original data provider, place the files in the following structure:

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

## Reconstructed preprocessing utility

A reconstructed utility is provided to reorganize an authorized D-Vlog release into the format expected by `model/BHDD-dvlog/train.py`:

```bash
python scripts/prepare_dvlog.py   --input-root /path/to/official_dvlog   --output-root dataset/dvlog
```

The script:

- reads `labels.csv`
- uses the official `fold` column to place files into `train`, `valid`, or `test`
- copies `*_acoustic.npy` into `Audio_feature/<split>/`
- copies `*_visual.npy` into `Video_feature/<split>/`
- converts string labels (`depression`, `normal`) into binary values (`1`, `0`)
- writes per-sample label files as `<sample_id>_Depression.csv`

This placeholder layout is included only to make the public training code easier to run after authorized data access is obtained.
