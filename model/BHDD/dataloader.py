from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BHDDDataset(Dataset):
    """Dataset for the BHDD model on the LMVD dataset.

    Expected feature shapes:
        - lm: (T_lm, 136)
        - au_pose_gaze: (T_lm, 29)
        - audio: (T_audio, 128)

    Label files are expected to be named ``{sample_id}_Depression.csv``.
    To stay compatible with the original research code, this loader supports
    multiple label CSV formats:
        1. A CSV whose *column name* is the class label (original code path).
        2. A CSV with a single cell containing the class label.
        3. A CSV with a column named ``label``.
    """

    def __init__(
        self,
        lm_path: str | Path,
        au_pose_gaze_path: str | Path,
        audio_path: str | Path,
        label_path: str | Path,
        file_list: Sequence[str],
        mode: str = "train",
        fixed_t_lm: int = 915,
        fixed_t_audio: int = 186,
    ) -> None:
        self.lm_path = Path(lm_path)
        self.au_pose_gaze_path = Path(au_pose_gaze_path)
        self.audio_path = Path(audio_path)
        self.label_path = Path(label_path)
        self.file_list = list(file_list)
        self.mode = mode
        self.fixed_t_lm = fixed_t_lm
        self.fixed_t_audio = fixed_t_audio

    @staticmethod
    def pad_or_truncate(
        x: torch.Tensor,
        max_length: int,
        dim: int = 0,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """Pad or truncate a tensor to a fixed length along one dimension."""
        current_length = x.size(dim)
        if current_length < max_length:
            pad_size = max_length - current_length
            pad_shape = list(x.size())
            pad_shape[dim] = pad_size
            pad = torch.full(pad_shape, fill_value, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=dim)
        return x.narrow(dim, 0, max_length)

    @staticmethod
    def _parse_label_value(label_file: Path) -> int:
        """Read a binary label from a CSV file.

        This helper intentionally preserves compatibility with the original code,
        which read the label from ``df.columns[0]``.
        """
        df = pd.read_csv(label_file)

        if "label" in df.columns and len(df) > 0:
            return int(df["label"].iloc[0])

        if len(df.columns) == 1:
            column_name = str(df.columns[0])
            try:
                return int(column_name)
            except ValueError:
                pass

        if df.shape[0] > 0 and df.shape[1] > 0:
            return int(df.iloc[0, 0])

        raise ValueError(f"Unable to parse label from: {label_file}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        filename = self.file_list[idx]
        sample_name = Path(filename).stem

        lm = np.load(self.lm_path / filename)
        au_pose_gaze = np.load(self.au_pose_gaze_path / filename)
        audio = np.load(self.audio_path / filename)
        label = self._parse_label_value(self.label_path / f"{sample_name}_Depression.csv")

        lm = torch.tensor(lm, dtype=torch.float32)
        au_pose_gaze = torch.tensor(au_pose_gaze, dtype=torch.float32)
        audio = torch.tensor(audio, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        audio = self.pad_or_truncate(audio, self.fixed_t_audio, dim=0)
        lm = self.pad_or_truncate(lm, self.fixed_t_lm, dim=0)
        au_pose_gaze = self.pad_or_truncate(au_pose_gaze, self.fixed_t_lm, dim=0)

        return lm, au_pose_gaze, audio, label


# Backward-compatible alias used by the original training script.
MyDataset = BHDDDataset
