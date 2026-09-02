from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BHDDDvlogDataset(Dataset):
    """Dataset loader for BHDD on the D-Vlog benchmark.

    Expected feature shapes:
        - video / landmark feature: (T_video, 136)
        - audio feature: (T_audio, 25)

    Expected label file naming:
        ``{sample_id}_Depression.csv``

    Supported label CSV formats:
        1. Original internal format where the CSV column name is the label.
        2. A single-cell CSV containing the label value.
        3. A CSV with a column named ``label``.
    """

    def __init__(
        self,
        video_path: str | Path,
        audio_path: str | Path,
        label_path: str | Path,
        file_list: Sequence[str],
        mode: str = "train",
        fixed_t_video: int = 600,
        fixed_t_audio: int = 600,
    ) -> None:
        self.video_path = Path(video_path)
        self.audio_path = Path(audio_path)
        self.label_path = Path(label_path)
        self.file_list = list(file_list)
        self.mode = mode
        self.fixed_t_video = fixed_t_video
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
        """Read a binary label from a CSV file."""
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

        video = np.load(self.video_path / filename)
        audio = np.load(self.audio_path / filename)
        label = self._parse_label_value(self.label_path / f"{sample_name}_Depression.csv")

        video = torch.tensor(video, dtype=torch.float32)
        audio = torch.tensor(audio, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        video = self.pad_or_truncate(video, self.fixed_t_video, dim=0)
        audio = self.pad_or_truncate(audio, self.fixed_t_audio, dim=0)

        return video, audio, label, sample_name


# Backward-compatible alias used by the original training script.
MyDataset = BHDDDvlogDataset
