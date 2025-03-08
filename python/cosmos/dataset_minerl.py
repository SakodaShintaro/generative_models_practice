"""MineRL tokensを読み込むdataset"""

from pathlib import Path

import numpy as np
import torch


class MineRLTokensDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, scene_low: int, scene_high: int, frame_len: int = 8) -> None:
        self.data_dir = data_dir
        subdir_list = sorted(data_dir.glob("*"))
        filtered_subdir_list = []
        for scene_id, subdir in enumerate(subdir_list):
            assert (data_dir / subdir / "tokens").exists()
            if scene_low <= scene_id <= scene_high:
                filtered_subdir_list.append(subdir)
        subdir_list = filtered_subdir_list

        self.path_list = []
        max_len = 20000
        for subdir in subdir_list:
            tokens_path_list = sorted(
                (data_dir / subdir / "tokens").glob("*.csv"),
            )
            tokens_path_list = tokens_path_list[:max_len]
            self.path_list.append(tokens_path_list)
        self.frame_len = frame_len
        self.len_per_subdir = len(self.path_list[0]) - (self.frame_len - 1)
        for i, subdir_path_list in enumerate(self.path_list):
            assert len(subdir_path_list) - (self.frame_len - 1) == self.len_per_subdir, (
                f"{i=}, {len(subdir_path_list)=}, {self.frame_len=}, {self.len_per_subdir=}"
            )

    def __len__(self) -> int:
        return len(self.path_list) * self.len_per_subdir

    def __getitem__(self, idx: int) -> torch.Tensor:
        subdir_idx = idx // self.len_per_subdir
        base_idx = idx % self.len_per_subdir

        path_list = self.path_list[subdir_idx]
        tokens_list = []
        for i in range(self.frame_len):
            tokens = np.loadtxt(path_list[base_idx + i], delimiter=",", dtype=np.int32)
            tokens_list.extend(tokens)
        tokens_list = np.stack(tokens_list, axis=0)
        return torch.from_numpy(tokens_list)  # [frame_len * num_tokens_per_frame]
