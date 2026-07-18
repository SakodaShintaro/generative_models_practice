"""Bench2Drive clip dataset for the video-generation testbed.

Each sample is a contiguous clip from one route's front camera:

    context frames : (T, 3, H, W)   the past, fed to the encoder
    future frames  : (N, 3, H, W)   the frames to predict
    future actions : (N, A)         controls applied at each future step

Frames come from ``camera/rgb_front/*.jpg`` and actions from ``anno/*.json.gz``.
An action is [throttle, steer, brake, speed/10]; frames are normalised to [0, 1]
(the range the frozen TAESD VAE expects).
A ``frame_stride`` subsamples the 10 Hz stream so there is visible motion between
selected frames.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

SPEED_SCALE = 10.0
ACTION_DIM = 4


def _read_action(anno_path: Path) -> list[float]:
    with gzip.open(anno_path, "rt") as f:
        d = json.load(f)
    return [
        float(d["throttle"]),
        float(d["steer"]),
        float(d["brake"]),
        float(d["speed"]) / SPEED_SCALE,
    ]


class Bench2DriveDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        image_size: int,
        context_frames: int,
        horizon: int,
        frame_stride: int,
        clip_stride: int,
        max_routes: int,
    ):
        super().__init__()
        self.image_size = image_size
        self.context_frames = context_frames
        self.horizon = horizon
        self.frame_stride = frame_stride
        span = (context_frames + horizon) * frame_stride  # raw frames covered by a clip

        routes = sorted(
            p for p in Path(data_root).iterdir() if (p / "camera" / "rgb_front").is_dir()
        )
        if max_routes > 0:
            routes = routes[:max_routes]

        # index every valid clip start as (route_idx, start_frame)
        self.routes = routes
        self.index: list[tuple[int, int]] = []
        for ri, route in enumerate(routes):
            frames = sorted((route / "camera" / "rgb_front").glob("*.jpg"))
            n = len(frames)
            for start in range(0, n - span + 1, clip_stride):
                self.index.append((ri, start))

    def __len__(self) -> int:
        return len(self.index)

    def _load_frame(self, route: Path, idx: int) -> np.ndarray:
        path = route / "camera" / "rgb_front" / f"{idx:05d}.jpg"
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR, (H, W, 3)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # [0, 1]
        return img.transpose(2, 0, 1)  # (3, H, W)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        route_idx, start = self.index[i]
        route = self.routes[route_idx]
        total = self.context_frames + self.horizon
        raw_ids = [start + k * self.frame_stride for k in range(total)]

        frames = np.stack([self._load_frame(route, j) for j in raw_ids])  # (total, 3, H, W)
        context = torch.from_numpy(frames[: self.context_frames])
        future = torch.from_numpy(frames[self.context_frames :])

        future_ids = raw_ids[self.context_frames :]
        actions = np.stack([_read_action(route / "anno" / f"{j:05d}.json.gz") for j in future_ids])
        return {
            "context": context,
            "future": future,
            "actions": torch.from_numpy(actions.astype(np.float32)),
        }
