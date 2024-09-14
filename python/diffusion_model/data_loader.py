"""DataLoader class for loading images from a directory."""

from pathlib import Path

import cv2
import numpy as np


class DataLoader:
    """DataLoader class for loading images from a directory."""

    def __init__(self, data_dir: Path, batch_size: int) -> None:
        self._image_path_list = list(Path(data_dir).glob("*.png"))
        self.batch_size = batch_size

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img / 255.0

    def __iter__(self) -> "DataLoader":
        """Return an iterator object."""
        self._index = 0
        np.random.default_rng().shuffle(self._image_path_list)
        return self

    def __next__(self) -> np.ndarray:
        if self._index >= len(self._image_path_list):
            raise StopIteration

        batch_paths = self._image_path_list[self._index : self._index + self.batch_size]
        batch_images = [self._load_image(path) for path in batch_paths]
        batch_images = np.array(batch_images, dtype=np.float32)

        self._index += self.batch_size
        return batch_images
