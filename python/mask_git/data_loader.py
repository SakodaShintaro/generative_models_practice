from glob import glob
import cv2
import numpy as np
import jax.numpy as jnp


class DataLoader:
    def __init__(self, data_dir: str, batch_size: int, max_num: int | None = None) -> None:
        self._image_path_list = glob(f"{data_dir}/*.png")
        self.batch_size = batch_size
        if max_num is not None:
            self._image_path_list = self._image_path_list[:max_num]

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        return img

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self._image_path_list):
            raise StopIteration

        batch_paths = self._image_path_list[self._index:self._index + self.batch_size]
        batch_images = [self._load_image(path) for path in batch_paths]
        batch_images = np.array(batch_images, dtype=np.float32)

        self._index += self.batch_size
        return jnp.array(batch_images)

    def shuffle(self):
        np.random.shuffle(self._image_path_list)

    def step_num_per_epoch(self) -> int:
        n = len(self._image_path_list)
        bs = self.batch_size
        return (n + bs - 1) // bs
