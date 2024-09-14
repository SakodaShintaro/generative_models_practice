from glob import glob
import cv2
import numpy as np


class DataLoader:
    def __init__(self, data_dir: str, batch_size: int) -> None:
        self._image_path_list = glob(f"{data_dir}/*.png")
        self.batch_size = batch_size

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        return img

    def __iter__(self):
        self._index = 0
        np.random.shuffle(self._image_path_list)
        return self

    def __next__(self):
        if self._index >= len(self._image_path_list):
            raise StopIteration

        batch_paths = self._image_path_list[self._index : self._index + self.batch_size]
        batch_images = [self._load_image(path) for path in batch_paths]
        batch_images = np.array(batch_images, dtype=np.float32)

        self._index += self.batch_size
        return batch_images
