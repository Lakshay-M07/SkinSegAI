
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.image_preprocessor import ImagePreprocessor


class LesionDataset(Dataset):

    def __init__(self, csv_path, project_root, augment=False,
                 mean=None, std=None):

        self._root = Path(project_root)
        self._augment = augment
        self._preprocessor = ImagePreprocessor(
            mean=mean, std=std
        )

        with open(csv_path, newline="") as file:
            self._records = list(csv.DictReader(file))

        if not self._records:
            raise ValueError("Dataset is empty.")

    def __len__(self):
        return len(self._records)

    def _apply_augmentation(self, image, mask):

        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)

        if random.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)

        k = random.randint(0, 3)

        image = np.rot90(image, k)
        mask = np.rot90(mask, k)

        if random.random() < 0.5:
            brightness = random.uniform(0.85, 1.15)
            image = np.clip(image * brightness, 0, 1)

        return image, mask

    def __getitem__(self, index):

        record = self._records[index]

        image_path = self._root / record["image"]
        mask_path = self._root / record["mask"]

        with Image.open(image_path) as image:
            image = self._preprocessor.normalize_image(
                image.convert("RGB")
            )

        with Image.open(mask_path) as mask:
            mask = self._preprocessor.normalize_mask(
                mask.convert("L")
            )

        if self._augment:
            image, mask = self._apply_augmentation(
                image, mask
            )

        image = np.ascontiguousarray(
            image.transpose(2, 0, 1)
        )

        mask = np.ascontiguousarray(
            mask[np.newaxis, :, :]
        )

        return (
            torch.from_numpy(image).float(),
            torch.from_numpy(mask).float()
        )
