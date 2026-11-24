
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class DatasetManager:

    def __init__(self, dataset_path):
        self._dataset_path = Path(dataset_path)

        self._image_dir = (
            self._dataset_path /
            "ISIC2018_Task1-2_Training_Input"
        )

        self._mask_dir = (
            self._dataset_path /
            "ISIC2018_Task1_Training_GroundTruth"
        )

        self._pairs = []

    @property
    def total_samples(self):
        return len(self._pairs)

    def load_dataset(self):

        if not self._image_dir.is_dir():
            raise FileNotFoundError(self._image_dir)

        if not self._mask_dir.is_dir():
            raise FileNotFoundError(self._mask_dir)

        images = {
            path.stem: path
            for path in self._image_dir.glob("*.jpg")
        }

        masks = {
            path.stem.removesuffix("_segmentation"): path
            for path in self._mask_dir.glob(
                "*_segmentation.png"
            )
        }

        missing_masks = images.keys() - masks.keys()
        missing_images = masks.keys() - images.keys()

        self._pairs = [
            (images[key], masks[key])
            for key in sorted(images.keys() & masks.keys())
        ]

        print(f"Total images: {len(images)}")
        print(f"Total masks: {len(masks)}")
        print(f"Matched pairs: {self.total_samples}")
        print(f"Missing masks: {len(missing_masks)}")
        print(f"Missing images: {len(missing_images)}")

        if missing_masks or missing_images:
            raise ValueError("Unmatched dataset files found.")

        if not self._pairs:
            raise ValueError("No image-mask pairs found.")

    def validate_dataset(self):

        if not self._pairs:
            raise RuntimeError("Load the dataset first.")

        errors = []

        for image_path, mask_path in tqdm(
            self._pairs,
            desc="Validating dataset"
        ):
            try:
                with Image.open(image_path) as image:
                    with Image.open(mask_path) as mask:

                        image.load()
                        mask.load()

                        if image.size != mask.size:
                            raise ValueError(
                                "Dimension mismatch"
                            )

                        grayscale = mask.convert("L")
                        histogram = grayscale.histogram()

                        if sum(histogram[1:255]) > 0:
                            raise ValueError(
                                "Non-binary mask"
                            )

                        if histogram[255] == 0:
                            raise ValueError(
                                "Empty lesion mask"
                            )

            except Exception as error:
                errors.append(
                    (image_path.name, str(error))
                )

        print("\n===== VALIDATION RESULTS =====")
        print(f"Total samples: {self.total_samples}")
        print(f"Valid samples: {self.total_samples - len(errors)}")
        print(f"Invalid samples: {len(errors)}")

        return errors

    def get_pairs(self):
        return self._pairs.copy()
