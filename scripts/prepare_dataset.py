
from src.data.metadata_loader import MetadataLoader
from pathlib import Path
import csv

from tqdm import tqdm

from src.data.dataset_manager import DatasetManager
from src.data.dataset_splitter import DatasetSplitter
from src.data.image_preprocessor import ImagePreprocessor


def main():

    root = Path(__file__).resolve().parents[1]

    manager = DatasetManager(root / "data" / "raw")
    manager.load_dataset()

    metadata_loader = MetadataLoader(root / "data/raw")
    metadata_path = metadata_loader.find_metadata()
    print("Metadata file:", metadata_path.name)


    splitter = DatasetSplitter(
    metadata_path=root / "data/raw/metadata.csv",
    random_seed=42
)

    preprocessor = ImagePreprocessor(
        image_size=(256, 256)
    )

    splits = splitter.split(manager.get_pairs())

    processed_dir = root / "data" / "processed_grouped"
    splits_dir = root / "data" / "splits_grouped"

    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, pairs in splits.items():

        print(f"\nPreparing {split_name} dataset")

        records = []

        for image_path, mask_path in tqdm(pairs):

            image, mask = preprocessor.process_pair(
                image_path,
                mask_path,
                processed_dir / split_name
            )

            records.append({
                "image": image.relative_to(root).as_posix(),
                "mask": mask.relative_to(root).as_posix()
            })

        csv_path = splits_dir / f"{split_name}.csv"

        with open(csv_path, "w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=["image", "mask"]
            )

            writer.writeheader()
            writer.writerows(records)

        print(f"Processed samples: {len(records)}")

    print("\n===== DATASET PREPARATION COMPLETE =====")

    for name, pairs in splits.items():
        print(f"{name.capitalize()}: {len(pairs)}")

    print("Total:", sum(map(len, splits.values())))


if __name__ == "__main__":
    main()
