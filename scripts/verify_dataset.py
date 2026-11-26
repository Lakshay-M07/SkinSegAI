
from pathlib import Path
import sys

# Ensure project root is in sys.path when script is executed directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.dataset_manager import DatasetManager


def main():

    project_root = Path(__file__).resolve().parents[1]

    manager = DatasetManager(
        project_root / "data" / "raw"
    )

    print("===== DATASET INFORMATION =====")

    manager.load_dataset()

    errors = manager.validate_dataset()

    if errors:
        print("\nDetected errors:")

        for filename, error in errors[:10]:
            print(f"{filename}: {error}")

        raise SystemExit(1)

    print("\nDataset verification successful!")


if __name__ == "__main__":
    main()
