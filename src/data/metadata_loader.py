
from pathlib import Path

import pandas as pd


class MetadataLoader:

    REQUIRED_COLUMNS = {
        "isic_id",
        "patient_id",
        "lesion_id"
    }

    def __init__(self, data_dir):
        self._data_dir = Path(data_dir)

    def _is_valid_metadata(self, path):

        try:
            columns = pd.read_csv(
                path, nrows=0
            ).columns

            return self.REQUIRED_COLUMNS.issubset(
                columns
            )

        except (
            OSError,
            UnicodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError
        ):
            return False

    def find_metadata(self):

        if not self._data_dir.is_dir():
            raise FileNotFoundError(
                "Raw dataset directory does not exist."
            )

        preferred = self._data_dir / "metadata.csv"

        if (
            preferred.is_file()
            and self._is_valid_metadata(preferred)
        ):
            return preferred

        candidates = [
            path
            for path in sorted(self._data_dir.glob("*.csv"))
            if self._is_valid_metadata(path)
        ]

        if not candidates:
            raise FileNotFoundError(
                "No valid metadata CSV found in data/raw. "
                "The file must contain isic_id, "
                "patient_id and lesion_id columns."
            )

        if len(candidates) > 1:
            names = ", ".join(
                path.name for path in candidates
            )

            raise ValueError(
                f"Multiple metadata files found: {names}. "
                "Rename the intended file to metadata.csv."
            )

        return candidates[0]
