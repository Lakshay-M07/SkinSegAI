
import hashlib
from collections import Counter

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


class DatasetSplitter:

    def __init__(self, metadata_path, random_seed=42):

        self._random_seed = random_seed

        self._metadata = pd.read_csv(
            metadata_path, dtype=str
        )

        required = {"isic_id", "patient_id", "lesion_id"}

        if not required.issubset(self._metadata.columns):
            raise ValueError("Required metadata columns missing.")

        if self._metadata["isic_id"].duplicated().any():
            raise ValueError("Duplicate ISIC metadata records.")

        self._metadata = self._metadata.set_index("isic_id")

    @staticmethod
    def _image_hash(path):

        with open(path, "rb") as file:
            return hashlib.file_digest(
                file, "sha256"
            ).hexdigest()

    @staticmethod
    def _clean_id(value):

        if pd.isna(value):
            return None

        value = str(value).strip()

        if value.lower() in {
            "", "unknown", "none", "nan",
            "not available", "n/a", "null"
        }:
            return None

        return value

    def _build_groups(self, pairs):

        parents = list(range(len(pairs)))
        owners = {}

        def find(index):

            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]

            return index

        def union(a, b):

            root_a = find(a)
            root_b = find(b)

            if root_a != root_b:
                parents[root_b] = root_a

        missing_patients = 0
        missing_lesions = 0

        for index, (image_path, _) in enumerate(pairs):

            image_id = image_path.stem

            if image_id not in self._metadata.index:
                raise ValueError(
                    f"Missing metadata: {image_id}"
                )

            row = self._metadata.loc[image_id]

            patient = self._clean_id(row["patient_id"])
            lesion = self._clean_id(row["lesion_id"])

            keys = [
                ("hash", self._image_hash(image_path))
            ]

            if patient is not None:
                keys.append(("patient", patient))
            else:
                missing_patients += 1

            if lesion is not None:
                keys.append(("lesion", lesion))
            else:
                missing_lesions += 1

            for key in keys:

                if key in owners:
                    union(index, owners[key])
                else:
                    owners[key] = index

        groups = [find(i) for i in range(len(pairs))]

        print("\n===== METADATA AUDIT =====")
        print("Total images:", len(pairs))
        print("Missing patient IDs:", missing_patients)
        print("Missing lesion IDs:", missing_lesions)
        print("Independent groups:", len(set(groups)))
        print("Largest group:", max(Counter(groups).values()))

        return groups

    def split(self, pairs):

        if not pairs:
            raise ValueError("Dataset is empty.")

        groups = self._build_groups(pairs)

        best = None
        best_error = float("inf")

        # Find group-safe splits close to 70/15/15.
        for attempt in range(32):

            seed = self._random_seed + attempt

            first = GroupShuffleSplit(
                n_splits=1,
                train_size=0.70,
                random_state=seed
            )

            train, remaining = next(
                first.split(pairs, groups=groups)
            )

            remaining_groups = [
                groups[i] for i in remaining
            ]

            second = GroupShuffleSplit(
                n_splits=1,
                train_size=0.50,
                random_state=seed
            )

            val_local, test_local = next(
                second.split(
                    remaining,
                    groups=remaining_groups
                )
            )

            indices = {
                "train": list(train),
                "validation": [
                    remaining[i] for i in val_local
                ],
                "test": [
                    remaining[i] for i in test_local
                ]
            }

            error = sum(
                abs(len(indices[name]) / len(pairs) - ratio)
                for name, ratio in [
                    ("train", 0.70),
                    ("validation", 0.15),
                    ("test", 0.15)
                ]
            )

            if error < best_error:
                best_error = error
                best = indices

        group_sets = {
            name: {groups[i] for i in indices}
            for name, indices in best.items()
        }

        assert not (
            group_sets["train"] & group_sets["validation"]
        )

        assert not (
            group_sets["train"] & group_sets["test"]
        )

        assert not (
            group_sets["validation"] & group_sets["test"]
        )

        all_indices = [
            i for indices in best.values() for i in indices
        ]

        assert len(all_indices) == len(set(all_indices))
        assert set(all_indices) == set(range(len(pairs)))

        print("\n===== LEAKAGE CHECK =====")
        print("Cross-split group overlap: 0")
        print("Group separation: PASSED")

        return {
            name: [pairs[i] for i in indices]
            for name, indices in best.items()
        }
