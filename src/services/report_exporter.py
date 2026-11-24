
import io
import json
import zipfile

import numpy as np
from PIL import Image

from src.core.segmentation_result import SegmentationResult


class ReportExporter:

    @staticmethod
    def _image_to_bytes(image):

        buffer = io.BytesIO()

        image.save(buffer, format="PNG")

        return buffer.getvalue()

    def export_zip(self, results):

        results = list(results)

        if not results:
            raise ValueError("No results to export.")

        buffer = io.BytesIO()

        with zipfile.ZipFile(
            buffer,
            mode="w",
            compression=zipfile.ZIP_DEFLATED
        ) as archive:

            summary = []

            for result in results:

                if not isinstance(
                    result, SegmentationResult
                ):
                    raise TypeError("Invalid result object.")

                name = (
                    result.algorithm.lower()
                    .replace("-", "")
                    .replace(" ", "_")
                )

                mask_image = Image.fromarray(
                    result.mask.astype(np.uint8) * 255
                )

                archive.writestr(
                    f"{name}/mask.png",
                    self._image_to_bytes(mask_image)
                )

                archive.writestr(
                    f"{name}/overlay.png",
                    self._image_to_bytes(result.overlay)
                )

                archive.writestr(
                    f"{name}/boundary.png",
                    self._image_to_bytes(result.boundary)
                )

                archive.writestr(
                    f"{name}/analysis.json",
                    json.dumps(
                        result.to_dict(),
                        indent=4
                    )
                )

                summary.append(result.to_dict())

            archive.writestr(
                "comparison_report.json",
                json.dumps(summary, indent=4)
            )

            archive.writestr(
                "DISCLAIMER.txt",
                "SkinSegAI is an academic skin lesion "
                "segmentation system. Its results are "
                "not medical diagnoses."
            )

        return buffer.getvalue()
