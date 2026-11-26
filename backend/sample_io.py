"""Validated loading of local dataset samples."""

import re

from backend import services

from backend.image_io import (
    MAX_UPLOAD_BYTES,
    decode_image,
    decode_reference,
)


def read_sample_file(path):

    with path.open("rb") as file:

        data = file.read(MAX_UPLOAD_BYTES + 1)

    if len(data) > MAX_UPLOAD_BYTES:

        raise ValueError(
            "Sample file exceeds the permitted size."
        )

    return data


def load_sample(name, *, include_reference=True):
    """Load a sample and its optional binary reference mask."""

    if (
        not isinstance(name, str)
        or re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}",
            name,
        ) is None
    ):

        raise ValueError("Invalid sample identifier.")

    image_path = services.SAMPLES_DIR / f"{name}.png"

    if not image_path.is_file():

        raise FileNotFoundError(
            "Requested sample does not exist."
        )

    image = decode_image(
        read_sample_file(image_path)
    )

    reference = None

    if include_reference:

        reference_path = (
            services.SAMPLES_MASK_DIR
            / f"{name}_segmentation.png"
        )

        if reference_path.is_file():

            reference = decode_reference(
                read_sample_file(reference_path)
            )

            if reference.shape != (
                image.height,
                image.width,
            ):

                raise ValueError(
                    "Sample reference dimensions "
                    "do not match the image."
                )

    return image, reference
