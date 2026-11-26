"""Bounded image decoding and PNG serialization."""

from io import BytesIO

import numpy as np

from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000

SUPPORTED_FORMATS = {"PNG", "JPEG"}


def read_upload(upload):
    """Read a bounded number of bytes from an uploaded file."""

    data = upload.file.read(MAX_UPLOAD_BYTES + 1)

    if not data:
        raise ValueError("Uploaded file is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(
            "Uploaded file exceeds the 20 MiB limit."
        )

    return data


def decode_image(data, *, reference_mask=False):

    try:

        with Image.open(BytesIO(data)) as image:

            if image.format not in SUPPORTED_FORMATS:
                raise ValueError(
                    "Only PNG and JPEG images are supported."
                )

            width, height = image.size

            if (
                width <= 0
                or height <= 0
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise ValueError(
                    "Image exceeds the permitted dimensions."
                )

            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(
                    "Animated images are not supported."
                )

            if reference_mask:

                if image.format != "PNG":
                    raise ValueError(
                        "Reference masks must be lossless PNG files."
                    )

                if image.mode not in (
                    "1", "L", "I", "I;16",
                    "I;16L", "I;16B",
                ):
                    raise ValueError(
                        "Reference masks must be single-channel."
                    )

                image.load()

                # SessionContext performs binary-value
                # and exact dimension validation.
                return np.asarray(image).copy()

            image.load()

            return image.convert("RGB").copy()

    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
    ) as exc:

        raise ValueError(
            "Unable to decode the uploaded image safely."
        ) from exc


def encode_png(image):

    buffer = BytesIO()

    image.save(buffer, format="PNG")

    return buffer.getvalue()


def mask_to_image(mask):

    return Image.fromarray(
        mask.astype(np.uint8) * 255
    )


def create_error_map(prediction, reference):

    rgb = np.zeros(
        (*prediction.shape, 3),
        dtype=np.uint8,
    )

    rgb[prediction & reference] = [16, 185, 129]
    rgb[prediction & ~reference] = [239, 68, 68]
    rgb[~prediction & reference] = [59, 130, 246]

    return Image.fromarray(rgb)
