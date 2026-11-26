"""
Model factory for SkinSegAI.

Provides a single entry point to construct any supported segmentation
architecture by name.  Keeps the original custom U-Net untouched and
adds pretrained-encoder variants via segmentation-models-pytorch (smp).
"""

import torch.nn as nn
import segmentation_models_pytorch as smp

from src.models.unet import UNet


# ------------------------------------------------------------------ #
#  Registry of supported architectures                                #
# ------------------------------------------------------------------ #

SUPPORTED_ARCHITECTURES = (
    "custom_unet",
    "resnet34_unet",
    "efficientnet_b2_unet",
)


# ------------------------------------------------------------------ #
#  ImageNet channel-wise statistics (used by pretrained encoders)      #
# ------------------------------------------------------------------ #

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def uses_imagenet_normalization(arch: str) -> bool:
    """Return True when *arch* expects ImageNet mean/std preprocessing."""
    return arch in ("resnet34_unet", "efficientnet_b2_unet")


# ------------------------------------------------------------------ #
#  Factory function                                                    #
# ------------------------------------------------------------------ #

def build_model(arch: str = "resnet34_unet") -> nn.Module:
    """Construct a segmentation model by architecture name.

    Parameters
    ----------
    arch : str
        One of :data:`SUPPORTED_ARCHITECTURES`.

    Returns
    -------
    nn.Module
        An uninitialised (or pretrained-encoder) segmentation model that
        maps ``(B, 3, H, W)`` → ``(B, 1, H, W)`` raw logits.

    Raises
    ------
    ValueError
        If *arch* is not recognised.
    """
    if arch == "custom_unet":
        return UNet(in_channels=3, out_channels=1)

    if arch == "resnet34_unet":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    if arch == "efficientnet_b2_unet":
        return smp.Unet(
            encoder_name="efficientnet-b2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    raise ValueError(
        f"Unknown architecture '{arch}'. "
        f"Supported: {SUPPORTED_ARCHITECTURES}"
    )
