
"""
Model factory for SkinSegAI.

Supports custom and pretrained-encoder U-Net architectures.
Separates training initialization from offline inference.
"""

import torch.nn as nn

from src.models.unet import UNet


SUPPORTED_ARCHITECTURES = (
    "custom_unet",
    "resnet34_unet",
    "efficientnet_b2_unet",
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def uses_imagenet_normalization(arch: str) -> bool:
    """Return whether the architecture uses ImageNet normalization."""

    return arch in (
        "resnet34_unet",
        "efficientnet_b2_unet",
    )


def build_model(
    arch: str = "resnet34_unet",
    *,
    pretrained: bool = True,
) -> nn.Module:
    """
    Construct a supported segmentation model.

    pretrained=True:
        Initialize encoder with ImageNet weights for training.

    pretrained=False:
        Construct architecture without downloading weights,
        allowing offline checkpoint loading.
    """

    if arch not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Supported: {SUPPORTED_ARCHITECTURES}"
        )

    if arch == "custom_unet":
        return UNet(
            in_channels=3,
            out_channels=1,
        )

    try:
        import segmentation_models_pytorch as smp

    except ImportError as exc:
        raise ImportError(
            f"Architecture '{arch}' requires "
            "segmentation-models-pytorch. "
            "Install the project requirements."
        ) from exc

    encoder_names = {
        "resnet34_unet": "resnet34",
        "efficientnet_b2_unet": "efficientnet-b2",
    }

    return smp.Unet(
        encoder_name=encoder_names[arch],
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
    )
