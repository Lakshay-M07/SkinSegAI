
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.analysis.lesion_analyzer import LesionAnalyzer
from src.models.model_factory import build_model
from src.segmentation.grabcut_segmenter import GrabCutSegmenter
from src.segmentation.unet_segmenter import (
    UNetSegmenter,
    compute_binary_entropy,
)
from src.services.segmentation_service import SegmentationService


def test_compute_binary_entropy_bounds():
    probs = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    entropy = compute_binary_entropy(probs)

    assert entropy.dtype == np.float32
    assert np.isclose(entropy[0, 1], 1.0, atol=1e-5)
    assert entropy[0, 0] < 1e-4
    assert entropy[0, 2] < 1e-4
    assert np.all((entropy >= 0.0) & (entropy <= 1.0))


def test_unet_segmenter_predict_with_uncertainty(tmp_path):
    model = build_model("custom_unet")
    ckpt_path = tmp_path / "custom.pth"
    torch.save(
        {
            "arch": "custom_unet",
            "model_state_dict": model.state_dict(),
            "image_size": (256, 256),
        },
        ckpt_path,
    )

    segmenter = UNetSegmenter(ckpt_path, device="cpu", threshold=0.5)
    assert segmenter.arch == "custom_unet"
    assert segmenter.has_dropout is False

    image = Image.new("RGB", (320, 240), color=(120, 90, 80))
    pred = segmenter.predict_with_uncertainty(image)

    assert set(pred.keys()) == {"prob_map", "binary_mask", "uncertainty_map"}
    assert pred["prob_map"].shape == (240, 320)
    assert pred["binary_mask"].shape == (240, 320)
    assert pred["uncertainty_map"].shape == (240, 320)
    assert pred["binary_mask"].dtype == np.bool_
    assert np.all((pred["prob_map"] >= 0.0) & (pred["prob_map"] <= 1.0))
    assert np.all(
        (pred["uncertainty_map"] >= 0.0) & (pred["uncertainty_map"] <= 1.0)
    )


def test_mc_dropout_keeps_batchnorm_eval_and_restores_eval(tmp_path):
    class DropoutUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(4)
            self.drop = nn.Dropout2d(p=0.5)
            self.out = nn.Conv2d(4, 1, kernel_size=1)

        def forward(self, x):
            assert self.bn.training is False, "BatchNorm must remain in eval mode"
            x = self.bn(self.conv(x))
            x = self.drop(x)
            return self.out(x)

    dummy_model = build_model("custom_unet")
    ckpt_path = tmp_path / "dummy.pth"
    torch.save(
        {
            "arch": "custom_unet",
            "model_state_dict": dummy_model.state_dict(),
            "image_size": (256, 256),
        },
        ckpt_path,
    )

    segmenter = UNetSegmenter(ckpt_path, device="cpu", threshold=0.5)
    segmenter._model = DropoutUNet()
    assert segmenter.has_dropout is True

    image = Image.new("RGB", (128, 128), color=(100, 100, 100))
    pred = segmenter.predict_with_uncertainty(image, mc_passes=4)

    assert pred["uncertainty_map"].shape == (128, 128)
    assert segmenter._model.training is False
    assert segmenter._model.drop.training is False


def test_lesion_analyzer_border_uncertainty_and_heatmap():
    analyzer = LesionAnalyzer()

    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True

    uncertainty = np.zeros((100, 100), dtype=np.float32)
    uncertainty[28:33, 28:72] = 0.9
    uncertainty[67:72, 28:72] = 0.9

    border_score = analyzer.compute_border_uncertainty(mask, uncertainty)
    assert 0.0 <= border_score <= 1.0
    assert border_score > 0.2

    image = Image.new("RGB", (100, 100), color=(200, 150, 150))
    heatmap, blended = analyzer.create_uncertainty_heatmap(image, uncertainty)
    assert heatmap.size == (100, 100)
    assert blended.size == (100, 100)


def test_segmentation_service_populates_uncertainty_for_unet_only(tmp_path):
    model = build_model("custom_unet")
    ckpt_path = tmp_path / "custom.pth"
    torch.save(
        {
            "arch": "custom_unet",
            "model_state_dict": model.state_dict(),
            "image_size": (256, 256),
        },
        ckpt_path,
    )

    unet = UNetSegmenter(ckpt_path, device="cpu")
    grabcut = GrabCutSegmenter(iterations=1)
    service = SegmentationService([unet, grabcut])

    image = Image.new("RGB", (128, 128), color=(120, 100, 90))
    unet_res = service.segment(image, "U-Net")
    grab_res = service.segment(image, "GrabCut")

    assert unet_res.prob_map is not None
    assert unet_res.uncertainty_map is not None
    if unet_res.mask.any() and not unet_res.mask.all():
        assert unet_res.border_uncertainty is not None
        assert 0.0 <= unet_res.border_uncertainty <= 1.0
    else:
        assert unet_res.border_uncertainty is None
    assert grab_res.prob_map is None
    assert grab_res.uncertainty_map is None
    assert grab_res.border_uncertainty is None
