
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.core.base_segmenter import BaseSegmenter
from src.models.model_factory import (
    build_model,
    uses_imagenet_normalization,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


_DROPOUT_TYPES = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)


def compute_binary_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute binary predictive entropy H(p) normalized to [0.0, 1.0].

    Peaks at 1.0 when p = 0.5 (maximum decision-boundary ambiguity) and
    approaches 0.0 when p -> 0 or p -> 1.
    """
    p = np.clip(probs.astype(np.float32), 1e-7, 1.0 - 1e-7)
    entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return np.clip(entropy, 0.0, 1.0).astype(np.float32)


class UNetSegmenter(BaseSegmenter):

    def __init__(self, checkpoint_path,
                 device=None, threshold=0.5):

        super().__init__()

        if not 0 < threshold < 1:
            raise ValueError("Invalid prediction threshold.")

        self._threshold = threshold

        self._device = torch.device(
            device or (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        self._image_size = tuple(
            checkpoint.get("image_size", (256, 256))
        )

        # Resolve architecture from checkpoint metadata;
        # fall back to custom_unet for legacy checkpoints.
        self._arch = checkpoint.get("arch", "custom_unet")
        self._model = build_model(
            self._arch,
            pretrained=False,
        )

        self._model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self._model.to(self._device)
        self._model.eval()

        # Cache normalization stats for inference
        self._uses_imagenet = uses_imagenet_normalization(
            self._arch
        )


    @classmethod
    def from_architecture(
        cls,
        architecture: str,
        *,
        device=None,
        threshold: float = 0.5,
        project_root=None,
    ):
        """Load a trained model using its architecture name."""

        from src.models.artifacts import ModelArtifactRegistry

        registry = ModelArtifactRegistry(project_root)

        checkpoint = registry.require_checkpoint(
            architecture
        )

        segmenter = cls(
            checkpoint,
            device=device,
            threshold=threshold,
        )

        if segmenter.arch != architecture:
            raise ValueError(
                f"Checkpoint architecture mismatch. "
                f"Expected '{architecture}', "
                f"found '{segmenter.arch}'."
            )

        return segmenter


    @property
    def name(self):
        return "U-Net"

    @property
    def arch(self):
        return self._arch

    @property
    def has_dropout(self):
        return any(
            isinstance(module, _DROPOUT_TYPES)
            for module in self._model.modules()
        )

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):

        if not 0 < value < 1:
            raise ValueError("Invalid prediction threshold.")

        self._threshold = value

    def _prepare_tensor(self, image):

        original_size = image.size
        prepared = self._prepare_image(image)

        array = np.asarray(
            prepared, dtype=np.float32
        ) / 255.0

        if self._uses_imagenet:
            mean = np.array(IMAGENET_MEAN, dtype=np.float32)
            std = np.array(IMAGENET_STD, dtype=np.float32)
            array = (array - mean) / std

        tensor = torch.from_numpy(
            np.ascontiguousarray(
                array.transpose(2, 0, 1)
            )
        ).unsqueeze(0).to(self._device)

        return tensor, original_size

    @staticmethod
    def _restore_float_map(array, original_size):

        restored = Image.fromarray(
            array.astype(np.float32), mode="F"
        ).resize(
            original_size,
            Image.Resampling.BILINEAR
        )
        return np.clip(
            np.asarray(restored, dtype=np.float32),
            0.0,
            1.0
        )

    def _enable_mc_dropout(self):
        """Keep BatchNorm in eval mode while enabling Dropout layers."""
        self._model.eval()
        for module in self._model.modules():
            if isinstance(module, _DROPOUT_TYPES):
                module.train()

    @torch.inference_mode()
    def predict_with_uncertainty(self, image, mc_passes=1):
        """Run inference and return probability, binary mask, and uncertainty.

        Returns
        -------
        dict
            Keys: ``prob_map`` (float32 [0, 1]), ``binary_mask`` (bool),
            ``uncertainty_map`` (float32 [0, 1]).
        """
        if mc_passes < 1:
            raise ValueError("mc_passes must be at least 1.")

        tensor, original_size = self._prepare_tensor(image)

        if mc_passes > 1 and self.has_dropout:
            try:
                self._enable_mc_dropout()
                pass_probs = []
                for _ in range(mc_passes):
                    logits = self._model(tensor)
                    pass_probs.append(
                        torch.sigmoid(logits)[0, 0]
                    )
                probs_2d = (
                    torch.stack(pass_probs, dim=0)
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                )
            finally:
                self._model.eval()
        else:
            self._model.eval()
            logits = self._model(tensor)
            probs_2d = (
                torch.sigmoid(logits)[0, 0]
                .cpu()
                .numpy()
            )

        prob_map = self._restore_float_map(
            probs_2d, original_size
        )
        uncertainty_map = compute_binary_entropy(prob_map)
        binary_mask = prob_map >= self._threshold

        return {
            "prob_map": prob_map,
            "binary_mask": binary_mask,
            "uncertainty_map": uncertainty_map,
        }

    @torch.inference_mode()
    def segment(self, image):
        return self.predict_with_uncertainty(
            image, mc_passes=1
        )["binary_mask"]
