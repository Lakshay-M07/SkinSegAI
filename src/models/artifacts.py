
"""Centralized model artifact management for SkinSegAI."""
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from dataclasses import dataclass

from src.models.model_factory import SUPPORTED_ARCHITECTURES


@dataclass(frozen=True)
class ModelArtifacts:
    """Immutable collection of paths for a model."""

    architecture: str
    checkpoint: Path
    training_history: Path
    comparison_summary: Path
    per_image_results: Path

    @property
    def is_trained(self) -> bool:
        return (
            self.checkpoint.is_file()
            and self.checkpoint.stat().st_size > 0
        )

    @property
    def has_training_history(self) -> bool:
        return self.training_history.is_file()

    @property
    def has_benchmark(self) -> bool:
        return (
            self.comparison_summary.is_file()
            and self.per_image_results.is_file()
        )


class ModelArtifactRegistry:
    """Resolve and manage architecture-specific artifacts."""

    _DISPLAY_NAMES = {
        "custom_unet": "Custom U-Net",
        "resnet34_unet": "ResNet-34 U-Net",
        "efficientnet_b2_unet": "EfficientNet-B2 U-Net",
    }

    def __init__(self, project_root=None):

        self._root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )

        self._checkpoint_root = (
            self._root / "artifacts" / "checkpoints"
        )

        self._metrics_root = (
            self._root / "artifacts" / "metrics"
        )

    @staticmethod
    def _validate_architecture(architecture: str) -> None:

        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Supported: {SUPPORTED_ARCHITECTURES}"
            )

    def get(self, architecture: str) -> ModelArtifacts:
        """Resolve artifact paths for a supported architecture."""

        self._validate_architecture(architecture)

        checkpoint_dir = (
            self._checkpoint_root / architecture
        )

        metrics_dir = (
            self._metrics_root / architecture
        )

        return ModelArtifacts(
            architecture=architecture,
            checkpoint=checkpoint_dir / "best_model.pth",
            training_history=(
                checkpoint_dir / "training_history.csv"
            ),
            comparison_summary=(
                metrics_dir / "comparison_summary.csv"
            ),
            per_image_results=(
                metrics_dir / "per_image_results.csv"
            ),
        )

    def trained_models(self) -> list[str]:
        """Return architectures with available checkpoints."""

        return [
            arch
            for arch in SUPPORTED_ARCHITECTURES
            if self.get(arch).is_trained
        ]

    def benchmarked_models(self) -> list[str]:
        """Return architectures with benchmark artifacts."""

        return [
            arch
            for arch in SUPPORTED_ARCHITECTURES
            if self.get(arch).has_benchmark
        ]

    def display_name(self, architecture: str) -> str:

        self._validate_architecture(architecture)

        return self._DISPLAY_NAMES[architecture]

    def require_checkpoint(self, architecture: str) -> Path:
        """Return an available checkpoint or raise an error."""

        artifacts = self.get(architecture)

        if not artifacts.is_trained:
            raise FileNotFoundError(
                f"No trained checkpoint for '{architecture}'. "
                f"Expected: {artifacts.checkpoint}"
            )

        return artifacts.checkpoint

    
    def create_run_directory(
        self,
        architecture: str,
        *,
        kind: str,
        smoke: bool = False,
    ) -> Path:
        """Create a unique directory for a new experiment."""

        self._validate_architecture(architecture)

        if kind not in ("training", "evaluation"):
            raise ValueError(
                f"Invalid experiment kind: {kind}"
            )

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        )

        unique_id = uuid4().hex[:8]

        prefix = "smoke" if smoke else "run"

        run_name = (
            f"{prefix}_{timestamp}_{unique_id}"
        )

        output_dir = (
            self._root
            / "artifacts"
            / "runs"
            / kind
            / architecture
            / run_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        return output_dir
