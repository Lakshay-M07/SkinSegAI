"""
Lightweight unit tests that verify forward-pass output shapes
for every supported architecture in SkinSegAI.
"""

import pytest
import torch

from src.models.model_factory import (
    build_model,
    SUPPORTED_ARCHITECTURES,
)


@pytest.mark.parametrize("arch", SUPPORTED_ARCHITECTURES)
def test_forward_pass_output_shape(arch):
    """Each architecture must produce (B, 1, 256, 256) logits."""

    model = build_model(arch)
    model.eval()

    batch = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        output = model(batch)

    assert output.shape == (2, 1, 256, 256), (
        f"{arch}: expected (2, 1, 256, 256), got {output.shape}"
    )


def test_unknown_architecture_raises():
    """build_model must reject unrecognised names."""

    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("does_not_exist")
