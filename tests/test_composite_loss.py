
import pytest
import torch

from src.training.losses import (
    CompositeSegmentationLoss,
    BoundaryLoss,
    DiceLoss,
)


def test_composite_loss_returns_finite_scalar():
    """Verify CompositeSegmentationLoss returns a 0-dimensional finite scalar."""
    criterion = CompositeSegmentationLoss(
        dice_weight=0.4, bce_weight=0.3, boundary_weight=0.3
    )

    logits = torch.randn(2, 1, 64, 64)
    targets = (torch.rand(2, 1, 64, 64) > 0.5).float()

    loss = criterion(logits, targets)

    assert loss.dim() == 0, f"Expected scalar tensor, got dim {loss.dim()}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() >= 0.0, f"Loss should be non-negative, got {loss.item()}"


def test_composite_loss_supports_backward():
    """Verify that loss.backward() computes finite gradients for model logits."""
    criterion = CompositeSegmentationLoss(
        dice_weight=0.4, bce_weight=0.3, boundary_weight=0.3
    )

    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = (torch.rand(2, 1, 64, 64) > 0.5).float()

    loss = criterion(logits, targets)
    loss.backward()

    assert logits.grad is not None, "Logits gradient should not be None."
    assert logits.grad.shape == logits.shape
    assert torch.all(torch.isfinite(logits.grad)), "All gradients must be finite."
    assert not torch.all(logits.grad == 0), "Gradients should not be all zeros."


def test_exact_match_lower_loss_than_shifted_and_blurry():
    """Verify exact mask match produces lower loss than shifted or blurry masks."""
    criterion = CompositeSegmentationLoss(
        dice_weight=0.4, bce_weight=0.3, boundary_weight=0.3
    )

    # Ground truth: a 20x20 foreground square centered at [22:42, 22:42]
    targets = torch.zeros(1, 1, 64, 64)
    targets[:, :, 22:42, 22:42] = 1.0

    # 1. Exact match prediction: confident positive inside, negative outside
    exact_logits = torch.where(targets == 1.0, 10.0, -10.0)

    # 2. Shifted mask prediction: box shifted by 10 pixels to [32:52, 22:42]
    shifted_targets = torch.zeros(1, 1, 64, 64)
    shifted_targets[:, :, 32:52, 22:42] = 1.0
    shifted_logits = torch.where(shifted_targets == 1.0, 10.0, -10.0)

    # 3. Blurry / uncertain mask prediction: logits around 0 (p ~ 0.5)
    blurry_logits = torch.zeros(1, 1, 64, 64)

    # 4. Inverted mask prediction: wrong on every pixel
    inverted_logits = torch.where(targets == 1.0, -10.0, 10.0)

    with torch.no_grad():
        exact_loss = criterion(exact_logits, targets).item()
        shifted_loss = criterion(shifted_logits, targets).item()
        blurry_loss = criterion(blurry_logits, targets).item()
        inverted_loss = criterion(inverted_logits, targets).item()

    assert exact_loss < shifted_loss, (
        f"Exact match loss ({exact_loss:.4f}) must be < shifted loss ({shifted_loss:.4f})"
    )
    assert exact_loss < blurry_loss, (
        f"Exact match loss ({exact_loss:.4f}) must be < blurry loss ({blurry_loss:.4f})"
    )
    assert exact_loss < inverted_loss, (
        f"Exact match loss ({exact_loss:.4f}) must be < inverted loss ({inverted_loss:.4f})"
    )


def test_individual_loss_components_logging():
    """Verify all individual loss components are correctly calculated and logged."""
    criterion = CompositeSegmentationLoss(
        dice_weight=0.4, bce_weight=0.3, boundary_weight=0.3
    )

    logits = torch.randn(2, 1, 64, 64)
    targets = (torch.rand(2, 1, 64, 64) > 0.5).float()

    total_loss, components = criterion(logits, targets, return_components=True)

    expected_keys = {"total_loss", "dice_loss", "bce_loss", "boundary_loss"}
    assert expected_keys.issubset(components.keys())

    for key, val in components.items():
        assert torch.isfinite(val), f"Component {key} is not finite: {val}"
        assert val.item() >= 0.0, f"Component {key} should be non-negative: {val.item()}"

    computed_total = (
        0.4 * components["dice_loss"]
        + 0.3 * components["bce_loss"]
        + 0.3 * components["boundary_loss"]
    )
    assert torch.allclose(total_loss, computed_total, atol=1e-5)


def test_boundary_loss_gradient_and_accuracy():
    """Verify BoundaryLoss independently handles boundary extraction and gradients."""
    boundary_loss = BoundaryLoss()

    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.zeros(2, 1, 32, 32)
    targets[:, :, 10:22, 10:22] = 1.0

    loss = boundary_loss(logits, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_handles_3d_target_tensors():
    """Verify criterion cleanly accepts (B, H, W) targets by automatically unsqueezing."""
    criterion = CompositeSegmentationLoss()

    logits = torch.randn(2, 1, 32, 32)
    targets_3d = (torch.rand(2, 32, 32) > 0.5).float()

    loss = criterion(logits, targets_3d)
    assert torch.isfinite(loss)


def test_invalid_weights_raise_error():
    """Verify constructor rejects negative weights or all-zero weights."""
    with pytest.raises(ValueError, match="non-negative"):
        CompositeSegmentationLoss(dice_weight=-0.1)

    with pytest.raises(ValueError, match="positive"):
        CompositeSegmentationLoss(dice_weight=0.0, bce_weight=0.0, boundary_weight=0.0)
