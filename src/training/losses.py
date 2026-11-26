
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self._smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.flatten(1)
        targets = targets.flatten(1)

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        dice = (
            2 * intersection + self._smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self._smooth
        )

        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss for backward compatibility."""

    def __init__(self, bce_weight=0.5):
        super().__init__()

        if not 0 <= bce_weight <= 1:
            raise ValueError("Invalid BCE weight.")

        self._bce_weight = bce_weight
        self._bce = nn.BCEWithLogitsLoss()
        self._dice = DiceLoss()

    def forward(self, logits, targets):

        bce = self._bce(logits, targets)
        dice = self._dice(logits, targets)

        return (
            self._bce_weight * bce
            + (1 - self._bce_weight) * dice
        )


class BoundaryLoss(nn.Module):
    """Differentiable morphological boundary-aware loss.

    Extracts soft boundary contours via morphological gradient
    (dilation - erosion) using native PyTorch max-pooling, then computes
    boundary overlap loss. This heavily penalizes imprecise or blurry
    lesion borders without external dependencies.
    """

    def __init__(self, kernel_size=3, smooth=1e-6):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        self._kernel_size = kernel_size
        self._padding = kernel_size // 2
        self._smooth = smooth

    def _extract_boundary(self, x):
        """Extract soft boundary contour using morphological gradient."""
        dilation = F.max_pool2d(
            x,
            kernel_size=self._kernel_size,
            stride=1,
            padding=self._padding
        )
        erosion = -F.max_pool2d(
            -x,
            kernel_size=self._kernel_size,
            stride=1,
            padding=self._padding
        )
        return dilation - erosion

    def forward(self, logits, targets):

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)

        probabilities = torch.sigmoid(logits)

        pred_boundary = self._extract_boundary(probabilities)
        target_boundary = self._extract_boundary(targets.float())

        pred_boundary_flat = pred_boundary.flatten(1)
        target_boundary_flat = target_boundary.flatten(1)

        intersection = (
            pred_boundary_flat * target_boundary_flat
        ).sum(dim=1)

        boundary_dice = (
            2.0 * intersection + self._smooth
        ) / (
            pred_boundary_flat.sum(dim=1)
            + target_boundary_flat.sum(dim=1)
            + self._smooth
        )

        return 1.0 - boundary_dice.mean()


class CompositeSegmentationLoss(nn.Module):
    """Composite segmentation loss combining region, pixel, and boundary terms.

    Combines:
    - Soft Dice Loss (region overlap)
    - BCE with Logits Loss (calibrated pixel-wise classification)
    - Boundary-Aware Loss (edge sharpening via morphological gradient)

    Parameters
    ----------
    dice_weight : float, default=0.4
        Relative weight for Soft Dice loss.
    bce_weight : float, default=0.3
        Relative weight for Binary Cross-Entropy loss.
    boundary_weight : float, default=0.3
        Relative weight for Boundary-Aware loss.
    smooth : float, default=1e-6
        Laplace smoothing factor for Dice calculations.
    """

    def __init__(
        self,
        dice_weight=0.4,
        bce_weight=0.3,
        boundary_weight=0.3,
        smooth=1e-6,
    ):
        super().__init__()

        if dice_weight < 0 or bce_weight < 0 or boundary_weight < 0:
            raise ValueError("Loss component weights must be non-negative.")

        total = dice_weight + bce_weight + boundary_weight
        if total == 0:
            raise ValueError("At least one loss component weight must be positive.")

        self._dice_weight = dice_weight
        self._bce_weight = bce_weight
        self._boundary_weight = boundary_weight

        self._dice = DiceLoss(smooth=smooth)
        self._bce = nn.BCEWithLogitsLoss()
        self._boundary = BoundaryLoss(smooth=smooth)

    @property
    def dice_weight(self):
        return self._dice_weight

    @property
    def bce_weight(self):
        return self._bce_weight

    @property
    def boundary_weight(self):
        return self._boundary_weight

    def forward(self, logits, targets, return_components=False):
        """Compute the weighted composite loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model logits of shape (B, 1, H, W) or (B, H, W).
        targets : torch.Tensor
            Ground truth binary masks of shape (B, 1, H, W) or (B, H, W).
        return_components : bool, default=False
            If True, returns (total_loss, components_dict).
            If False, returns total_loss tensor.

        Returns
        -------
        torch.Tensor or tuple(torch.Tensor, dict)
            Weighted total loss, optionally with dictionary of loss components.
        """
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)

        dice_loss = self._dice(logits, targets)
        bce_loss = self._bce(logits, targets)
        boundary_loss = self._boundary(logits, targets)

        total_loss = (
            self._dice_weight * dice_loss
            + self._bce_weight * bce_loss
            + self._boundary_weight * boundary_loss
        )

        components = {
            "total_loss": total_loss,
            "dice_loss": dice_loss,
            "bce_loss": bce_loss,
            "boundary_loss": boundary_loss,
        }

        if return_components:
            return total_loss, components

        return total_loss

    def compute_components(self, logits, targets):
        """Compute and return dictionary of individual loss components."""
        _, components = self.forward(logits, targets, return_components=True)
        return components
