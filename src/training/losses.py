
import torch
import torch.nn as nn


class DiceLoss(nn.Module):

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
