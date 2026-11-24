
import torch


class MetricsEvaluator:

    def __init__(self, threshold=0.5):
        self._threshold = threshold

    @staticmethod
    def _divide(numerator, denominator, empty=0.0):

        return torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1),
            torch.full_like(denominator, empty)
        )

    @torch.no_grad()
    def evaluate_masks(self, predictions, targets):

        predictions = predictions.bool().flatten(1)
        targets = targets.bool().flatten(1)

        tp = (predictions & targets).sum(dim=1).float()
        fp = (predictions & ~targets).sum(dim=1).float()
        fn = (~predictions & targets).sum(dim=1).float()

        dice = self._divide(
            2 * tp, 2 * tp + fp + fn, empty=1.0
        )

        iou = self._divide(
            tp, tp + fp + fn, empty=1.0
        )

        precision = self._divide(tp, tp + fp)
        recall = self._divide(tp, tp + fn)

        return {
            "dice": dice.mean().item(),
            "iou": iou.mean().item(),
            "precision": precision.mean().item(),
            "recall": recall.mean().item()
        }

    @torch.no_grad()
    def evaluate(self, logits, targets):

        predictions = (
            torch.sigmoid(logits) >= self._threshold
        )

        return self.evaluate_masks(
            predictions, targets
        )
