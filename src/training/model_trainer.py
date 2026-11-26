
import csv
import math
import torch

from pathlib import Path
from tqdm import tqdm

from src.training.losses import CompositeSegmentationLoss
from src.evaluation.metrics import MetricsEvaluator


class ModelTrainer:

    def __init__(self, model, device, output_dir,
                 learning_rate=0.001, patience=7,
                 arch="custom_unet",
                 criterion=None,
                 dice_weight=0.4,
                 bce_weight=0.3,
                 boundary_weight=0.3):

        self._device = device
        self._model = model.to(device)

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(
            parents=True, exist_ok=True
        )

        if criterion is not None:
            self._criterion = criterion
        else:
            self._criterion = CompositeSegmentationLoss(
                dice_weight=dice_weight,
                bce_weight=bce_weight,
                boundary_weight=boundary_weight
            )

        self._evaluator = MetricsEvaluator()

        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        self._scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer,
                mode="max",
                factor=0.5,
                patience=3
            )
        )

        self._patience = patience
        self._best_dice = -1.0
        self._history = []
        self._arch = arch

    def _run_epoch(self, loader, training):

        self._model.train(training)

        totals = {
            "loss": 0.0,
            "total_loss": 0.0,
            "dice_loss": 0.0,
            "bce_loss": 0.0,
            "boundary_loss": 0.0,
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }

        samples = 0

        progress = tqdm(
            loader,
            desc="Training" if training else "Validation"
        )

        with torch.set_grad_enabled(training):

            for images, masks in progress:

                images = images.to(self._device)
                masks = masks.to(self._device)

                if training:
                    self._optimizer.zero_grad(
                        set_to_none=True
                    )

                logits = self._model(images)

                if hasattr(self._criterion, "compute_components"):
                    loss, loss_components = self._criterion(
                        logits, masks, return_components=True
                    )
                else:
                    loss = self._criterion(logits, masks)
                    loss_components = {"total_loss": loss}

                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    raise RuntimeError("Non-finite loss.")

                if training:
                    loss.backward()
                    self._optimizer.step()

                metrics = self._evaluator.evaluate(
                    logits.detach(), masks
                )

                batch_size = images.size(0)
                samples += batch_size

                totals["loss"] += loss_value * batch_size

                for comp_name, comp_tensor in loss_components.items():
                    val = comp_tensor.item() if hasattr(comp_tensor, "item") else float(comp_tensor)
                    if comp_name not in totals:
                        totals[comp_name] = 0.0
                    totals[comp_name] += val * batch_size

                # Ensure total_loss is synchronized with loss
                if "total_loss" not in loss_components:
                    totals["total_loss"] += loss_value * batch_size

                for name, value in metrics.items():
                    totals[name] += value * batch_size

                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    dice=f"{metrics['dice']:.4f}"
                )

        if samples == 0:
            raise ValueError("Data loader is empty.")

        return {
            name: value / samples
            for name, value in totals.items()
        }

    def _save_checkpoint(self, epoch, validation):

        checkpoint_data = {
            "epoch": epoch,
            "arch": self._arch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict":
                self._optimizer.state_dict(),
            "scheduler_state_dict":
                self._scheduler.state_dict(),
            "validation_metrics": validation,
            "image_size": (256, 256),
            "threshold": 0.5
        }

        if hasattr(self._criterion, "dice_weight"):
            checkpoint_data["loss_weights"] = {
                "dice_weight": self._criterion.dice_weight,
                "bce_weight": self._criterion.bce_weight,
                "boundary_weight": self._criterion.boundary_weight,
            }

        torch.save(checkpoint_data, self._output_dir / "best_model.pth")

    def _save_history(self):

        path = self._output_dir / "training_history.csv"

        with open(path, "w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(self._history[0].keys())
            )

            writer.writeheader()
            writer.writerows(self._history)

    def fit(self, train_loader, val_loader, epochs=30):

        stale_epochs = 0

        for epoch in range(1, epochs + 1):

            print(f"\n===== Epoch {epoch}/{epochs} =====")

            train = self._run_epoch(
                train_loader, training=True
            )

            validation = self._run_epoch(
                val_loader, training=False
            )

            self._scheduler.step(validation["dice"])

            record = {
                "epoch": epoch,
                "learning_rate":
                    self._optimizer.param_groups[0]["lr"]
            }

            for name, value in train.items():
                record[f"train_{name}"] = value

            for name, value in validation.items():
                record[f"val_{name}"] = value

            self._history.append(record)
            self._save_history()

            dice_str = f"Dice: {train.get('dice_loss', 0.0):.4f}"
            bce_str = f"BCE: {train.get('bce_loss', 0.0):.4f}"
            bnd_str = f"Boundary: {train.get('boundary_loss', 0.0):.4f}"
            print(f"Train loss: {train['loss']:.4f} ({dice_str}, {bce_str}, {bnd_str})")

            val_dice_str = f"Dice: {validation.get('dice_loss', 0.0):.4f}"
            val_bce_str = f"BCE: {validation.get('bce_loss', 0.0):.4f}"
            val_bnd_str = f"Boundary: {validation.get('boundary_loss', 0.0):.4f}"
            print(f"Validation loss: {validation['loss']:.4f} ({val_dice_str}, {val_bce_str}, {val_bnd_str})")

            print(f"Validation Dice: {validation['dice']:.4f}")
            print(f"Validation IoU: {validation['iou']:.4f}")

            if validation["dice"] > self._best_dice:

                self._best_dice = validation["dice"]
                stale_epochs = 0

                self._save_checkpoint(epoch, validation)

                print("Best model checkpoint saved.")

            else:
                stale_epochs += 1

            if stale_epochs >= self._patience:
                print("Early stopping triggered.")
                break

        print("\nTraining completed.")
        print(f"Best validation Dice: {self._best_dice:.4f}")

        return self._history.copy()
