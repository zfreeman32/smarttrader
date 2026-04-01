from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from model_training.ote_training.torch_models import build_torch_model


@dataclass
class TorchTrainingResult:
    model: nn.Module
    val_probabilities: np.ndarray
    training_history: List[Dict[str, Any]]
    checkpoint: Optional[Dict[str, Any]] = None


class SequenceDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> None:
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        self.w = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index], self.w[index]


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: float) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = (probs * targets) + ((1.0 - probs) * (1.0 - targets))
        alpha_t = (self.alpha * targets) + ((1.0 - self.alpha) * (1.0 - targets))
        loss = alpha_t * torch.pow(1.0 - p_t, self.gamma) * bce
        if sample_weight is not None:
            loss = loss * sample_weight.float()
        return loss.mean()


def safe_average_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    if not np.isfinite(y_score).all():
        raise FloatingPointError("Average precision received non-finite prediction scores.")
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.0
    return float(average_precision_score(y_true, y_score, sample_weight=sample_weight))


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def resolve_amp_usage(
    model_type: str,
    requested_use_amp: bool,
    device: torch.device,
) -> bool:
    if not requested_use_amp or device.type != "cuda":
        return False
    # TCN training on CUDA AMP has produced non-finite logits during full-data
    # refits, so keep this path in full precision by default.
    if model_type == "tcn":
        return False
    return True


def build_phase_schedule(
    total_epochs: int,
    warmup_epochs: Optional[int] = None,
    main_epochs: Optional[int] = None,
    fine_epochs: Optional[int] = None,
    tail_epochs: int = 0,
    fine_lr_scale: float = 0.35,
    tail_lr_scale: float = 0.10,
) -> List[Dict[str, float]]:
    explicit_schedule = any(value is not None for value in (warmup_epochs, main_epochs, fine_epochs)) or tail_epochs > 0
    if explicit_schedule:
        if warmup_epochs is None or main_epochs is None or fine_epochs is None:
            raise ValueError("Explicit phase scheduling requires warmup, main, and fine epoch counts.")
        phase_totals = {
            "warmup": int(warmup_epochs),
            "main": int(main_epochs),
            "fine": int(fine_epochs),
            "tail": int(tail_epochs),
        }
        if any(value < 0 for value in phase_totals.values()):
            raise ValueError(f"Phase epoch counts cannot be negative: {phase_totals}")
        if sum(phase_totals.values()) != int(total_epochs):
            raise ValueError(
                "Explicit phase schedule must sum to total epochs. "
                f"Got {phase_totals} for total_epochs={total_epochs}."
            )
        schedule: List[Dict[str, float]] = []
        if phase_totals["warmup"] > 0:
            schedule.append({"name": "warmup", "epochs": float(phase_totals["warmup"]), "train_fraction": 0.35, "lr_scale": 1.0})
        if phase_totals["main"] > 0:
            schedule.append({"name": "main", "epochs": float(phase_totals["main"]), "train_fraction": 1.0, "lr_scale": 1.0})
        if phase_totals["fine"] > 0:
            schedule.append(
                {"name": "fine", "epochs": float(phase_totals["fine"]), "train_fraction": 1.0, "lr_scale": float(fine_lr_scale)}
            )
        if phase_totals["tail"] > 0:
            schedule.append(
                {"name": "tail", "epochs": float(phase_totals["tail"]), "train_fraction": 1.0, "lr_scale": float(tail_lr_scale)}
            )
        if not schedule:
            raise ValueError("Explicit phase schedule produced no training phases.")
        return schedule

    if total_epochs <= 2:
        return [{"name": "main", "epochs": float(max(total_epochs, 1)), "train_fraction": 1.0, "lr_scale": 1.0}]

    warmup_epochs = max(1, total_epochs // 5)
    fine_epochs = max(1, total_epochs // 5)
    main_epochs = max(1, total_epochs - warmup_epochs - fine_epochs)

    while (warmup_epochs + main_epochs + fine_epochs) > total_epochs and fine_epochs > 0:
        fine_epochs -= 1
    while (warmup_epochs + main_epochs + fine_epochs) < total_epochs:
        main_epochs += 1

    schedule = [
        {"name": "warmup", "epochs": float(warmup_epochs), "train_fraction": 0.35, "lr_scale": 1.0},
        {"name": "main", "epochs": float(main_epochs), "train_fraction": 1.0, "lr_scale": 1.0},
    ]
    if fine_epochs > 0:
        schedule.append({"name": "fine", "epochs": float(fine_epochs), "train_fraction": 1.0, "lr_scale": float(fine_lr_scale)})
    return schedule


def create_loader(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = SequenceDataset(X=X, y=y, w=w)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )


def evaluate_torch_model(
    model: nn.Module,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    w_eval: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> Tuple[np.ndarray, float]:
    probabilities = predict_torch_model(
        model=model,
        X=X_eval,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
    )
    score = safe_average_precision(y_eval, probabilities, sample_weight=w_eval)
    return probabilities, score


def train_torch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    w_eval: np.ndarray,
    trainer_config: Mapping[str, Any],
) -> TorchTrainingResult:
    torch.manual_seed(int(trainer_config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(trainer_config["seed"]))

    device = choose_device()
    model_type = str(trainer_config["model_type"])
    use_amp = resolve_amp_usage(
        model_type=model_type,
        requested_use_amp=bool(trainer_config["use_amp"]),
        device=device,
    )
    pin_memory = device.type == "cuda"
    batch_size = int(trainer_config["batch_size"])
    learning_rate = float(trainer_config["learning_rate"])
    weight_decay = float(trainer_config["weight_decay"])
    gradient_clip = float(trainer_config["gradient_clip"])

    model = build_torch_model(
        model_type=model_type,
        input_size=int(trainer_config["input_size"]),
        hidden_size=int(trainer_config["hidden_size"]),
        num_layers=int(trainer_config["num_layers"]),
        dropout=float(trainer_config["dropout"]),
    ).to(device)

    criterion = WeightedFocalLoss(
        alpha=float(trainer_config["focal_alpha"]),
        gamma=float(trainer_config["focal_gamma"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    schedule = build_phase_schedule(
        int(trainer_config["epochs"]),
        warmup_epochs=trainer_config.get("warmup_epochs"),
        main_epochs=trainer_config.get("main_epochs"),
        fine_epochs=trainer_config.get("fine_epochs"),
        tail_epochs=int(trainer_config.get("tail_epochs", 0) or 0),
        fine_lr_scale=float(trainer_config.get("fine_lr_scale", 0.35)),
        tail_lr_scale=float(trainer_config.get("tail_lr_scale", 0.10)),
    )
    patience = max(3, min(8, int(trainer_config["epochs"]) // 3 if int(trainer_config["epochs"]) > 3 else 3))

    history: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_probabilities = np.zeros(len(y_eval), dtype=np.float32)
    best_epoch = 0
    epochs_without_improvement = 0
    global_epoch = 0
    stop_reason: Optional[str] = None

    for phase in schedule:
        phase_name = str(phase["name"])
        phase_epochs = int(phase["epochs"])
        train_fraction = float(phase["train_fraction"])
        phase_lr = learning_rate * float(phase["lr_scale"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = phase_lr

        train_rows = max(batch_size, int(round(len(X_train) * train_fraction)))
        train_rows = min(train_rows, len(X_train))
        train_loader = create_loader(
            X=X_train[:train_rows],
            y=y_train[:train_rows],
            w=w_train[:train_rows],
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
        )

        for _ in range(phase_epochs):
            global_epoch += 1
            model.train()
            running_loss = 0.0
            sample_count = 0
            non_finite_batches = 0

            for features, targets, sample_weight in train_loader:
                features = features.to(device, non_blocking=pin_memory)
                targets = targets.to(device, non_blocking=pin_memory)
                sample_weight = sample_weight.to(device, non_blocking=pin_memory)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(features)
                    loss = criterion(logits, targets, sample_weight)

                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    non_finite_batches += 1
                    stop_reason = "non_finite_train_batch"
                    optimizer.zero_grad(set_to_none=True)
                    break

                scaler.scale(loss).backward()
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()

                batch_size_actual = int(targets.shape[0])
                running_loss += float(loss.detach().cpu()) * batch_size_actual
                sample_count += batch_size_actual

            train_loss = running_loss / max(sample_count, 1)
            if non_finite_batches > 0:
                history.append(
                    {
                        "epoch": float(global_epoch),
                        "phase": phase_name,
                        "train_rows": float(train_rows),
                        "train_loss": float("nan"),
                        "val_auprc": float("nan"),
                        "best_val_auprc": float(best_score) if np.isfinite(best_score) else float("nan"),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "improved": False,
                        "non_finite_batches": float(non_finite_batches),
                        "stop_reason": stop_reason,
                        "amp_enabled": bool(use_amp),
                    }
                )
                break

            try:
                val_probabilities, val_auprc = evaluate_torch_model(
                    model=model,
                    X_eval=X_eval,
                    y_eval=y_eval,
                    w_eval=w_eval,
                    batch_size=batch_size,
                    device=device,
                    use_amp=use_amp,
                )
            except FloatingPointError:
                stop_reason = "non_finite_eval_predictions"
                history.append(
                    {
                        "epoch": float(global_epoch),
                        "phase": phase_name,
                        "train_rows": float(train_rows),
                        "train_loss": float(train_loss),
                        "val_auprc": float("nan"),
                        "best_val_auprc": float(best_score) if np.isfinite(best_score) else float("nan"),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "improved": False,
                        "non_finite_batches": 0.0,
                        "stop_reason": stop_reason,
                        "amp_enabled": bool(use_amp),
                    }
                )
                break

            scheduler.step(val_auprc)

            improved = val_auprc > (best_score + 1e-5)
            if improved:
                best_score = val_auprc
                best_state = checkpoint_state_dict(model)
                best_probabilities = val_probabilities
                best_epoch = global_epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            history.append(
                {
                    "epoch": float(global_epoch),
                    "phase": phase_name,
                    "train_rows": float(train_rows),
                    "train_loss": float(train_loss),
                    "val_auprc": float(val_auprc),
                    "best_val_auprc": float(best_score),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "improved": bool(improved),
                    "non_finite_batches": 0.0,
                    "stop_reason": None,
                    "amp_enabled": bool(use_amp),
                }
            )

            if stop_reason is not None or epochs_without_improvement >= patience:
                break

        if stop_reason is not None or epochs_without_improvement >= patience:
            break

    if best_state is None and stop_reason is not None:
        raise RuntimeError(
            "Torch training diverged before producing a stable checkpoint "
            f"({stop_reason}). Retry with a lower learning rate or full precision."
        )

    if best_state is None:
        best_state = checkpoint_state_dict(model)
        best_probabilities, best_score = evaluate_torch_model(
            model=model,
            X_eval=X_eval,
            y_eval=y_eval,
            w_eval=w_eval,
            batch_size=batch_size,
            device=device,
            use_amp=use_amp,
        )
        best_epoch = global_epoch

    model.load_state_dict(best_state)
    model.eval()

    checkpoint = {
        "model_state_dict": best_state,
        "model_config": {
            "model_type": str(trainer_config["model_type"]),
            "input_size": int(trainer_config["input_size"]),
            "hidden_size": int(trainer_config["hidden_size"]),
            "num_layers": int(trainer_config["num_layers"]),
            "dropout": float(trainer_config["dropout"]),
        },
        "best_val_auprc": float(best_score),
        "best_epoch": int(best_epoch),
        "history": history,
        "stop_reason": stop_reason,
        "amp_enabled": bool(use_amp),
    }

    return TorchTrainingResult(
        model=model,
        val_probabilities=np.asarray(best_probabilities, dtype=np.float32),
        training_history=history,
        checkpoint=checkpoint,
    )


def load_torch_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: Optional[torch.device] = None,
) -> nn.Module:
    if device is None:
        device = choose_device()

    model_config = checkpoint["model_config"]
    model = build_torch_model(
        model_type=str(model_config["model_type"]),
        input_size=int(model_config["input_size"]),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_torch_model(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: Optional[torch.device] = None,
    use_amp: bool = False,
) -> np.ndarray:
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = choose_device()

    model.eval()
    pin_memory = device.type == "cuda"
    features = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(features),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )

    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for (batch_features,) in loader:
            batch_features = batch_features.to(device, non_blocking=pin_memory)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch_features)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Torch model produced non-finite logits during prediction.")
            probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(probabilities).all():
                raise FloatingPointError("Torch model produced non-finite probabilities during prediction.")
            predictions.append(probabilities)

    if not predictions:
        return np.empty(0, dtype=np.float32)
    joined = np.concatenate(predictions, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(joined).all():
        raise FloatingPointError("Torch model produced non-finite probabilities during prediction.")
    return joined
