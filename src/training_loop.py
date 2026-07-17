from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .early_stopping import EarlyStopping


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epoch_time_sec: list[float] = field(default_factory=list)


@dataclass
class TrainingResult:
    model: nn.Module
    history: TrainingHistory
    epochs_trained: int
    total_time_sec: float


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[Optimizer] = None,
    max_grad_norm: Optional[float] = None,
    desc: str = "",
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0

    with torch.set_grad_enabled(is_train):
        for inputs, target in tqdm(loader, desc=desc, leave=False):
            inputs, target = inputs.to(device), target.to(device)

            if is_train:
                optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, target)

            if is_train:
                loss.backward()
                # gradient clipping to prevent gradient exploding problem
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (output.argmax(dim=1) == target).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    scheduler: ReduceLROnPlateau,
    optimizer: Optimizer,
    epochs: int = 25,
    max_grad_norm: float = 0.1,
) -> TrainingResult:
    early_stop = EarlyStopping()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = TrainingHistory()
    start_time = time.time()

    epoch_bar = tqdm(range(epochs), desc="Epochs")
    for epoch in epoch_bar:
        epoch_start = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, max_grad_norm=max_grad_norm, desc="Train",
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, device, desc="Val",
        )

        scheduler.step(val_loss)
        curr_lr = optimizer.param_groups[0]["lr"]

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)
        history.lr.append(curr_lr)
        history.epoch_time_sec.append(time.time() - epoch_start)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
            lr=curr_lr,
        )

        early_stop(val_loss, model)
        if early_stop.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}/{epochs}.")
            break

    # Restore the weights that generated the lowest validation error,
    # whether early stopping fired or the epoch budget simply ran out.
    model.load_state_dict(early_stop.best_model_weights)

    return TrainingResult(
        model=model,
        history=history,
        epochs_trained=len(history.train_loss),
        total_time_sec=time.time() - start_time,
    )
