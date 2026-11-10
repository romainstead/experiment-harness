import os
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset
import mlflow
import csv

CSV_FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time"]


def seed_everything(seed=0, num_threads=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)


def configure_mlp(n_layers: int, hidden: int, dropout: float):
    layers = []
    in_features = 64
    for _ in range(n_layers):
        layers.extend([
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        ])
        in_features = hidden
    layers.append(torch.nn.Linear(in_features, 10))
    return torch.nn.Sequential(*layers)


def train_model(model, num_epochs: int, lr: float, train_loader, val_loader, weight_decay, csv_file,
                device="cpu"):
    model = model.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        train_correct = 0
        train_total = 0
        train_loss = 0
        model.train()
        for digits, labels in train_loader:
            digits = digits.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(digits)
            _loss = loss(outputs, labels)
            _loss.backward()
            optimizer.step()
            train_loss += _loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        train_loss /= train_total

        with torch.no_grad():
            val_correct = 0
            val_total = 0
            model.eval()
            val_loss = 0
            for digits, labels in val_loader:
                digits = digits.to(device)
                labels = labels.to(device)
                outputs = model(digits)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss(outputs, labels).item() * labels.size(0)
            val_loss /= val_total

            epoch_end = time.perf_counter()
            epoch_time = epoch_end - epoch_start
            metrics = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_correct / train_total, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_correct / val_total, 4),
                "epoch_time": round(epoch_time, 4),
            }
            if mlflow.active_run() is not None:
                mlflow.log_metrics(metrics, step=epoch + 1)

            print(
                f'Epoch {epoch + 1}/{num_epochs}\n'
                f'train loss={train_loss:.4f}\n'
                f'val loss={val_loss:.4f}\n'
                f'train acc={train_correct / train_total:.4f}\n'
                f'val acc={val_correct / val_total:.4f}\n')

            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writerows([metrics])
    return model


def eval_model(model, test_loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for digits, labels in test_loader:
            digits = digits.to(device)
            labels = labels.to(device)

            outputs = model(digits)
            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


def write_predictions(model, loader, writer, split):
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0

    with torch.no_grad():
        for digits, labels, indices in loader:
            predicted = model(digits.to(device)).argmax(dim=1).cpu()
            writer.writerows(
                (split, index, y_true, y_pred)
                for index, y_true, y_pred in zip(
                    indices.tolist(), labels.tolist(), predicted.tolist()
                )
            )
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


def make_dataset(X, y, idxs, *, include_indices=False):
    Xt = torch.tensor(X[idxs] / 16.0, dtype=torch.float32)
    yt = torch.tensor(y[idxs], dtype=torch.long)
    if include_indices:
        indices = torch.as_tensor(idxs, dtype=torch.long)
        return TensorDataset(Xt, yt, indices)
    return TensorDataset(Xt, yt)


def run_correct_param_and_clean_git_check(config):
    if int(config.get("split_seed")) < 0:
        raise ValueError("split_seed должен быть >= 0")
    if float(config.get("train").get("weight_decay")) < 0:
        raise ValueError("weight_decay должен быть >= 0")

    positive_params = {
        "batch_size": int(config.get("train").get("batch_size")),
        "lr": float(config.get("train").get("lr")),
        "epochs": int(config.get("train").get("epochs")),
        "hidden": int(config.get("model").get("hidden")),
        "n_layers": int(config.get("model").get("n_layers")),
    }
    for name, value in positive_params.items():
        if value <= 0:
            raise SystemExit(f"{name} должен быть > 0, получено {value}")

    dirty = subprocess.run(["git", "status", "‐‐porcelain"], capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("Есть незакоммиченные изменения — итоговый запуск невозможен:\n" + dirty)
