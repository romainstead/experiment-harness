import os
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset


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


def train_model(model, num_epochs: int, lr: float, train_loader, val_loader, weight_decay,
                device="cpu"):
    model = model.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
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

            print(
                f'Epoch {epoch + 1}/{num_epochs}\n'
                f'train loss={train_loss:.4f}\n'
                f'val loss={val_loss:.4f}\n'
                f'train acc={train_correct / train_total:.4f}\n'
                f'val acc={val_correct / val_total:.4f}\n')
    return model


def make_dataset(X, y, idxs):
    Xt = torch.tensor(X[idxs] / 16.0, dtype=torch.float32)
    yt = torch.tensor(y[idxs], dtype=torch.long)
    return TensorDataset(Xt, yt)
