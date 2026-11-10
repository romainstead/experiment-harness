import argparse
import hashlib
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import sklearn
import yaml
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import seed_everything, configure_mlp, train_model, make_dataset, eval_model, CSV_FIELDS
import uuid
import csv


def prepare_data(split_seed):
    # загрузка обучающих данных
    X, y = load_digits(return_X_y=True)
    idx = np.arange(len(y))

    # разбиение на train, val, test
    rest, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=split_seed)
    train, val = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=split_seed)

    # сохранение разбиения
    Path("data").mkdir(exist_ok=True)
    split = {k: sorted(map(int, v)) for k, v in
             {"train": train, "val": val, "test": test}.items()}
    split_path = Path("data/split.json")
    split_path.write_text(json.dumps(split, sort_keys=True), encoding="utf-8")
    sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    entry = {
        "source": "sklearn.datasets.load_digits",
        "version": f"scikit-learn {sklearn.__version__}",
        "selection": {"total": len(y), "used": len(y), "excluded": 0},
        "split_rule": (f"stratified by class, split_seed={split_seed}: test = 20% of all; "
                       "val = 25% of the remainder; train = the rest"),
        "files": [{"path": "data/split.json", "sha256": sha, "records": len(y)}],
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
    }
    manifest_path = Path("data/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["lr0_digits"] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return X, y, train, val, test, sha


def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def main():
    os.makedirs("runs", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    seed_everything(seed=args.seed, num_threads=1)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(config.get("experiment"))

    split_seed = int(config.get("split_seed"))
    batch_size = int(config.get("train").get("batch_size"))
    lr = float(config.get("train").get("lr"))
    weight_decay = float(config.get("train").get("weight_decay"))
    epochs = int(config.get("train").get("epochs"))

    hidden_layers = int(config.get("model").get("hidden"))
    n_layers = int(config.get("model").get("n_layers"))
    dropout = float(config.get("model").get("dropout"))

    run_id = uuid.uuid4().hex
    os.makedirs(f"runs/{run_id}")
    with open(f"runs/{run_id}/config.yaml", "w") as yaml_config:
        yaml.dump(config, yaml_config)
    with open(f"runs/{run_id}/history.csv", "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_FIELDS)

        with mlflow.start_run(run_name=run_id) as run:
            mlflow.log_params({
                "seed": args.seed,
                "split_seed": split_seed,
                "lr": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "n_layers": n_layers,
                "hidden": hidden_layers,
                "dropout": dropout,
            })
            X, y, train, val, test, sha = prepare_data(split_seed=split_seed)
            mlflow.set_tag("split_sha256", sha)
            train_loader = DataLoader(make_dataset(X, y, train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(make_dataset(X, y, val), batch_size=batch_size)

            model = configure_mlp(n_layers, hidden_layers, dropout)
            train_model(
                model,
                num_epochs=epochs,
                lr=lr,
                train_loader=train_loader,
                val_loader=val_loader,
                weight_decay=weight_decay,
                csv_file=csv_file
            )
            if args.eval_test:
                test_loader = DataLoader(make_dataset(X, y, test), batch_size=batch_size)
                test_accuracy = eval_model(model, test_loader)
                print(f"Test accuracy: {test_accuracy:.2%}")


if __name__ == "__main__":
    main()
