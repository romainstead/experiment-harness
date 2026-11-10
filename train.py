import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import sklearn
import torch
import yaml
import optuna
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import seed_everything, configure_mlp, train_model, make_dataset, write_predictions, CSV_FIELDS, run_correct_param_and_clean_git_check
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


def summarize_metrics(run_dir, eval_test):
    with (run_dir / "history.csv").open(newline="", encoding="utf-8") as history_file:
        history = list(csv.DictReader(history_file))
    if not history:
        raise ValueError("history.csv contains no completed epochs")

    with (run_dir / "predictions.csv").open(newline="", encoding="utf-8") as predictions_file:
        predictions = list(csv.DictReader(predictions_file))

    metrics = {
        "val_loss": float(history[-1]["val_loss"]),
        "training_time_seconds": math.fsum(float(row["epoch_time"]) for row in history),
    }
    for split in ("val", "test") if eval_test else ("val",):
        rows = [row for row in predictions if row["split"] == split]
        if not rows:
            raise ValueError(f"predictions.csv contains no predictions for {split}")
        metrics[f"{split}_acc"] = sum(row["y_true"] == row["y_pred"] for row in rows) / len(rows)

    return metrics


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

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex[:16]
    os.makedirs(f"runs/{run_id}")

    with open(f"runs/{run_id}/config.yaml", "w") as yaml_config:
        yaml.dump(config, yaml_config)

    X, y, train, val, test, sha = prepare_data(split_seed=split_seed)
    repo_dir = Path(__file__).resolve().parent
    metadata = {"run_id": run_id, "status": "RUNNING", "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": "", "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(), "python_version": "", "pytorch_version": torch.__version__,
                "sklearn_version": sklearn.__version__, "optuna_version": optuna.__version__,
                "mlflow_version": mlflow.__version__, "num_threads": config.get("runtime").get("num_threads"),
                "seed": args.seed, "split_sha": sha}

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

            mlflow.set_tag("split_sha256", sha)
            train_loader = DataLoader(make_dataset(X, y, train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(make_dataset(X, y, val), batch_size=batch_size)

            model = configure_mlp(n_layers, hidden_layers, dropout)
            try:
                run_correct_param_and_clean_git_check(config)
                model = train_model(
                    model,
                    num_epochs=epochs,
                    lr=lr,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    weight_decay=weight_decay,
                    csv_file=csv_file
                )

                run_dir = Path("runs") / run_id
                predictions_path = run_dir / "predictions.csv"
                with predictions_path.open("w", newline="", encoding="utf-8") as predictions_file:
                    writer = csv.writer(predictions_file)
                    writer.writerow(["split", "index", "y_true", "y_pred"])

                    val_predictions_loader = DataLoader(
                        make_dataset(X, y, val, include_indices=True),
                        batch_size=batch_size,
                    )
                    write_predictions(model, val_predictions_loader, writer, "val")

                    if args.eval_test:
                        test_loader = DataLoader(
                            make_dataset(X, y, test, include_indices=True),
                            batch_size=batch_size,
                        )
                        write_predictions(model, test_loader, writer, "test")

                csv_file.flush()
                metrics = summarize_metrics(run_dir, args.eval_test)
                (run_dir / "metrics.json").write_text(
                    json.dumps(metrics, indent=2) + "\n",
                    encoding="utf-8",
                )
                if args.eval_test:
                    print(f"Test accuracy: {metrics['test_acc']:.2%}")
            except Exception as e:
                metadata["status"] = "FAILED"
                raise
            else:
                metadata["status"] = "COMPLETED"
            finally:
                metadata["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(f"runs/{run_id}/metadata.json", "w") as meta_file:
                    json.dump(metadata, meta_file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
