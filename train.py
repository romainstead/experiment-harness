import hashlib
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import seed_everything, configure_mlp, train_model, make_dataset


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


def main():
    seed_everything(seed=0, num_threads=1)
    X, y, train, val, _, _ = prepare_data(split_seed=0)
    train_loader = DataLoader(make_dataset(X, y, train), batch_size=64, shuffle=True)
    val_loader = DataLoader(make_dataset(X, y, val), batch_size=64)
    model = configure_mlp(1, 64, 0.0)
    train_model(
        model,
        num_epochs=10,
        lr=1e-3,
        train_loader=train_loader,
        val_loader=val_loader,
        weight_decay=0.0,
    )


if __name__ == "__main__":
    main()
