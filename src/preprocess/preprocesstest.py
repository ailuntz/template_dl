from pathlib import Path

import numpy as np
import pandas as pd


class preprocesstest:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self) -> None:
        raw_dir = Path(self.cfg.get("raw_dir", "data/raw"))
        out_dir = Path(self.cfg.get("out_dir", "data/processed"))
        train_size = int(self.cfg.get("train_size", 256))
        val_size = int(self.cfg.get("val_size", 64))
        test_size = int(self.cfg.get("test_size", 64))
        num_features = int(self.cfg.get("num_features", 32))
        num_classes = int(self.cfg.get("num_classes", 10))

        raw_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        self._generate_raw_data(raw_dir, train_size, val_size, test_size, num_features, num_classes)
        self._process_csv(raw_dir, out_dir)
        print(f"预处理完成: raw={raw_dir}, processed={out_dir}")

    @staticmethod
    def _generate_raw_data(raw_dir: Path, train_size: int, val_size: int,
                           test_size: int, num_features: int, num_classes: int) -> None:
        np.random.seed(42)
        for split, size in [("train", train_size), ("val", val_size), ("test", test_size)]:
            data = np.random.randn(size, num_features)
            labels = np.random.randint(0, num_classes, size)
            df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(num_features)])
            df["label"] = labels
            df.to_csv(raw_dir / f"{split}.csv", index=False)

    @staticmethod
    def _process_csv(raw_dir: Path, out_dir: Path) -> None:
        train_df = pd.read_csv(raw_dir / "train.csv")
        val_df = pd.read_csv(raw_dir / "val.csv")
        test_df = pd.read_csv(raw_dir / "test.csv")

        feature_cols = [col for col in train_df.columns if col != "label"]
        mean = train_df[feature_cols].mean()
        std = train_df[feature_cols].std()

        for df, split in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            df[feature_cols] = (df[feature_cols] - mean) / (std + 1e-8)
            df.to_csv(out_dir / f"{split}.csv", index=False)
