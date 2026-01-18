from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.ett_small_forecast import StandardScaler


class SWaTAnomalyDataset(Dataset):
    def __init__(self, cfg: Dict, split: str):
        dataset_cfg = cfg["dataset"][cfg["dataset_name"]]
        model_cfg = cfg["model"][cfg["model_name"]]
        task_name = str(model_cfg.get("task_name") or "").lower()
        task_cfg = (cfg.get("task") or {}).get(task_name)
        if task_cfg is None:
            raise ValueError("缺少 task 配置")
        data_cfg = cfg["data"]
        self.root = Path(dataset_cfg["root"])
        train_path = dataset_cfg.get("train_path") or "swat_train2.csv"
        test_path = dataset_cfg.get("test_path") or "swat2.csv"
        self.train_path = Path(train_path)
        if not self.train_path.is_absolute():
            self.train_path = self.root / self.train_path
        self.test_path = Path(test_path)
        if not self.test_path.is_absolute():
            self.test_path = self.root / self.test_path
        schema = dataset_cfg.get("schema") or {}
        self.label_col = str(schema.get("label_col") or dataset_cfg.get("label_col") or "Normal/Attack")
        self.seq_len = int(task_cfg["seq_len"])
        self.step = int(task_cfg.get("window_step", 1))
        self.split = split
        self.data = None
        self.labels = None
        self._load_data(data_cfg)

    def _load_data(self, data_cfg: Dict) -> None:
        if not self.train_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.train_path}")
        if not self.test_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.test_path}")

        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        if self.label_col in train_df.columns:
            train_df = train_df.drop(columns=[self.label_col])
        if self.label_col in test_df.columns:
            raw_labels = test_df[self.label_col].to_numpy()
            test_df = test_df.drop(columns=[self.label_col])
        else:
            raw_labels = test_df.iloc[:, -1].to_numpy()
            test_df = test_df.iloc[:, :-1]
        test_labels = self._to_binary_labels(raw_labels)

        scaler = StandardScaler()
        scaler.fit(train_df.values)
        train = scaler.transform(train_df.values)
        test = scaler.transform(test_df.values)

        data_len = len(train)
        split_mode = str(data_cfg.get("split_mode") or "ratio").lower()
        if split_mode == "tsl":
            val_start = int(data_len * 0.8)
            val_start = max(val_start, self.seq_len)
            if self.split == "train":
                self.data = train
                self.labels = np.zeros(len(self.data), dtype=np.float32)
            elif self.split == "val":
                self.data = train[val_start:]
                self.labels = np.zeros(len(self.data), dtype=np.float32)
            elif self.split in {"test", "predict"}:
                self.data = test
                if len(test_labels) != len(test):
                    size = min(len(test_labels), len(test))
                    self.data = self.data[:size]
                    test_labels = test_labels[:size]
                self.labels = test_labels.astype(np.float32)
            else:
                raise ValueError(f"不支持的 split: {self.split}")
            return

        split_cfg = data_cfg.get("split") or {"train": 0.8, "val": 0.2, "test": 0.0}
        train_ratio = float(split_cfg.get("train", 0.8))
        val_ratio = float(split_cfg.get("val", 0.2))
        train_end = int(data_len * train_ratio)
        val_end = int(data_len * (train_ratio + val_ratio))
        train_end = max(train_end, self.seq_len)
        val_end = min(max(val_end, train_end + self.seq_len), data_len)

        if self.split == "train":
            self.data = train[:train_end]
            self.labels = np.zeros(len(self.data), dtype=np.float32)
        elif self.split == "val":
            self.data = train[train_end:val_end]
            self.labels = np.zeros(len(self.data), dtype=np.float32)
        elif self.split in {"test", "predict"}:
            self.data = test
            if len(test_labels) != len(test):
                size = min(len(test_labels), len(test))
                self.data = self.data[:size]
                test_labels = test_labels[:size]
            self.labels = test_labels.astype(np.float32)
        else:
            raise ValueError(f"不支持的 split: {self.split}")

    def _to_binary_labels(self, values: np.ndarray) -> np.ndarray:
        if values.dtype.kind in {"i", "u", "f"}:
            return values.astype(np.float32)
        labels = []
        for v in values:
            s = str(v).strip().lower()
            if s in {"1", "true", "attack"}:
                labels.append(1.0)
            else:
                labels.append(0.0)
        return np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        if len(self.data) < self.seq_len:
            return 0
        return (len(self.data) - self.seq_len) // self.step + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = idx * self.step
        end = start + self.seq_len
        seq_x = self.data[start:end]
        seq_y = seq_x
        mask = np.ones_like(seq_x, dtype=np.float32)
        labels = self.labels[start:end]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        return seq_x, seq_y, labels, mask
