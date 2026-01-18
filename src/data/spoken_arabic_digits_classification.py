from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_CACHE = {}


def _parse_ts_file(path: Path) -> Tuple[List[List[np.ndarray]], List[str]]:
    series_list = []
    labels = []
    with path.open("r", encoding="utf-8") as f:
        data_started = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not data_started:
                if line.lower() == "@data":
                    data_started = True
                continue
            parts = line.split(":")
            if len(parts) < 2:
                continue
            label = parts[-1].strip()
            dims = []
            for dim in parts[:-1]:
                dim = dim.strip()
                if not dim:
                    dims.append(np.array([], dtype=np.float32))
                    continue
                vals = []
                for v in dim.split(","):
                    v = v.strip()
                    if not v:
                        continue
                    if v in {"?", "NaN", "nan"}:
                        vals.append(np.nan)
                    else:
                        vals.append(float(v))
                dims.append(np.array(vals, dtype=np.float32))
            series_list.append(dims)
            labels.append(label)
    return series_list, labels


def _as_matrix(sample: List[np.ndarray]) -> np.ndarray:
    if not sample:
        return np.zeros((0, 0), dtype=np.float32)
    lengths = [len(d) for d in sample]
    if not lengths or min(lengths) == 0:
        return np.zeros((0, len(sample)), dtype=np.float32)
    length = min(lengths)
    data = np.stack([d[:length] for d in sample], axis=1)
    return np.nan_to_num(data)


def _build_cache(train_path: Path, test_path: Path, max_seq_len: int | None):
    key = (str(train_path), str(test_path), max_seq_len)
    if key in _CACHE:
        return _CACHE[key]

    train_series, train_labels = _parse_ts_file(train_path)
    test_series, test_labels = _parse_ts_file(test_path)
    all_series = train_series + test_series
    num_features = max((len(s) for s in all_series), default=0)

    if max_seq_len is None or max_seq_len <= 0:
        max_seq_len = 0
        for sample in all_series:
            if not sample:
                continue
            length = min(len(d) for d in sample if len(d) > 0)
            if length > max_seq_len:
                max_seq_len = length

    labels_all = sorted(set(train_labels + test_labels), key=lambda x: int(float(str(x).strip())))
    label_map = {label: idx for idx, label in enumerate(labels_all)}

    cache = {
        "train_series": train_series,
        "train_labels": train_labels,
        "test_series": test_series,
        "test_labels": test_labels,
        "num_features": num_features,
        "max_seq_len": max_seq_len,
        "label_map": label_map,
    }
    _CACHE[key] = cache
    return cache


def _compute_stats(series: List[List[np.ndarray]], num_features: int):
    sums = np.zeros(num_features, dtype=np.float64)
    sums_sq = np.zeros(num_features, dtype=np.float64)
    count = 0
    for sample in series:
        arr = _as_matrix(sample)
        if arr.size == 0:
            continue
        sums += arr.sum(axis=0)
        sums_sq += (arr ** 2).sum(axis=0)
        count += arr.shape[0]
    if count == 0:
        mean = np.zeros(num_features, dtype=np.float32)
        std = np.ones(num_features, dtype=np.float32)
    else:
        mean = (sums / count).astype(np.float32)
        var = sums_sq / count - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)
    return mean, std


class SpokenArabicDigitsDataset(Dataset):
    def __init__(self, cfg: Dict, split: str):
        dataset_cfg = cfg["dataset"][cfg["dataset_name"]]
        data_cfg = cfg["data"]
        root = Path(dataset_cfg["root"])
        train_name = dataset_cfg.get("train_path") or "SpokenArabicDigits_TRAIN.ts"
        test_name = dataset_cfg.get("test_path") or "SpokenArabicDigits_TEST.ts"
        train_path = Path(train_name)
        if not train_path.is_absolute():
            train_path = root / train_path
        test_path = Path(test_name)
        if not test_path.is_absolute():
            test_path = root / test_path
        max_seq_len = int(dataset_cfg.get("max_seq_len", 0))
        self.normalize = bool(dataset_cfg.get("normalize", True))
        self.flatten = bool(dataset_cfg.get("flatten", True))

        cache = _build_cache(train_path, test_path, max_seq_len)
        self.max_seq_len = cache["max_seq_len"]
        self.num_features = cache["num_features"]
        self.label_map = cache["label_map"]
        self.mean = None
        self.std = None

        split_cfg = data_cfg.get("split") or {"train": 0.8, "val": 0.2, "test": 0.0}
        use_train_val_split = bool(data_cfg.get("use_train_val_split", True))
        if use_train_val_split:
            if split in {"train", "val"}:
                series = cache["train_series"]
                labels = cache["train_labels"]
                total = len(series)
                train_ratio = float(split_cfg.get("train", 0.8))
                val_ratio = float(split_cfg.get("val", 0.2))
                train_end = int(total * train_ratio)
                val_end = int(total * (train_ratio + val_ratio))
                if split == "train":
                    self.series = series[:train_end]
                    self.labels = labels[:train_end]
                else:
                    self.series = series[train_end:val_end]
                    self.labels = labels[train_end:val_end]
            elif split in {"test", "predict"}:
                self.series = cache["test_series"]
                self.labels = cache["test_labels"]
            else:
                raise ValueError(f"不支持的 split: {split}")
        else:
            if split == "train":
                self.series = cache["train_series"]
                self.labels = cache["train_labels"]
            elif split in {"val", "test", "predict"}:
                self.series = cache["test_series"]
                self.labels = cache["test_labels"]
            else:
                raise ValueError(f"不支持的 split: {split}")

        if self.normalize:
            self.mean, self.std = _compute_stats(self.series, self.num_features)

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, idx: int):
        sample = self.series[idx]
        label = self.labels[idx]
        arr = _as_matrix(sample)
        seq_len = arr.shape[0]
        if self.normalize and arr.size > 0:
            arr = (arr - self.mean) / (self.std + 1e-6)
        if arr.shape[0] < self.max_seq_len:
            pad_len = self.max_seq_len - arr.shape[0]
            pad = np.zeros((pad_len, arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])
        elif arr.shape[0] > self.max_seq_len:
            arr = arr[:self.max_seq_len]
            seq_len = self.max_seq_len
        if self.flatten:
            features = torch.tensor(arr.reshape(-1), dtype=torch.float32)
        else:
            features = torch.tensor(arr, dtype=torch.float32)
        target = torch.tensor(self.label_map[label], dtype=torch.long)
        valid_len = min(seq_len, self.max_seq_len)
        mask = torch.zeros(self.max_seq_len, dtype=torch.float32)
        if valid_len > 0:
            mask[:valid_len] = 1.0
        return features, target, mask
