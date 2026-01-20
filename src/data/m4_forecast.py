from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.hf import ensure_hf_dataset

class M4Meta:
    horizons_map = {
        "Yearly": 6,
        "Quarterly": 8,
        "Monthly": 18,
        "Weekly": 13,
        "Daily": 14,
        "Hourly": 48,
    }
    history_size = {
        "Yearly": 1.5,
        "Quarterly": 1.5,
        "Monthly": 1.5,
        "Weekly": 10,
        "Daily": 10,
        "Hourly": 10,
    }


def _load_m4_values(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if hasattr(data, "files"):
        if not data.files:
            raise ValueError(f"空的 npz 文件: {path}")
        return data[data.files[0]]
    return data


class M4ForecastDataset(Dataset):
    def __init__(self, cfg: Dict, split: str):
        dataset_cfg = cfg["dataset"][cfg["dataset_name"]]
        model_cfg = cfg["model"][cfg["model_name"]]
        task_name = str(model_cfg.get("task_name") or "").lower()
        task_cfg = (cfg.get("task") or {}).get(task_name)
        if task_cfg is None:
            raise ValueError("缺少 task 配置")
        self.root = Path(dataset_cfg["root"])
        ensure_hf_dataset(self.root, dataset_cfg.get("huggingface_repo"))
        self.seasonal_patterns = str(dataset_cfg["seasonal_patterns"])
        self.seq_len = int(task_cfg["seq_len"])
        self.label_len = int(task_cfg.get("label_len", 0))
        self.pred_len = int(task_cfg["pred_len"])
        self.sample_seed = int(cfg.get("seed", 42))
        self.history_size = float(M4Meta.history_size[self.seasonal_patterns])
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.split = split
        self.ids = []
        self.series = []
        self.train_series = []
        self.test_series = []
        self._read_data()

    def _read_data(self) -> None:
        info_path = self.root / "M4-info.csv"
        train_path = self.root / "training.npz"
        test_path = self.root / "test.npz"
        if not info_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {info_path}")
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {train_path} 或 {test_path}")

        info = pd.read_csv(info_path)
        mask = info["SP"] == self.seasonal_patterns
        indices = info.index[mask].to_list()
        self.ids = info.loc[mask, "M4id"].to_list()

        train_values = _load_m4_values(train_path)
        if self.split == "train":
            self.series = [train_values[i][~np.isnan(train_values[i])] for i in indices]
            return

        test_values = _load_m4_values(test_path)
        self.train_series = [train_values[i][~np.isnan(train_values[i])] for i in indices]
        self.test_series = [test_values[i][~np.isnan(test_values[i])] for i in indices]

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.series)
        return len(self.test_series)

    def __getitem__(self, idx: int):
        insample = np.zeros((self.seq_len, 1), dtype=np.float32)
        insample_mask = np.zeros((self.seq_len, 1), dtype=np.float32)
        out_len = self.label_len + self.pred_len
        outsample = np.zeros((out_len, 1), dtype=np.float32)
        outsample_mask = np.zeros((out_len, 1), dtype=np.float32)

        if self.split == "train":
            sampled = self.series[idx]
            rng = np.random.default_rng(self.sample_seed + idx)
            cut_point = rng.integers(
                low=max(1, len(sampled) - self.window_sampling_limit),
                high=len(sampled),
                endpoint=False,
            )
            insample_window = sampled[max(0, cut_point - self.seq_len):cut_point]
            outsample_window = sampled[max(0, cut_point - self.label_len):min(len(sampled), cut_point + self.pred_len)]
            out_start = 0
        else:
            insample_window = self.train_series[idx][-self.seq_len:]
            outsample_window = self.test_series[idx]
            out_start = self.label_len

        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample[out_start:out_start + len(outsample_window), 0] = outsample_window
        outsample_mask[out_start:out_start + len(outsample_window), 0] = 1.0

        seq_x = torch.tensor(insample, dtype=torch.float32)
        seq_y = torch.tensor(outsample, dtype=torch.float32)
        return seq_x, seq_y, insample_mask, outsample_mask
