from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.ett_small_forecast import _load_ett_small


class ETTSmallImputationDataset(Dataset):
    def __init__(self, cfg: Dict, split: str):
        self.split = split
        model_cfg = cfg["model"][cfg["model_name"]]
        task_name = str(model_cfg.get("task_name") or "").lower()
        task_cfg = (cfg.get("task") or {}).get(task_name)
        if task_cfg is None:
            raise ValueError("缺少 task 配置")
        self.mask_rate = float(task_cfg.get("mask_rate", 0.25))
        self.mask_seed = int(cfg.get("seed", 42))
        if self.mask_rate < 0 or self.mask_rate >= 1:
            raise ValueError("mask_rate 需要在 [0, 1) 范围内")

        payload = _load_ett_small(cfg, split)
        self.data_x = payload["data_x"]
        self.data_stamp = payload["data_stamp"]
        self.scaler = payload["scaler"]
        self.seq_len = payload["seq_len"]
        self.use_time_features = payload["use_time_features"]

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end].astype(np.float32)
        seq_y = seq_x.copy()
        rng = np.random.default_rng(self.mask_seed + index)
        mask = rng.random(seq_x.shape) < self.mask_rate
        seq_x[mask] = 0.0

        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if not self.use_time_features:
            return seq_x, seq_y, mask

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, mask

    def __len__(self) -> int:
        length = len(self.data_x) - self.seq_len + 1
        return max(length, 0)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)


__all__ = ["ETTSmallImputationDataset"]
