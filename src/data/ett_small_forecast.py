from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray) -> None:
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std == 0] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


def build_time_features(df_stamp: pd.DataFrame, freq: str, date_col: str) -> np.ndarray:
    df_stamp = df_stamp.copy()
    df_stamp[date_col] = pd.to_datetime(df_stamp[date_col])
    df_stamp["month"] = df_stamp[date_col].dt.month
    df_stamp["day"] = df_stamp[date_col].dt.day
    df_stamp["weekday"] = df_stamp[date_col].dt.weekday
    df_stamp["hour"] = df_stamp[date_col].dt.hour
    if freq == "t":
        df_stamp["minute"] = df_stamp[date_col].dt.minute
    return df_stamp.drop(columns=[date_col]).values


def _load_ett_small(cfg: Dict, split: str) -> Dict:
    dataset_cfg = cfg["dataset"][cfg["dataset_name"]]
    model_cfg = cfg["model"][cfg["model_name"]]
    task_name = str(model_cfg.get("task_name") or "").lower()
    task_cfg = (cfg.get("task") or {}).get(task_name)
    if task_cfg is None:
        raise ValueError("缺少 task 配置")
    data_cfg = cfg["data"]
    schema = dataset_cfg["schema"]
    root = Path(dataset_cfg["root"])
    infer_path = dataset_cfg["infer_data_path"]
    if split == "predict" and infer_path:
        data_path = infer_path
    else:
        data_path = dataset_cfg["data_path"]
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = root / data_path
    date_col = str(schema["date_col"])
    features = str(schema["features"])
    target = str(schema["target"])
    freq = str(schema["freq"])
    seq_len = int(task_cfg["seq_len"])
    label_len = int(task_cfg.get("label_len", 0))
    pred_len = int(task_cfg["pred_len"])
    scale = bool(task_cfg.get("scale", False))
    timeenc = int(task_cfg.get("timeenc", 0))
    split_mode = str(dataset_cfg.get("split_mode") or "ratio").lower()
    split_ratio = data_cfg["split"] if split_mode == "ratio" else None
    use_time_features = bool(task_cfg.get("time_features", False))

    if features not in {"S", "M", "MS"}:
        raise ValueError(f"不支持的 features: {features}")
    if timeenc not in {0}:
        raise ValueError(f"暂不支持 timeenc={timeenc}")
    if split_mode not in {"ratio", "tsl"}:
        raise ValueError(f"不支持的 split_mode: {split_mode}")
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    df_raw = pd.read_csv(data_path)
    if date_col not in df_raw.columns:
        raise ValueError(f"数据缺少 {date_col} 列")

    if features == "S":
        cols = [target]
    else:
        cols = [col for col in df_raw.columns if col != date_col]
    if target not in cols:
        raise ValueError(f"目标列不存在: {target}")

    target_index = cols.index(target)
    data = df_raw[cols].values.astype(np.float32)
    num_features = data.shape[1]
    out_dim = 1 if features == "MS" else num_features

    scaler = StandardScaler()
    if split_mode == "tsl":
        if freq == "h":
            scale_freq = 1
        elif freq == "t":
            scale_freq = 4
        else:
            raise ValueError(f"不支持的 freq: {freq}")

        month = 30 * 24 * scale_freq
        border1s = [0, 12 * month - seq_len, 12 * month + 4 * month - seq_len]
        border2s = [12 * month, 12 * month + 4 * month, 12 * month + 8 * month]

        if split == "predict":
            border1, border2 = 0, len(data)
        else:
            type_map = {"train": 0, "val": 1, "test": 2}
            set_type = type_map.get(split)
            if set_type is None:
                raise ValueError(f"不支持的 split: {split}")
            border1 = border1s[set_type]
            border2 = border2s[set_type]

        if scale:
            scaler.fit(data[border1s[0]:border2s[0]])
            data = scaler.transform(data)
    else:
        n = len(data)
        train_ratio = float(split_ratio["train"])
        val_ratio = float(split_ratio["val"])
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        min_train = seq_len + pred_len
        train_end = min(max(train_end, min_train), n)
        val_end = min(max(val_end, train_end + pred_len), n)

        if split == "predict":
            border1, border2 = 0, n
        elif split == "train":
            border1, border2 = 0, train_end
        elif split == "val":
            border1, border2 = max(0, train_end - seq_len), val_end
        elif split == "test":
            border1, border2 = max(0, val_end - seq_len), n
        else:
            raise ValueError(f"不支持的 split: {split}")

        if scale:
            scaler.fit(data[0:train_end])
            data = scaler.transform(data)

    data_x = data[border1:border2]
    data_y = data[border1:border2]

    data_stamp = None
    if use_time_features:
        df_stamp = df_raw[[date_col]].iloc[border1:border2]
        data_stamp = build_time_features(df_stamp, freq, date_col)

    return {
        "data_x": data_x,
        "data_y": data_y,
        "data_stamp": data_stamp,
        "scaler": scaler,
        "num_features": num_features,
        "target_index": target_index,
        "out_dim": out_dim,
        "seq_len": seq_len,
        "label_len": label_len,
        "pred_len": pred_len,
        "features": features,
        "use_time_features": use_time_features,
    }


class ETTSmallForecastDataset(Dataset):
    def __init__(self, cfg: Dict, split: str):
        self.split = split
        payload = _load_ett_small(cfg, split)
        self.data_x = payload["data_x"]
        self.data_y = payload["data_y"]
        self.data_stamp = payload["data_stamp"]
        self.scaler = payload["scaler"]
        self.num_features = payload["num_features"]
        self.target_index = payload["target_index"]
        self.out_dim = payload["out_dim"]
        self.seq_len = payload["seq_len"]
        self.label_len = payload["label_len"]
        self.pred_len = payload["pred_len"]
        self.features = payload["features"]
        self.use_time_features = payload["use_time_features"]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len if self.label_len > 0 else s_end
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        if self.features == "MS":
            seq_y = seq_y[:, [self.target_index]]

        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)

        if not self.use_time_features:
            return seq_x, seq_y

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        length = len(self.data_x) - self.seq_len - self.pred_len + 1
        return max(length, 0)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)


__all__ = ["ETTSmallForecastDataset", "_load_ett_small"]
