from pathlib import Path

import numpy as np


def save_classification_preds(arr: np.ndarray, save_dir: Path) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "preds.csv"
    if arr.ndim == 1:
        data = arr.reshape(-1, 1)
        header = ["pred"]
    elif arr.ndim == 2:
        pred_idx = arr.argmax(axis=1)
        data = np.concatenate([pred_idx.reshape(-1, 1), arr], axis=1)
        header = ["pred"] + [f"c{i}" for i in range(arr.shape[1])]
    else:
        data = arr.reshape(arr.shape[0], -1)
        header = [f"f{i}" for i in range(data.shape[1])]
    np.savetxt(out_path, data, delimiter=",", header=",".join(header), comments="")
    return out_path
