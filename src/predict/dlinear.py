from pathlib import Path

import numpy as np


def save_dlinear_preds(arr: np.ndarray, save_dir: Path) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "preds.csv"
    if arr.ndim == 1:
        data = arr.reshape(-1, 1)
        header = ["t0"]
    elif arr.ndim == 2:
        data = arr
        header = [f"t{i}" for i in range(arr.shape[1])]
    elif arr.ndim == 3:
        steps, dims = arr.shape[1], arr.shape[2]
        data = arr.reshape(arr.shape[0], steps * dims)
        header = [f"t{t}_c{c}" for t in range(steps) for c in range(dims)]
    else:
        data = arr.reshape(arr.shape[0], -1)
        header = [f"f{i}" for i in range(data.shape[1])]
    np.savetxt(out_path, data, delimiter=",", header=",".join(header), comments="")
    return out_path
