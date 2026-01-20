from pathlib import Path


def ensure_hf_dataset(root: Path, repo: str | None) -> None:
    if not repo:
        return
    if root.exists():
        try:
            if any(root.iterdir()):
                return
        except PermissionError:
            pass
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("缺少 huggingface_hub，请先安装") from exc
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(root),
        local_dir_use_symlinks=False,
    )
