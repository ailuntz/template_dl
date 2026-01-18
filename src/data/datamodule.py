from typing import Optional, Type

import lightning as L
from torch.utils.data import DataLoader, Dataset


class BaseDataModule(L.LightningDataModule):
    def __init__(self, cfg: dict, dataset_cls: Type[Dataset]):
        super().__init__()
        self.cfg = cfg
        self.dataset_cls = dataset_cls
        dataset_cfg = cfg["dataset"][cfg["dataset_name"]]
        data_cfg = cfg["data"]
        loader_cfg = data_cfg["loader"]
        self.batch_size = int(loader_cfg["batch_size"])
        self.num_workers = int(loader_cfg["num_workers"])
        self.persistent_workers = bool(loader_cfg.get("persistent_workers", False))
        if "drop_last" in loader_cfg:
            self.drop_last_train = bool(loader_cfg["drop_last"])
            self.drop_last_eval = bool(loader_cfg["drop_last"])
        else:
            train_cfg = loader_cfg.get("train") or {}
            eval_cfg = loader_cfg.get("eval") or {}
            self.drop_last_train = bool(train_cfg.get("drop_last", False))
            self.drop_last_eval = bool(eval_cfg.get("drop_last", False))
        self.use_predict_set = bool(dataset_cfg["infer_data_path"])
        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None
        self.predict_set: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_set = self.dataset_cls(self.cfg, "train")
            self.val_set = self.dataset_cls(self.cfg, "val")
        if stage in (None, "test", "predict"):
            self.test_set = self.dataset_cls(self.cfg, "test")
        if stage in (None, "predict") and self.use_predict_set:
            self.predict_set = self.dataset_cls(self.cfg, "predict")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last_train,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last_eval,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last_eval,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        dataset = self.predict_set if self.predict_set is not None else self.test_set
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last_eval,
            persistent_workers=self.persistent_workers,
        )
