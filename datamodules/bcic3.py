from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from .base import BaseDataModule
from utils.load_bcic3 import load_bcic3
import os


class BCICIII_IVa(BaseDataModule):
    all_subject_ids = [1, 2, 3, 4, 5]
    class_names = ["hand(R)", "feet"]
    channels = 3
    classes = 2

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic3([self.subject_id], self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        X = self.dataset["data"][str(self.subject_id)]["train"]
        y = self.dataset["labels"][str(self.subject_id)]["train"]
        X_test = self.dataset["data"][str(self.subject_id)]["test"]
        y_test = self.dataset["labels"][str(self.subject_id)]["test"]

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

class BCICIII_IVaLOSO(BaseDataModule):
    all_subject_ids = [1, 2, 3, 4, 5]
    class_names = ["hand(R)", "feet"]
    val_dataset = None
    channels = 3
    classes = 2

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic3(self.all_subject_ids, self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        train_subjects = [subj for subj in self.all_subject_ids if subj != self.subject_id]
        X = np.concatenate([self.dataset["data"][str(subj)]["train"] for subj in train_subjects], axis=0)
        y = np.concatenate([self.dataset["labels"][str(subj)]["train"] for subj in train_subjects], axis=0)
        X_val = np.concatenate(
            [self.dataset["data"][str(subj)]["test"] for subj in train_subjects],
            axis=0)
        y_val = np.concatenate(
            [self.dataset["labels"][str(subj)]["test"] for subj in train_subjects],
            axis=0)

        X_test = self.dataset["data"][str(self.subject_id)]["test"]
        y_test = self.dataset["labels"][str(self.subject_id)]["test"]

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True)
