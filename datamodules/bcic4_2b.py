from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule
from utils.load_bcic4 import load_bcic4


class BCICIV2b(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["hand(L)", "hand(R)"]
    channels = 3
    classes = 2

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2b",
                                  preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("session")
        train_datasets = [splitted_ds[f"session_{session}"] for session in [0, 1, 2]]
        test_datasets = [splitted_ds[f"session_{session}"] for session in [3, 4]]

        # load the data
        X = np.concatenate(
            [run.windows.load_data()._data for train_dataset in train_datasets for run
             in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for train_dataset in train_datasets for run in
                            train_dataset.datasets], axis=0)
        X_test = np.concatenate(
            [run.windows.load_data()._data for test_dataset in test_datasets for run in
             test_dataset.datasets], axis=0)
        y_test = np.concatenate([run.y for test_dataset in test_datasets for run in
                                 test_dataset.datasets], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)


class BCICIV2bLOSO(BCICIV2b):
    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super(BCICIV2bLOSO, self).__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(
            subject_ids=self.all_subject_ids, dataset="2b",
            preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("subject")
        train_subjects = [
            subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]
        train_datasets = [
            splitted_ds[str(subj_id)].split("session")[f"session_{session}"] for
            subj_id in train_subjects for session in [0, 1, 2]]
        val_datasets = [
            splitted_ds[str(subj_id)].split("session")[f"session_{session}"] for
            subj_id in train_subjects for session in [3, 4]]
        test_datasets = [
            splitted_ds[str(self.subject_id)].split("session")[f"session_{session}"]
            for session in [3, 4]]

        # load the data
        X = np.concatenate([run.windows.load_data()._data for train_dataset in
                            train_datasets for run in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for train_dataset in train_datasets for run in
                            train_dataset.datasets], axis=0)
        X_val = np.concatenate([run.windows.load_data()._data for val_dataset in
                            val_datasets for run in val_dataset.datasets], axis=0)
        y_val = np.concatenate([run.y for val_dataset in val_datasets for run in
                            val_dataset.datasets], axis=0)
        X_test = np.concatenate([run.windows.load_data()._data for test_dataset in test_datasets
                                 for run in test_dataset.datasets], axis=0)
        y_test = np.concatenate([run.y for test_dataset in test_datasets for run in
                                 test_dataset.datasets], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"])
