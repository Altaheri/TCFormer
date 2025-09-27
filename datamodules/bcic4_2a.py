
from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule
from utils.load_bcic4 import load_bcic4
from sklearn.model_selection import train_test_split
import os


class BCICIV2a(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4 
    
    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2a",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("session")
        train_dataset, test_dataset = splitted_ds["session_T"], splitted_ds["session_E"]

        # load the data
        X = np.concatenate(
            [run.windows.load_data()._data for run in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for run in train_dataset.datasets], axis=0)
        X_test = np.concatenate(
            [run.windows.load_data()._data for run in test_dataset.datasets], axis=0)
        y_test = np.concatenate([run.y for run in test_dataset.datasets], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)                                                                
        # self.train_dataset = BaseDataModule._make_tensor_dataset(X, y, 
                                                                #  preprocessing_dict=self.preprocessing_dict, mode="train")
        # self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test, 
                                                                #  preprocessing_dict=self.preprocessing_dict, mode="test")


class BCICIV2aTVT(BaseDataModule):
    val_dataset = None
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4 

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2a",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()

        # Split by session
        splitted_ds = self.dataset.split("session")
        session1 = splitted_ds["session_T"]  # training + validation
        session2 = splitted_ds["session_E"]  # testing only
        
        # Load session 1 data
        X = np.concatenate([run.windows.load_data()._data for run in session1.datasets], axis=0)
        y = np.concatenate([run.y for run in session1.datasets], axis=0)

        # Split session 1: 80% train, 20% validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.preprocessing_dict.get("seed", 42), stratify=y)

        # Load session 2 as test set
        X_test = np.concatenate([run.windows.load_data()._data for run in session2.datasets], axis=0)
        y_test = np.concatenate([run.y for run in session2.datasets], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X_train, X_val, X_test = BaseDataModule._z_scale_tvt(X_train, X_val, X_test)

        # Create datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
        # self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="train")
        # self.val_dataset   = BaseDataModule._make_tensor_dataset(X_val, y_val, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="val")
        # self.test_dataset  = BaseDataModule._make_tensor_dataset(X_test, y_test, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="test")

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True,
                        #   persistent_workers=True,          # ↩︎ keeps workers alive between epochs
                        #   prefetch_factor=4                 # ↩︎ each worker preloads 4 future batches                          
                        )


class BCICIV2aLOSO(BCICIV2a):
    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=self.all_subject_ids, dataset="2a",
                                  preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("subject")
        train_subjects = [
            subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]
        train_datasets = [splitted_ds[str(subj_id)].split("session")["session_T"]
                            for subj_id in train_subjects]
        val_datasets = [splitted_ds[str(subj_id)].split("session")["session_E"]
                        for subj_id in train_subjects]
        test_dataset = splitted_ds[str(self.subject_id)].split("session")["session_E"]

        # load the data
        X = np.concatenate([run.windows.load_data()._data for train_dataset in
                            train_datasets for run in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for train_dataset in train_datasets for run in
                            train_dataset.datasets], axis=0)
        X_val = np.concatenate([run.windows.load_data()._data for val_dataset in
                            val_datasets for run in val_dataset.datasets], axis=0)
        y_val = np.concatenate([run.y for val_dataset in val_datasets for run in
                            val_dataset.datasets], axis=0)
        X_test = np.concatenate([run.windows.load_data()._data for run in test_dataset.datasets],
                                axis=0)
        y_test = np.concatenate([run.y for run in test_dataset.datasets], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

        # self.train_dataset = BaseDataModule._make_tensor_dataset(X, y, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="train")
        # self.val_dataset   = BaseDataModule._make_tensor_dataset(X_val, y_val, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="val")
        # self.test_dataset  = BaseDataModule._make_tensor_dataset(X_test, y_test, 
        #                                                          preprocessing_dict=self.preprocessing_dict, mode="test")

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True,
                        #   persistent_workers=True,          # ↩︎ keeps workers alive between epochs
                        #   prefetch_factor=4                 # ↩︎ each worker preloads 4 future batches                          
                        )
