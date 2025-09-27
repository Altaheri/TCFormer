from pathlib import Path
import os

import mne
import numpy as np
from scipy.io import loadmat


DATA_PATH = os.path.join(Path(__file__).resolve().parents[1], "data")


def load_bcic3_data_per_subject(subject_id: str, preprocessing_dict: dict):
    path = os.path.join(DATA_PATH, f"data_set_IVa_{subject_id}_mat", "100Hz", f"data_set_IVa_{subject_id}.mat")
    label_path = os.path.join(DATA_PATH, f"true_labels_{subject_id}.mat")
    mat = loadmat(path)
    mat_labels = loadmat(label_path)
    data = mat["cnt"]
    marker = mat["mrk"][0][0][0]
    labels = mat_labels["true_y"]
    test_idx = mat_labels["test_idx"]

    name = mat["nfo"]["name"][0][0][0]
    sfreq = mat["nfo"]["fs"][0][0][0][0]
    ch_names = [_[0] for _ in mat["nfo"]["clab"][0][0][0]]

    info = mne.create_info(ch_names, ch_types=["eeg"]*118, sfreq=sfreq)
    info["description"] = name

    raw = mne.io.RawArray(data.T*0.1, info)
    raw.resample(preprocessing_dict["sfreq"])
    raw.filter(l_freq=preprocessing_dict["low_cut"], h_freq=preprocessing_dict["high_cut"])

    channels = ["C3", "Cz", "C4"]
    channel_selection = preprocessing_dict.get("channel_selection", False)
    if channel_selection:
        raw.pick_channels(channels)

    start = int(preprocessing_dict["sfreq"] * preprocessing_dict["start"])
    stop = int(preprocessing_dict["sfreq"] * preprocessing_dict["stop"])
    trial_length = int(preprocessing_dict["sfreq"] * 3.5) - start + stop
    trials = np.zeros((labels.shape[-1], raw._data.shape[0], trial_length))
    for i, m in enumerate(marker[0]):
        trials[i] = raw._data[:, m+start:m+start+trials.shape[-1]]

    trials_dict = {
        "train": trials[:test_idx[0, 0]-1],
        "test": trials[test_idx[0]-1]
    }
    labels_dict = {
        "train": labels[0, :test_idx[0, 0] - 1] - 1,
        "test": labels[0, test_idx[0] - 1] - 1
    }

    return trials_dict, labels_dict


def load_bcic3(subject_ids=[1], prepr_dict=None):
    lookup_dict = {
        "1": "aa",
        "2": "al",
        "3": "av",
        "4": "aw",
        "5": "ay",
    }
    data, labels = {}, {}
    for subject_id in subject_ids:
        subject_id_translated = lookup_dict[str(subject_id)]
        data[str(subject_id)], labels[str(subject_id)] = load_bcic3_data_per_subject(
            subject_id_translated, prepr_dict
        )
    return {"data": data, "labels": labels}
