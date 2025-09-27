from typing import Dict

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
    scale,
)


def load_bcic4(subject_ids: list, dataset: str = "2a", preprocessing_dict: Dict = None,
              verbose: str = "WARNING"):
    dataset_name = "BNCI2014001" if dataset == "2a" else "BNCI2014004"
    dataset = MOABBDataset(dataset_name, subject_ids=subject_ids)

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=verbose),
        Preprocessor(scale, factor=1e6, apply_on_array=True),
        Preprocessor("resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose)
    ]

    l_freq, h_freq = preprocessing_dict["low_cut"], preprocessing_dict["high_cut"]
    if l_freq is not None or h_freq is not None:
        preprocessors.append(Preprocessor("filter", l_freq=l_freq, h_freq=h_freq,
                                          verbose=verbose))

    preprocess(dataset, preprocessors)

    # create windows
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(preprocessing_dict["start"] * sfreq)
    trial_stop_offset_samples = int(preprocessing_dict["stop"] * sfreq)
    windows_dataset = create_windows_from_events(
        dataset, trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples, preload=False
    )

    return windows_dataset
