from datamodules import BCICIII_IVa, BCICIII_IVaLOSO, BCICIV2a, BCICIV2aTVT,\
    BCICIV2aLOSO, BCICIV2b, BCICIV2bLOSO, HighGamma, HighGammaLOSO, REH_MI


def get_datamodule_cls(dataset_name):
    if dataset_name == "bcic3":
        datamodule_cls = BCICIII_IVa
    elif dataset_name == "bcic3_loso":
        datamodule_cls = BCICIII_IVaLOSO
    elif dataset_name == "bcic2a":
        datamodule_cls = BCICIV2a
    elif dataset_name == "bcic2a_tvt":
        datamodule_cls = BCICIV2aTVT
    elif dataset_name == "bcic2a_loso":
        datamodule_cls = BCICIV2aLOSO
    elif dataset_name == "bcic2b":
        datamodule_cls = BCICIV2b
    elif dataset_name == "bcic2b_loso":
        datamodule_cls = BCICIV2bLOSO
    elif dataset_name == "hgd":
        datamodule_cls = HighGamma
    elif dataset_name == "hgd_loso":
        datamodule_cls = HighGammaLOSO
    elif dataset_name == "reh_mi":
        datamodule_cls = REH_MI
    else:
        raise NotImplementedError(f"No dataset with name: {dataset_name}")

    return datamodule_cls
