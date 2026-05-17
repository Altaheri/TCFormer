from models import (
    TCFormer,
    ATCNet,
    BaseNet,
    # SST_DPN,
    EEGConformer,
    EEGNet,
    EEGTCNet,
    ShallowNet,
    TSSEFFNet,
    CTNet,
    MSCFormer,
    EEGDeformer,
)


model_dict = dict(
    # SST_DPN=SST_DPN,
    TCFormer=TCFormer,
    ATCNet=ATCNet,
    BaseNet=BaseNet,
    EEGConformer=EEGConformer,
    EEGNet=EEGNet,
    EEGTCNet=EEGTCNet,
    ShallowNet=ShallowNet,
    TSSEFFNet=TSSEFFNet,
    CTNet = CTNet,
    MSCFormer = MSCFormer,
    EEGDeformer=EEGDeformer,
)


def get_model_cls(model_name):
    return model_dict[model_name]
