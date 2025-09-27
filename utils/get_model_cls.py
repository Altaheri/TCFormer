from models import (
    TCFormer,
    ATCNet,
    ATCNet_2_0,
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
    ATCNet_2_0 = ATCNet_2_0,
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
