"""
Data load with pre-processing
"""

import monai.transforms as mt
from copy import deepcopy


def pre_transforms(crop_size, resample_spacing):

    train_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"]), # load image
            mt.EnsureChannelFirstd(keys=["image", "label"]),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"), # adjust orientation (RAS)
            mt.Spacingd(keys=["image", "label"], pixdim=(resample_spacing[0], resample_spacing[1],resample_spacing[2]), mode=("bilinear", "nearest")),
            mt.ScaleIntensityRangePercentilesd(keys="image",lower=10, upper=90, b_min=0, b_max=1, relative=True, channel_wise=True),
            mt.RandCropByPosNegLabeld(keys=["image", "label"], label_key = "label", spatial_size=crop_size, pos=1.0, neg=0.0, num_samples=1),
            mt.ToTensord(keys=["image", "label"]),
        ]
    )
    
    val_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"]), 
            mt.EnsureChannelFirstd(keys=["image", "label"]),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"), 
            mt.Spacingd(keys=["image", "label"], pixdim=(resample_spacing[0], resample_spacing[1],resample_spacing[2]), mode=("bilinear", "nearest")),
            mt.ScaleIntensityRangePercentilesd(keys="image",lower=10, upper=90, b_min=0, b_max=1, relative=True, channel_wise=True),
            mt.RandCropByPosNegLabeld(keys=["image", "label"], label_key = "label", spatial_size=crop_size, pos=1.0, neg=0.0, num_samples=1),
            mt.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"]), 
            mt.EnsureChannelFirstd(keys=["image", "label"]),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"), 
            mt.Spacingd(keys=["image", "label"], pixdim=(resample_spacing[0], resample_spacing[1],resample_spacing[2]), mode=("bilinear", "nearest")),
            mt.ScaleIntensityRangePercentilesd(keys="image",lower=10, upper=90, b_min=0, b_max=1, relative=True, channel_wise=True),
            mt.ToTensord(keys=["image", "label"]),
        ]
    )

    return train_transform, val_transform, test_transform

