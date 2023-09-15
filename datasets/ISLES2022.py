from .base import BaseDataset
import os
from typing import Sequence
from monai.data import DataLoader
from monai.data.dataset import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Resized,
    NormalizeIntensityd,
    ConcatItemsd,
    AsDiscreted,
    CropForegroundd,
    RandSpatialCropd,
    RandShiftIntensityd,
    DeleteItemsd,
    ScaleIntensityRangePercentilesd,
    RandScaleCropd,
    RandRotated,
    RandFlipd,
    RandAxisFlipd,
    ToTensord,
    AsDiscreted,
    RandCropByPosNegLabeld,
)
from sklearn.model_selection import train_test_split


class ISLES2022(BaseDataset):
    def __init__(
        self,
        isles_data_dir: str,
        image_size: Sequence[int],
        totalnumber: int = 250,
        test_size: float = 0.3,
    ) -> None:
        self.data = []
        for i in range(1, totalnumber + 1):
            dwi_path = os.path.join(
                isles_data_dir,
                "rawdata",
                "sub-strokecase{}".format("%04d" % i),
                "ses-0001",
                "sub-strokecase{}_ses-0001_dwi.nii.gz".format("%04d" % i),
            )
            adc_path = dwi_path.replace("dwi", "adc")
            flair_path = dwi_path.replace("dwi", "flair")
            mask_path = dwi_path.replace("rawdata", "derivatives").replace("dwi", "msk")

            single = {
                "dwi": dwi_path,
                "adc": adc_path,
                "flair": flair_path,
                "label": mask_path,
            }
            self.data.append(single)

        # TODO: split train and test
        self.train_data, self.val_data = train_test_split(
            self.data, test_size=test_size
        )

        image_key, x_key, y_key = "image", ["dwi", "adc"], ["label"]

        self.train_transform = Compose(
            [
                LoadImaged(x_key + y_key),
                EnsureChannelFirstd(x_key + y_key),
                ConcatItemsd(x_key, image_key),
                DeleteItemsd(x_key),  # 为了后面能concat
                Spacingd([image_key] + y_key, pixdim=(1, 1, 1)),
                ScaleIntensityRangePercentilesd(
                    image_key, lower=5, upper=95, b_min=0.0, b_max=255.0
                ),
                CropForegroundd([image_key] + y_key, image_key),
                RandCropByPosNegLabeld(
                    [image_key] + y_key,
                    label_key=y_key[0],
                    spatial_size=image_size,
                    num_samples=2,
                ),
                # RandScaleCropd([image_key]+y_key,roi_scale=0.4),
                RandAxisFlipd([image_key] + y_key, prob=0.5),
                RandFlipd([image_key] + y_key, prob=0.5),
                RandRotated([image_key] + y_key, prob=0.5),
                ToTensord(keys=[image_key] + y_key),
            ]
        )

        self.val_transform = Compose(
            [
                LoadImaged(x_key + y_key),
                EnsureChannelFirstd(x_key + y_key),
                ConcatItemsd(x_key, image_key),
                DeleteItemsd(x_key),  # 为了后面能concat
                Spacingd([image_key] + y_key, pixdim=(1, 1, 1)),
                ScaleIntensityRangePercentilesd(
                    image_key, lower=5, upper=95, b_min=0.0, b_max=255.0
                ),
                CropForegroundd([image_key] + y_key, image_key),
                ToTensord(keys=[image_key] + y_key),
            ]
        )

    def get_train_loader(
        self, batch_size: int = 8, num_workers: int = 0, shuffle: bool = True, **kwargs
    ):
        # TODO: train data
        train_dataset = Dataset(self.train_data, self.train_transform)
        return DataLoader(
            train_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )

    def get_val_loader(
        self, batch_size: int = 8, num_workers: int = 0, shuffle: bool = True, **kwargs
    ):
        # TODO: val data
        val_dataset = Dataset(self.val_data, self.val_transform)
        return DataLoader(
            val_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
