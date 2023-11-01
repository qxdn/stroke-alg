from .base import BaseDataset
import os
from typing import Sequence, Callable
from monai.data import DataLoader
from monai.data.dataset import Dataset
from monai.data.utils import pad_list_data_collate
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


class ISLES2017(BaseDataset):
    def __init__(
        self,
        isles_data_dir: str = "D:/Code/ML_ISLES2017/data/ISLES/train",
        image_size: Sequence[int] = (96, 96, 16),
        test_size: float = 0.3,
    ) -> None:
        root_folders: list[str] = os.listdir(isles_data_dir)
        self.datas = []
        for root_folder in root_folders:
            root_folder = os.path.join(isles_data_dir, root_folder)
            folders = os.listdir(root_folder)
            data = {}
            for folder in folders:
                data_type = folder.split(".")[4]
                nii_file = os.path.join(root_folder, folder, folder + ".nii")
                data[data_type] = nii_file

            self.datas.append(data)

        self.train_data, self.val_data = train_test_split(
            self.datas, test_size=test_size
        )

        x_key, label_key = [
            "MR_ADC",
            "MR_MTT",
            "MR_rCBF",
            "MR_rCBV",
            "MR_Tmax",
            "MR_TTP",
        ], "OT"
        image_key = "image"

        self.train_transform = Compose(
            [
                LoadImaged(keys=x_key + [label_key]),
                EnsureChannelFirstd(keys=x_key + [label_key]),
                ConcatItemsd(keys=x_key, name=image_key),
                RandCropByPosNegLabeld(
                    keys=[image_key, label_key],
                    label_key=label_key,
                    spatial_size=image_size,
                    num_samples=2,
                ),
                NormalizeIntensityd(keys=[image_key]),
                RandRotated(
                    keys=[image_key, label_key],
                    range_x=180,
                    range_y=180,
                    range_z=180,
                    prob=0.5,
                ),
                ToTensord(keys=[image_key, label_key]),
            ]
        )

        self.val_transform = Compose(
            [
                LoadImaged(keys=x_key + [label_key]),
                EnsureChannelFirstd(keys=x_key + [label_key]),
                ConcatItemsd(keys=x_key, name=image_key),
            ]
        )

    def get_train_dataset(self):
        return Dataset(self.train_data, self.train_transform)

    def get_train_loader(
        self, batch_size: int = 8, num_workers: int = 0, shuffle: bool = True, **kwargs
    ):
        train_dataset = Dataset(self.train_data, self.train_transform)
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs
        )

    def get_val_loader(
        self, batch_size: int = 8, num_workers: int = 0, shuffle: bool = True, **kwargs
    ):
        test_dataset = Dataset(self.val_data, self.val_transform)
        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs
        )
