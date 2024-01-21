from utils import load_weight, get_config
from datasets import ISLES2022, ISLES2017
from monai.transforms import (
    Compose,
    LoadImaged,
    LoadImage,
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
import numpy as np
from collections import Counter



# dataset = ISLES2022("D:/datasets/ISLES/dataset-ISLES22^public^unzipped^version")
# lesionVolumes = []

# for data in dataset.data:
#     mask_path = data["label"]
#     loader = LoadImage(dtype=np.bool8)
#     label, metadata = loader(mask_path)    # 把label的地址传给loader

#     labelSize = label.size()
#     lesionVolume = (label==True).sum().item()

#     lesionVolumes.append({"mask_path": mask_path, "labelSize": labelSize, "lesionVolume": lesionVolume})


# volumesCount = [item["lesionVolume"] for item in lesionVolumes]







dataset = ISLES2017("D:/datasets/ISLES/ISLES2017/ISLES2017_Training")

lesionVolumes_2017 = []

for data in dataset.datas:
    mask_path = data["OT"]
    loader = LoadImage()
    label = loader(mask_path)    # 把label的地址传给loader

    labelSize = label.size()
    lesionVolume = (label==True).sum().item()

    lesionVolumes_2017.append({"mask_path": mask_path, "labelSize": labelSize, "lesionVolume": lesionVolume})


volumesCount_2017 = [item["lesionVolume"] for item in lesionVolumes_2017]








# 0-100 100-200 200-400 400-600 600-800 800-1000 1000-2000 2000+
cnt = [0 for i in range(8)]
for volume in volumesCount_2017:
    if volume <= 100:
        cnt[0] += 1
    elif volume <= 200:
        cnt[1] += 1
    elif volume <= 400:
        cnt[2] += 1
    elif volume <= 600:
        cnt[3] += 1
    elif volume <= 800:
        cnt[4] += 1
    elif volume <= 1000:
        cnt[5] += 1
    elif volume <= 2000:
        cnt[6] += 1
    else:
        cnt[7] += 1
    

print(cnt)




