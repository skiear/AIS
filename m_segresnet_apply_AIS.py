# updated by Tang 2024.3
import glob
import os
import torch
from pathlib import Path
import torchio as tio
from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
)
from monai.utils import set_determinism
from scipy import ndimage


def fill_holes(volume):
    return ndimage.median_filter(volume, size=2)


roi_size = 128
device = torch.device("cuda:1")
set_determinism(seed=0)

root_dir = '/'
file_dir = root_dir+'Data'
model_dir = os.path.join(root_dir, 'models')

cmodelname = os.path.join(model_dir, "FDA_c2_128_segresnet.pth")
curname = '_seg2.nii.gz'
sourcename = "_FDA_mas_reg.nii.gz"
VAL_AMP = True
img_files = sorted(glob.glob(os.path.join(file_dir, "*/*" + sourcename)))
testlist = []
for img in img_files:
    new_file = img.replace(sourcename, curname)
    if not Path(new_file).exists():
        testlist.append({"image": img})
    else:
        print(f'File {new_file} already exists.')


print(f'共有图像{len(img_files)}待分割图像{len(testlist)}')

model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=3,
    out_channels=2,
    dropout_prob=0.2,
).to(device)

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


def inference(input):
    with torch.cuda.amp.autocast(enabled=VAL_AMP):
        return sliding_window_inference(
            inputs=input,
            roi_size=(roi_size, roi_size, roi_size),
            sw_batch_size=8,
            predictor=model,
            overlap=0.75,
        )


val_org_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ]
)
post_transforms = Compose([
    EnsureTyped(keys="pred"),
    Invertd(
        keys="pred",
        transform=val_org_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
    ),
    Activationsd(keys="pred", sigmoid=True),
    AsDiscreted(keys="pred", threshold=0.5),
])

model.load_state_dict(torch.load(cmodelname, map_location=device))
model.eval()

for testimg in testlist:
    newfilename = testimg["image"].replace(sourcename, curname)
    print(newfilename)
    val_org_ds = CacheDataset(
        data=[{"image": testimg["image"]}],
        transform=val_org_transforms,
        cache_num=1,
        cache_rate=1.0,
        num_workers=0,
    )
    val_org_loader = DataLoader(
        val_org_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
    )

    with torch.no_grad():
        for val_data in val_org_loader:
            val_inputs = val_data["image"].to(device)
            val_data["pred"] = inference(val_inputs)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs = from_engine(["pred"])(val_data)

            image_data = tio.ScalarImage(testimg["image"])
            mask = image_data.data[0]
            mask[:] = 0
            for jjj in range(2):
                temp = val_outputs[0][jjj].unsqueeze(0)
                mask[temp[0] > 0] = jjj + 1
            image_data.data = fill_holes(mask.unsqueeze(0))
            image_data.save(newfilename)
    print('处理完成:', testimg["image"])
