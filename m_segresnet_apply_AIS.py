import glob
import os
import torch
from pathlib import Path
import torchio as tio
from monai.data import DataLoader, decollate_batch, CacheDataset
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
device = torch.device("cuda:0")
set_determinism(seed=0)
roi_size=128
root_dir='/'   ### set path
file_dir=root_dir+'TestData/'
model_dir=root_dir+'models/'
cmodelname=os.path.join(model_dir,"FDB_c2_segresnet.pth")
curname='_WMH_Core_128.nii.gz'
sourcename = "_FDB_msk.nii.gz"
#####
set_determinism(seed=0)
img_files = sorted(glob.glob(os.path.join(file_dir, "*/*"+sourcename)))
print(len(img_files))
testnum=len(img_files)
testlist = []
for i in range(0,len(img_files)):
    im=img_files[i]
    sub_name = im.split(sourcename)[-2]
    gth=Path(sub_name+curname)
    if gth.exists():
        print(im+'dddddddddddddddd')
    else:
        testlist.append({"image": img_files[i]})

print(len(testlist))

val_interval = 1
VAL_AMP = True
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=3,
    out_channels=2,
    dropout_prob=0.2,
).to(device)
#
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(roi_size, roi_size, roi_size),
            sw_batch_size=8,
            predictor=model,
            overlap=0.8,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

val_org_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ]
)

val_org_ds = CacheDataset(
    data=testlist, #文件列表
    transform=val_org_transforms, #定义变换
    cache_num=1,
    cache_rate=1.0,
    num_workers=0,
)
val_org_loader = DataLoader(
    val_org_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
)

######
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
model.load_state_dict(torch.load(cmodelname))
model.eval()

with torch.no_grad():
    for val_data in val_org_loader:
        val_inputs = val_data["image"].to(device)
        dim = val_inputs.shape
        val_inputs = val_inputs.reshape(dim[0], dim[1], dim[2], dim[3], dim[4])
        val_data["pred"] = inference(val_inputs)
        val_data = [post_transforms(i) for i in decollate_batch(val_data)]
        val_outputs = from_engine(["pred"])(val_data)
        filename = val_data[0]["image_meta_dict"]["filename_or_obj"]
        subname = filename.split(sourcename)[-2]
        print(subname)
        image_data=tio.ScalarImage(filename)
        mask=image_data.data[0,:,:,:]
        dim = mask.shape
        mask = mask.reshape(1, dim[0], dim[1], dim[2])
        mask[:,:,:,:]=0
        image_data.data = mask
        for jjj in range(2):
            temp = val_outputs[0][jjj, :, :, :]
            temp = temp.reshape(1, dim[0], dim[1], dim[2])
            image_data.data[temp>0]=jjj+1
        image_data.save(subname + curname)
print('DDDDDDDDDDDDDDDDDDDDDDDDD')
