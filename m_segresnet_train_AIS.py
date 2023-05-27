import glob
import os
import time
import torch

import numpy as np
from monai.data import DataLoader, decollate_batch, CacheDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByPosNegLabeld,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
)
from monai.utils import set_determinism
roi_size=128
set_determinism(seed=0)
device = torch.device("cuda:1")
root_dir=''####add loacal path
file_dir=root_dir+'Data/'
model_dir=root_dir+'models/'
cmodelname=os.path.join(model_dir,"FDB_c2_128_segresnet.pth")
img_files = sorted(glob.glob(os.path.join(file_dir, "*/*_FDB_msk.nii.gz")))
lb_files = sorted(glob.glob(os.path.join(file_dir,  "*/*_label.nii.gz")))
####################2022训练###########
havvv=0
omodelname=cmodelname
trainlist=[]
vallist=[]
for i in range(0,60):
    trainlist.append({"image": img_files[i],"label": lb_files[i]})

for i in range(60,80):
    vallist.append({"image": img_files[i],"label": lb_files[i]})


class ConvertToMultiChannelBasedOnMultiClassesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(d[key] == 1)
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d

train_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnMultiClassesd(keys="label"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(roi_size, roi_size, roi_size),
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        EnsureTyped(keys=["image", "label"]),
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnMultiClassesd(keys="label"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(roi_size, roi_size, roi_size),
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ]
)

train_ds = CacheDataset(
    data=trainlist, #文件列表
    transform=train_transform, #定义变换
    cache_num=4,
    cache_rate=1.0,
    num_workers=0,
)
train_loader = DataLoader(
    train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True
)

val_ds = CacheDataset(
    data=vallist, #文件列表
    transform=val_transform, #定义变换
    cache_num=4,
    cache_rate=1.0,
    num_workers=0,
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
)
########Create Model, Loss, Optimizer
max_epochs = 5000
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
if havvv==1:
    model.load_state_dict(
        torch.load(omodelname)
    )

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

#
# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 155),
            sw_batch_size=1,
            predictor=model,
            overlap=0.8,
        )
    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

#
# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

#######################Execute a typical PyTorch training process
best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_c = []
metric_values_p = []
metric_values_d = []

total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device), #######
            batch_data["label"].to(device), #######
        )
        dim = inputs.shape
        inputs = inputs.reshape(dim[0], dim[1], dim[2], dim[3], dim[4])
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)  #######
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                dim = val_inputs.shape
                val_inputs = val_inputs.reshape(dim[0], dim[1], dim[2], dim[3], dim[4])
                val_outputs = inference(val_inputs)
                # print(val_output.shape)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    cmodelname,
                )
                print("saved new best metric model"+cmodelname)
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start