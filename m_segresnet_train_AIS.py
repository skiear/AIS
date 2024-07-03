import glob,os,time,json,shutil
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
from datetime import datetime

# 设置参数和路径
roi_size = 128
set_determinism(seed=0)
VAL_AMP = True
max_epochs = 1000
val_interval = 10
device = torch.device("cuda:0")
root_dir = r''
file_dir = os.path.join(root_dir, 'Seg_360subs')
model_dir = os.path.join(root_dir, 'models')
modelname = os.path.join(model_dir, "FDA_c2_128_segresnet.pth")
img_files = sorted(glob.glob(os.path.join(file_dir, "*/*_FDA_mas_reg.nii.gz")))
lb_files = sorted(glob.glob(os.path.join(file_dir, "*/*_Label.nii.gz")))
pre_trained = 1
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename =model_dir+ '/' + f"training_results_{current_time}.json"

if pre_trained > 0:
    tempmodelname=model_dir+ '/' + f"FDA_c2_128_segresnet_training_results_{current_time}.pth"
    shutil.copyfile(modelname, tempmodelname)

# 数据集分割
trainlist = [{"image": img_files[i], "label": lb_files[i]} for i in range(260)]
vallist = [{"image": img_files[i], "label": lb_files[i]} for i in range(260, 320)]

class ConvertToMultiChannelBasedOnMultiClassesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = [d[key] == 1, d[key] == 2]
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d

train_transform = Compose([
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
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    EnsureTyped(keys=["image", "label"]),
])

val_transform = Compose([
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
])

# 创建数据集和数据加载器
train_ds = CacheDataset(data=trainlist, transform=train_transform, cache_rate=1.0, num_workers=0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

val_ds = CacheDataset(data=vallist, transform=val_transform, cache_rate=1.0, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

# 初始化模型、损失函数、优化器和学习率调度器
model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=3,
    out_channels=2,
    dropout_prob=0.2
).to(device)

if pre_trained == 1:
    model.load_state_dict(torch.load(modelname, map_location=device))

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

dice_metric = DiceMetric(include_background=True, reduction="mean")
post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# 推理函数
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.8,
        )
    with torch.cuda.amp.autocast(enabled=VAL_AMP):
        return _compute(input)

# 设置混合精度训练和cuDNN优化
scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True

# 训练过程
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
total_start = time.time()
dice_history=[]
loss_history=[]
for epoch in range(max_epochs):
    epoch_start = time.time()
    model.train()
    epoch_loss = 0

    for step, batch_data in enumerate(train_loader, start=1):
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    lr_scheduler.step()
    epoch_loss /= step

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), modelname)
                print(f"Saved new best metric model: {modelname}")
            print(f"Current Epoch: {epoch + 1} Current Mean Dice: {metric:.4f}")
            dice_history.append(metric)
            loss_history.append(epoch_loss)

print(f"Total Training Time: {(time.time() - total_start):.4f} seconds")
results = {
    'loss_history': loss_history,
    'dice_history': dice_history
}
with open(filename, 'w') as f:
    json.dump(results, f)
print(f"Results saved to {filename}")
