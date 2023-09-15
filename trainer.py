import torch
from datasets import ISLES2022
from tqdm import tqdm
from accelerate import Accelerator
import os
import monai
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
)
from monai.data.utils import decollate_batch
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from torch.utils.tensorboard import SummaryWriter
from utils import set_seed

join = os.path.join
# 加速
accelerator = Accelerator()

work_dir = "./work_dir"
set_seed()
os.makedirs(work_dir, exist_ok=True)
if accelerator.is_local_main_process:
    model_save_path = join(work_dir, f"exp{len(os.listdir(work_dir)) + 1}")
    writer = SummaryWriter(model_save_path)
    os.makedirs(model_save_path, exist_ok=True)

image_sizes = (96, 96, 96)
batch_size = 1
epochs = 200


dataset = ISLES2022(
    "D:/datasets/ISLES/dataset-ISLES22^public^unzipped^version",
    image_size=image_sizes,
)
train_dataloader = dataset.get_train_loader(batch_size=batch_size)
val_dataloader = dataset.get_val_loader(batch_size=batch_size)


# model
# model = UNETR(2, 2, image_sizes)
from testmodel import NN

model = NN(2, 2)
print(model)
# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# device
cpu = torch.device("cpu")
device = accelerator.device
model.to(device)
# train
model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader
)
# loss function
loss_func = DiceCELoss(to_onehot_y=True, softmax=True)
# metric
metric = DiceMetric()
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = AsDiscrete(to_onehot=2)

best_dice = 0
for epoch in range(epochs):
    # train
    train_epoch_loss = 0
    model.train()
    metric.reset()
    with tqdm(
        train_dataloader, unit="batch", disable=not accelerator.is_local_main_process
    ) as tepoch:
        for step, batch_data in enumerate(tepoch):
            tepoch.set_description(f"Train Epoch {epoch}")
            optimizer.zero_grad()
            image, label = batch_data["image"].to(device), batch_data["label"].to(
                device
            )
            output = model(image)
            loss = loss_func(output, label)
            accelerator.backward(loss)
            optimizer.step()
            train_epoch_loss += loss.item()
            # metric
            output = [post_pred(i) for i in decollate_batch(output)]
            label = [post_label(i) for i in decollate_batch(label)]
            metric(output, label)
            tepoch.set_postfix(
                loss=train_epoch_loss / (step + 1), dice=metric.aggregate().item()
            )

        train_epoch_loss /= step + 1
        train_dice = metric.aggregate().item()

        if accelerator.is_local_main_process:
            writer.add_scalar("train/loss", train_epoch_loss, epoch)
            writer.add_scalar("train/dice", train_dice, epoch)

    # val
    val_epoch_loss = 0
    model.eval()
    metric.reset()
    with torch.no_grad():
        with tqdm(
            val_dataloader,
            unit="batch",
            disable=not accelerator.is_local_main_process,
        ) as tepoch:
            for step, batch_data in enumerate(tepoch):
                tepoch.set_description(f"Val Epoch {epoch}")
                image, label = batch_data["image"].to(device), batch_data["label"].to(
                    device
                )
                # 推理
                output = sliding_window_inference(
                    image, image_sizes, sw_batch_size=batch_size, predictor=model
                )
                loss = loss_func(output, label)
                output = [post_pred(i) for i in decollate_batch(output)]
                label = [post_label(i) for i in decollate_batch(label)]
                metric(output, label)
                val_epoch_loss += loss.item()
                tepoch.set_postfix(
                    loss=val_epoch_loss / (step + 1), dice=metric.aggregate().item()
                )

            val_epoch_loss /= step + 1
            val_dice = metric.aggregate().item()

            if accelerator.is_local_main_process:
                writer.add_scalar("val/loss", val_epoch_loss, epoch)
                writer.add_scalar("val/dice", val_dice, epoch)

    # save model
    if accelerator.is_local_main_process:
        unwrap_model = accelerator.unwrap_model(model)
        # save epoch model
        #torch.save(
        #    unwrap_model.state_dict(), join(model_save_path, f"epoch_{epoch}.pth")
        #)
        # save latest model
        torch.save(unwrap_model.state_dict(), join(model_save_path, "latest_model.pth"))
        # save best model
        if best_dice < val_dice:
            best_dice = val_dice
            torch.save(
                unwrap_model.state_dict(), join(model_save_path, "best_model.pth")
            )
