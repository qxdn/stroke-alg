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
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from tensorboardX import SummaryWriter
from utils import set_seed, load_weight, get_config
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from nets import UNETRET, CAFormerUnet, SimpleCAUnet, CAFormerPolyUnet
import wandb

join = os.path.join
# 加速
accelerator = Accelerator(log_with="wandb")
# config
config = get_config()
epochs = config.epochs
batch_size = config.batch_size
image_sizes = config.image_size
set_seed(config.seed)


work_dir = "./work_dir"
os.makedirs(work_dir, exist_ok=True)
exp_name = f"exp{len(os.listdir(work_dir)) + 1}"
model_save_path = join(work_dir, exp_name)
if accelerator.is_local_main_process:
    writer = SummaryWriter(model_save_path)
    os.makedirs(model_save_path, exist_ok=True)

# wandb
accelerator.init_trackers(
    project_name="stroke-segmentation",
    config=config,
    init_kwargs={
        "wandb": {
            "name": exp_name,
            "dir": model_save_path,
        }
    },
)


dataset = ISLES2022(
    config.data_path,
    image_size=image_sizes,
)
train_dataloader = dataset.get_train_loader(batch_size=batch_size)
val_dataloader = dataset.get_val_loader(batch_size=batch_size)


# model
# model = UNETR(2, 2, image_sizes)
# model = UNETRET(2, 2, image_sizes)
# from testmodel import NN
# model = NN(2, 2)
# model = CAFormerUnet(2,3,depths=(3,3,9,3),drop_path_rate=0.5,add=False)
# model = SimpleCAUnet(2, drop_path_rate=0.5)
model = CAFormerPolyUnet(2, drop_path_rate=0.5)

if config.resume_path != None:
    model = load_weight(model, config.resume_path)
    print("load weight from {}".format(config.resume_path))

print(model)
from torchinfo import summary

summary(model, (1, 2, *image_sizes), device="cpu")
# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# scheduler
scheduler = WarmupCosineSchedule(
    optimizer,
    warmup_steps=config.warmup * accelerator.num_processes,
    t_total=epochs * accelerator.num_processes,
)

# device
cpu = torch.device("cpu")
device = accelerator.device
model.to(device)
# train
model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, scheduler, train_dataloader, val_dataloader
)
# loss function
loss_func = DiceFocalLoss(to_onehot_y=True, softmax=True)
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

        scheduler.step()
        train_epoch_loss /= step + 1
        train_dice = metric.aggregate().item()

        if accelerator.is_local_main_process:
            writer.add_scalar("train/loss", train_epoch_loss, epoch)
            writer.add_scalar("train/dice", train_dice, epoch)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # wandb
        accelerator.log(
            {
                "train/loss": train_epoch_loss,
                "train/dice": train_dice,
                "train/lr": scheduler.get_last_lr()[0],
            },
            step=epoch,
        )

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

            # wandb
            accelerator.log(
                {"val/loss": val_epoch_loss, "val/dice": val_dice}, step=epoch
            )

    # save model
    if accelerator.is_local_main_process:
        unwrap_model = accelerator.unwrap_model(model)
        # save epoch model
        # torch.save(
        #    unwrap_model.state_dict(), join(model_save_path, f"epoch_{epoch}.pth")
        # )
        # save latest model
        accelerator.save(
            unwrap_model.state_dict(), join(model_save_path, "latest_model.pth")
        )
        # save best model
        if best_dice < val_dice:
            best_dice = val_dice
            accelerator.save(
                unwrap_model.state_dict(), join(model_save_path, "best_model.pth")
            )


accelerator.end_training()
