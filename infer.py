import os
from utils import load_weight, get_config
from datasets import ISLES2022
from monai.networks.nets import UNETR
from monai.metrics import DiceMetric, HausdorffDistanceMetric, compute_roc_auc
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
)
from monai.data.utils import decollate_batch
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
import torch
from tqdm import tqdm
from monai.inferers import sliding_window_inference
import shutil
import numpy as np
import matplotlib.pyplot as plt
from accelerate import Accelerator
from thop import profile
from sklearn.metrics import roc_curve

join = os.path.join
# 加速
accelerator = Accelerator()

work_dir = "./work_dir"
os.makedirs(work_dir, exist_ok=True)
if accelerator.is_local_main_process:
    model_save_path = join(work_dir, f"exp{len(os.listdir(work_dir)) + 1}")
    os.makedirs(model_save_path, exist_ok=True)

config = get_config()
image_sizes = config.image_size
batch_size=config.batch_size

dataset = ISLES2022(config.data_path, image_size=image_sizes)

val_dataloader = dataset.get_val_loader(batch_size=batch_size)

# model
model = UNETR(2, 2, image_sizes)

assert config.resume_path != None, "resume path can't be none"

model = load_weight(model, config.resume_path)
print(f"load weight from {config.resume_path}")
filename = os.path.basename(config.resume_path)
shutil.copyfile(config.resume_path,join(model_save_path,filename))

cpu = torch.device("cpu")
device = accelerator.device

model,val_dataloader = accelerator.prepare(model,val_dataloader)

# metric
metrics = {"dice": DiceMetric(), "hausdorff": HausdorffDistanceMetric()}
# loss function
loss_func = DiceFocalLoss(to_onehot_y=True, softmax=True)
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(threshold=0.5), AsDiscrete(to_onehot=2)])

model.to(device)
model.eval()
for metric in metrics.values():
    metric.reset()
val_loss = 0


# TFLOPS
input = torch.randn(2, 2, 96, 96, 96).to(device)
macs, params = profile(model, inputs=(input, ))
print(f"MACs: {macs / 10e12}T, TFLOPs: {2 * macs / 10e12}, Params: {params / 10e6}M")



with torch.no_grad():
    with tqdm(val_dataloader,unit="batch",disable=not accelerator.is_local_main_process) as tepoch:
        for step,batch_data in enumerate(tepoch):
            tepoch.set_description("val")
            image, label = batch_data["image"].to(device), batch_data["label"].to(device)
            # 推理
            output = sliding_window_inference(
                image, image_sizes, sw_batch_size=batch_size, predictor=model
            )
            outputs = output.flatten().cpu()

            loss = loss_func(output, label)
            output = [post_pred(i) for i in decollate_batch(output)]
            label = [post_label(i) for i in decollate_batch(label)]

            labels = torch.cat(label).flatten().cpu()
            auc_value = compute_roc_auc(outputs, labels)
            fpr, tpr, thresholds = roc_curve(labels, outputs)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc_value)
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Segmentation Model')
            plt.legend(loc="lower right")
            plt.show()

            for metric in metrics.values():
                metric(output, label)
            val_loss += loss.item()
            metric_result = {}
            for name, metric in metrics.items():
                metric_result[name] = metric.aggregate().item()
            tepoch.set_postfix(
                loss=val_loss / (step + 1), **metric_result          )

            if accelerator.is_local_main_process and step % 10 == 0:
                # save image
                b,c,w,h,d = image.shape
                sample_w = np.random.randint(0,w)
                sample_b = np.random.randint(0,b)

                image,output,label = image.to(cpu),output[sample_b].to(cpu),label[sample_b].to(cpu)

                for i in range(w):
                    if label[1,i,:].sum() > 100:
                        sample_w = i
                        break
                sample_image = image[sample_b,0,sample_w,:]
                sample_output = output[1,sample_w,:]
                sample_label = label[1,sample_w,:]

                plt.subplot(1,2,1)
                plt.title("label")
                plt.imshow(sample_image, cmap='gray')
                plt.imshow(sample_label,cmap='Reds',alpha=0.5)

                plt.subplot(1,2,2)
                plt.title("predict")
                plt.imshow(sample_image, cmap='gray')
                plt.imshow(sample_output,cmap='Reds',alpha=0.5)

                plt.savefig(join(model_save_path,f'step_{step}.png'))
        
        val_loss /= step + 1
        val_dice = metrics['dice'].aggregate().item()
        val_hausdorff = metrics['hausdorff'].aggregate().item()
        
        if accelerator.is_local_main_process:
            print(f"val loss: {val_loss}, val dice: {val_dice}, val hausdorff: {val_hausdorff}")
            # ROC curve
            # x2 = output.flatten().cpu()
            # auc_value = compute_roc_auc(x2, label[0].flatten().cpu())
            # fpr, tpr, thresholds = roc_curve(label[0].flatten().cpu(), x2)
            # labels = torch.cat(labels).flatten().cpu()
            # outputs = torch.cat(outputs).flatten().cpu()
            # auc_value = compute_roc_auc(outputs, labels)
            # fpr, tpr, thresholds = roc_curve(labels, outputs)
            # plt.figure()
            # plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc_value)
            # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('ROC Curve for Segmentation Model')
            # plt.legend(loc="lower right")
            # plt.show()