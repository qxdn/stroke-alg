import os
from utils import load_weight, get_config
from datasets import ISLES2022
from monai.networks.nets import UNETR
from monai.metrics import DiceMetric
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

join = os.path.join

work_dir = "./work_dir"
os.makedirs(work_dir, exist_ok=True)
model_save_path = join(work_dir, f"exp{len(os.listdir(work_dir)) + 1}")
os.makedirs(model_save_path, exist_ok=True)

config = get_config()
image_sizes = config.image_size
batch_size=config.batch_size

dataset = ISLES2022(config.data_path, image_size=image_sizes)

val_dataloader = dataset.get_val_loader(batch_size=batch_size)

# model
from testmodel import NN
model = NN(2,2)
#model = UNETR(2, 2, image_sizes)

assert config.resume_path != None, "resume path can't be none"

#model = load_weight(model, config.resume_path)
print(f"load weight from {config.resume_path}")
filename = os.path.basename(config.resume_path)
shutil.copyfile(config.resume_path,join(model_save_path,filename))

cpu = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else cpu


# metric
metric = DiceMetric()
# loss function
loss_func = DiceFocalLoss(to_onehot_y=True, softmax=True)
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = AsDiscrete(to_onehot=2)

model.to(device)
model.eval()
metric.reset()
val_loss = 0

with tqdm(val_dataloader,unit="batch") as tepoch:
    for step,batch_data in enumerate(tepoch):
        tepoch.set_description("val")
        image, label = batch_data["image"].to(device), batch_data["label"].to(device)
        # 推理
        output = sliding_window_inference(
            image, image_sizes, sw_batch_size=batch_size, predictor=model
        )
        loss = loss_func(output, label)
        output = [post_pred(i) for i in decollate_batch(output)]
        label = [post_label(i) for i in decollate_batch(label)]
        metric(output, label)
        val_loss += loss.item()
        tepoch.set_postfix(
            loss=val_loss / (step + 1), dice=metric.aggregate().item()
        )

        if step % 10 == 0:
            
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
    val_dice = metric.aggregate().item()