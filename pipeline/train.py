import os
import time
import datetime
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from loss.Charbonnier_Loss import L1_Charbonnier_loss as Charbonnier_Loss

from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision.transforms import Pad
import torchvision.transforms.functional as TF
import random

from natsort import natsorted
from glob import glob
from collections import OrderedDict

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_last_path(path, session):
    return natsorted(glob(os.path.join(path, '*%s' % session)))[-1]

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[10:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    print("MOdel loading successfully!")

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'trainA')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'trainB')))

        self.inp_filenames = [os.path.join(rgb_dir, 'trainA', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'trainB', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target_enh

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def aug_pad(self, aug_0, inp_img, tar_img_enh):
        if aug_0 == 1:  # left - top
            # pad
            h_pad = max(self.ps[0] - inp_img.shape[1], 0)
            w_pad = max(self.ps[1] - inp_img.shape[2], 0)
            pad = Pad((w_pad, h_pad, 0, 0))
            inp_img = pad(inp_img)
            tar_img_enh = pad(tar_img_enh)
        elif aug_0 == 2:
            # pad
            h_pad = max(self.ps[0] - inp_img.shape[1], 0)
            w_pad = max(self.ps[1] - inp_img.shape[2], 0)
            pad = Pad((0, h_pad, w_pad, 0))
            inp_img = pad(inp_img)
            tar_img_enh = pad(tar_img_enh)
        elif aug_0 == 3:
            # pad
            h_pad = max(self.ps[0] - inp_img.shape[1], 0)
            w_pad = max(self.ps[1] - inp_img.shape[2], 0)
            pad = Pad((w_pad, 0, 0, h_pad))
            inp_img = pad(inp_img)
            tar_img_enh = pad(tar_img_enh)
        elif aug_0 == 4:
            # pad
            h_pad = max(self.ps[0] - inp_img.shape[1], 0)
            w_pad = max(self.ps[1] - inp_img.shape[2], 0)
            pad = Pad((0, 0, w_pad, h_pad))
            inp_img = pad(inp_img)
            tar_img_enh = pad(tar_img_enh)
        return inp_img,tar_img_enh

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img_enh = Image.open(tar_path).convert('RGB')

        inp_img = TF.to_tensor(inp_img)
        tar_img_enh = TF.to_tensor(tar_img_enh)

        aug_0 = random.randint(0,1) if inp_img.shape == tar_img_enh.shape else 0
        if aug_0 == 0:
            # resize
            inp_img = TF.resize(inp_img, (self.ps[0], self.ps[1]))
            tar_img_enh = TF.resize(tar_img_enh, (self.ps[0], self.ps[1]))
        elif inp_img.shape[1] < self.ps[0] or inp_img.shape[2] < self.ps[1]:
            inp_img, tar_img_enh = self.aug_pad(aug_0, inp_img, tar_img_enh) # pad
            # randomcrop window
            h_rand = random.randint(0, tar_img_enh.shape[1] - self.ps[0])
            w_rand = random.randint(0, tar_img_enh.shape[2] - self.ps[1])
            inp_img = inp_img[:, h_rand :h_rand + self.ps[0], w_rand :w_rand + self.ps[1]]
            tar_img_enh = tar_img_enh[:, h_rand :h_rand + self.ps[0] , w_rand :w_rand + self.ps[1]]
        else:
            # randomcrop window
            h_rand = random.randint(0, tar_img_enh.shape[1] - self.ps[0])
            w_rand = random.randint(0, tar_img_enh.shape[2] - self.ps[1])
            inp_img = inp_img[:, h_rand :h_rand + self.ps[0], w_rand :w_rand + self.ps[1]]
            tar_img_enh = tar_img_enh[:, h_rand :h_rand + self.ps[0] , w_rand :w_rand + self.ps[1]]

        aug_1 = random.randint(0, 8)
        # Data Augmentations
        if aug_1 == 1:
            inp_img = inp_img.flip(1)
            tar_img_enh = tar_img_enh.flip(1)
        elif aug_1 == 2:
            inp_img = inp_img.flip(2)
            tar_img_enh = tar_img_enh.flip(2)
        elif aug_1 == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img_enh = torch.rot90(tar_img_enh, dims=(1, 2))
        elif aug_1 == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img_enh = torch.rot90(tar_img_enh, dims=(1, 2), k=2)
        elif aug_1 == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img_enh = torch.rot90(tar_img_enh, dims=(1, 2), k=3)
        elif aug_1 == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img_enh = torch.rot90(tar_img_enh.flip(1), dims=(1, 2))
        elif aug_1 == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img_enh = torch.rot90(tar_img_enh.flip(2), dims=(1, 2))

        return inp_img, tar_img_enh

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'trainA')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'trainB')))

        self.inp_filenames = [os.path.join(rgb_dir, 'trainA', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'trainB', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target_enh

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img_enh = Image.open(tar_path).convert('RGB')

        inp_img = TF.to_tensor(inp_img)
        tar_img_enh = TF.to_tensor(tar_img_enh)

        inp_img = TF.resize(inp_img, (self.ps[0], self.ps[1]))
        tar_img_enh = TF.resize(tar_img_enh, (self.ps[0], self.ps[1]))

        return inp_img, tar_img_enh

def get_training_data(dir, img_options):
    assert os.path.exists(dir)
    return DataLoaderTrain(dir, img_options)

def get_validation_data(dir, img_options):
    assert os.path.exists(dir)
    return DataLoaderVal(dir, img_options)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_best_metrics(weights):
    checkpoint = torch.load(weights)
    return checkpoint["PSNR"], checkpoint["SSIM"]

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])

def network_parameters(nets):
    num_params = sum(param.numel() for param in nets.parameters())
    return num_params

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

from pytorch_msssim import ssim
def torchSSIM(tar_img, prd_img):
    return ssim(tar_img, prd_img, data_range=1.0, size_average=True)



from model import UDAFT

model_opt = {'IMG_SIZE': [256, 256],
       'IN_CHANS': 3,
       'OUT_CHANS': 3,
       'PATCH_SIZE': 2,
       'DEPTH_EN': [8, 8, 8, 8],
       'EMB_DIM': 32,
       'APE': False,
       'PATCH_NORM': True,
       'MLP_RATIO': 4.0,
       'QKV_BIAS': True,
       'QK_SCALE': 8,
       'HEAD_NUM': [8, 8, 8, 8],
       'WIN_SIZE': 8,
       'DROP_RATE': 0,
       'DROP_PATH_RATE': 0.1,
       'ATTN_DROP_RATE': 0.,
       'USE_CHECKPOINTS': False}
optim_opt = {'BATCH': 2,
             'EPOCHS': 800,
             'LR_INITIAL': 5e-4,
             'LR_MIN': 1e-6,
             'BETA1': 0.9}
train_opt = {'MODEL_NAME': 'DUAFT',
             'NUM_WORKS': 8,
             'TRAIN_PS': [256, 256],
             'VAL_PS': [256, 256]}


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Set Seeds
torch.backends.cudnn.benchmark = True
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Path configurate
model_name = 'DUAFT'
save_dir = 'checkpoints'
model_dir = os.path.join(save_dir, model_name, 'models')
mkdir(model_dir)
result_dir = os.path.join(save_dir, model_name, 'results')
mkdir(result_dir)
log_dir = os.path.join(save_dir, model_name, 'log')
mkdir(log_dir)


train_data_dir = train
val_data_dir = test

logname = os.path.join(save_dir, model_name, datetime.datetime.now().isoformat()[:13] + '-' +
                       datetime.datetime.now().isoformat()[14:16] + '-' +
                       datetime.datetime.now().isoformat()[17:19] + '.txt')
with open(logname, 'a') as f:
        f.write(str(i) + ':\t' + str(model_opt[str(i)]) + '\n')


# Build Model
print('================= Building Model =================')
model_restore = URSCT().cuda()

device_ids = 1

# Log
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{model_name}')

# Optimizer
new_lr = float(optim_opt['LR_INITIAL'])
optimizer = optim.Adam(model_restore.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

# Learning rate strategy
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, optim_opt['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(optim_opt['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

# Loss
Charbonnier_loss = Charbonnier_Loss().cuda()
from test_tensor import VGG19_PercepLoss
L_vgg = VGG19_PercepLoss().cuda()

# Resume (Continue training by a pretrained model)
start_epoch = 1
best_psnr, best_ssim = 0., 0.
RESUME = True
if RESUME:
    print("================= Loading Resuming configuration ================= ")
    path_pth = r'/checkpoints/models'
    path_chk_rest = get_last_path(path_pth, 'ckp.pth')
    load_checkpoint(model_restore, path_chk_rest)
    start_epoch = load_start_epoch(path_chk_rest) + 1
    best_psnr, best_ssim = load_best_metrics(path_chk_rest)
    load_optim(optimizer, path_chk_rest)
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print(path_pth + str(' _latest.pth'))

# DataLoaders
print("================= Creating dataloader ================= ")
train_dataset = get_training_data(train_data_dir, {'patch_size': train_opt['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=optim_opt['BATCH'],
                          shuffle=True, num_workers=train_opt['NUM_WORKS'], drop_last=False)
val_dataset = get_validation_data(val_data_dir, {'patch_size': train_opt['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, num_workers=train_opt['NUM_WORKS'],
                        drop_last=False)

# Show the training configuration
print(f'''================ Training details ================
------------------------------------------------------------------
    Model Name:   {model_name}
    Train patches size: {str(train_opt['TRAIN_PS']) + 'x' + str(train_opt['TRAIN_PS'])}
    Val patches size:   {str(train_opt['VAL_PS']) + 'x' + str(train_opt['VAL_PS'])}
    Model parameters:   {network_parameters(model_restore)}
    Start/End epochs:   {str(start_epoch) + '~' + str(optim_opt['EPOCHS'])}
    Best PSNR:          {str(best_psnr)}
    Best SSIM:          {str(best_ssim)}
    Batch sizes:        {optim_opt['BATCH']}
    Learning rate:      {new_lr}
    GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')


# Start training
total_start_time = time.time()
train_show = True
best_epoch_psnr = start_epoch
best_epoch_ssim = start_epoch

eval_now = len(train_loader) // 4
print('================ Training ================')
for epoch in range(start_epoch, optim_opt['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss_total = 0
    epoch_loss_l1 = 0
    epoch_loss_vgg = 0

    # train
    model_restore.train()

    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        optimizer.zero_grad()

        input_ = data[0].cuda()
        target_enh = data[1].cuda()
        restored_enh = model_restore(input_)

        loss_l1_charbonnier = Charbonnier_loss(restored_enh, target_enh)
        loss_per = L_vgg(restored_enh, target_enh)
        loss_total = loss_l1_charbonnier + loss_per * 0.1

        if train_show:
            with torch.no_grad():
                pbar.set_description("[Epoch] {} [Mode] TRAIN [Loss_enh] {:.4f} [PSNR_enh] {:.4f} [SSIM_enh] {:.4f}".format(epoch,
                                                                                                                            loss_total.item(),
                                                                                                                            torchPSNR(restored_enh, target_enh),
                                                                                                                            torchSSIM(restored_enh, target_enh)))
        # backward
        loss_total.backward()
        optimizer.step()

        epoch_loss_l1 += loss_l1_charbonnier.item()
        epoch_loss_vgg += 0
        epoch_loss_total += loss_total.item()


        ## Evaluation
        if (i + 1) % eval_now == 0 and i > 0:
            model_restore.eval()
            PSNRs = []
            SSIMs = []
            pbar = tqdm(val_loader)
            for ii, data_val in enumerate(pbar, 0):
                input_ = data_val[0].cuda()
                target_enh = data_val[1].cuda()
                restored_enh = model_restore(input_)
                with torch.no_grad():
                    for res, tar in zip(restored_enh, target_enh):
                        temp_psnr = torchPSNR(res, tar)
                        temp_ssim = torchSSIM(restored_enh, target_enh)
                        PSNRs.append(temp_psnr)
                        SSIMs.append(temp_ssim)
                        pbar.set_description("[Epoch] {} [MODE] VALID [PSNR] {:.4f} [SSIM] {:.4f}".format(epoch,
                                                                                                          torchPSNR(restored_enh, target_enh),
                                                                                                          torchSSIM(restored_enh, target_enh)))

            PSNRs = torch.stack(PSNRs).mean().item()
            SSIMs = torch.stack(SSIMs).mean().item()


            # Save the best PSNR model of validation
            if PSNRs > best_psnr:
                best_psnr = PSNRs
                best_epoch_psnr = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restore.state_dict(),
                            'PSNR': best_psnr,
                            'SSIM': best_ssim
                            }, os.path.join(model_dir, "model_bestPSNR.pth"))
            print("[PSNR] {:.4f} [Best_PSNR] {:.4f} (epoch {})".format(PSNRs, best_psnr, best_epoch_psnr))

            # Save the best SSIM model of validation
            if SSIMs > best_ssim:
                best_ssim = SSIMs
                best_epoch_ssim = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restore.state_dict(),
                            'PSNR': best_psnr,
                            'SSIM': best_ssim
                            }, os.path.join(model_dir, "model_bestSSIM.pth"))
            print("[SSIM] {:.4f}  [Best_SSIM] {:.4f} (epoch {})".format(SSIMs, best_ssim, best_epoch_ssim))

            torch.save({'epoch': epoch,
                        'state_dict': model_restore.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'PSNR': best_psnr,
                        'SSIM': best_ssim
                        }, os.path.join(model_dir, "epoch{}.pth".format(epoch)))


            writer.add_scalar('val/PSNR', PSNRs, epoch)
            writer.add_scalar('val/SSIM', SSIMs, epoch)

            print("[Ep %d it %d\t PSNR: %.4f\t SSIM: %.4f\t] ----  [best_Ep_PSNR %d Best_Ep_SSIM %.d]"
                  "----  [Loss %.4f Learning Rate %.6f] "
                  % (epoch, i, PSNRs, SSIMs, best_epoch_psnr, best_epoch_ssim, loss_total.item(), scheduler.get_lr()[0]))
            with open(logname, 'a') as f:
                f.write(
                    "[Ep %d it %d\t PSNR: %.4f\t SSIM: %.4f\t] ----  [best_Ep_PSNR %d Best_Ep_SSIM %.d]"
                    "----  [Loss %.4f Learning Rate %.6f] "
                    % (
                    epoch, i, PSNRs, SSIMs, best_epoch_psnr, best_epoch_ssim, loss_total.item(), scheduler.get_lr()[0])
                    + '\n')
    scheduler.step()

    with open(logname, 'a') as f:
        f.write(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLoss_l1: {:.4f}\tloss_vgg: {:.4f}\tLearningRate {:.6f}".format(
                epoch, time.time() - epoch_start_time,
                epoch_loss_total, epoch_loss_l1, epoch_loss_vgg,
                scheduler.get_lr()[0]) + '\n')

    print("------------------------------------------------------------------")
    print("[Epoch] {} [Time] {:.4f} [Loss] {:.4f} [Loss_l1] {:.4f} [loss_vgg] {:.4f}  [Learning Rate] {:.6f}".format(epoch,
                                                                                                                     time.time() - epoch_start_time,
                                                                                                                     epoch_loss_total,
                                                                                                                     epoch_loss_l1,
                                                                                                                     epoch_loss_vgg,
                                                                                                                     scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restore.state_dict(),
                'optimizer': optimizer.state_dict(),
                'PSNR': best_psnr,
                'SSIM': best_ssim
                }, os.path.join(model_dir, "model_latest.pth"))

    writer.add_scalar('train/loss', epoch_loss_total, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
writer.close()

print('Total training time: {:.1f} hours'.format(((time.time() - total_start_time) / 60 / 60)))