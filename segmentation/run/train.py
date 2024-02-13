import os
import matplotlib.pyplot as plt
import argparse
import cv2
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from torch.utils.tensorboard import SummaryWriter

from albumentations.core.composition import Compose, OneOf
from tqdm import tqdm

from segmentation.model.unet import UNetWithResnet50Encoder
from segmentation.utils.dataset import CustomDataset
from utils.metrics import iou_pytorch
import torchvision.transforms as transforms


class Train:
    def __init__(self, config):
        super().__init__()
        self.cfg = config

        self.device = torch.device(f"{self.cfg['device']}")

        os.makedirs(f"{self.cfg['output_ckp']}", exist_ok=True)
        os.makedirs(f"{self.cfg['output_sample']}", exist_ok=True)
        os.makedirs(f"{self.cfg['output_log']}", exist_ok=True)

        self.train_transform = Compose([
            A.Resize(self.cfg['IMG_HEIGHT'], self.cfg['IMG_WIDTH']),
            OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(), ], p=0.65),
            OneOf([
                A.HueSaturationValue(),
                A.RandomBrightness(),
                A.RandomContrast(), ], p=0.8),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.val_transform = Compose([
            A.Resize(self.cfg['IMG_HEIGHT'], self.cfg['IMG_WIDTH']),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.to_image_transform = transforms.Compose([
            transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
        ])

        self.random_seed = 32

    def visualize(self, image, label, seg_image, ep, idx):
        f, ax = plt.subplots(1, 3, figsize=(20, 8))
        ax[0].imshow(image)
        ax[1].imshow(label, vmax=255, vmin=0, cmap='gray')
        ax[2].imshow(seg_image, vmax=255, vmin=0, cmap='gray')

        ax[0].set_title('Original Image')
        ax[1].set_title('Ground Truth')
        ax[2].set_title('UNet')

        ax[0].title.set_size(25)
        ax[1].title.set_size(25)
        ax[2].title.set_size(25)

        f.tight_layout()
        os.makedirs(f"{self.cfg['output_sample']}/{ep}", exist_ok=True)

        plt.savefig(f"{self.cfg['output_sample']}/{ep}/{idx}_sample.jpg")
        plt.close()

    def run(self):
        print('----------[Load Dataset]----------')
        tr_dataset = CustomDataset(self.cfg['dataset_path'], train=True, transform=self.train_transform)
        val_dataset = CustomDataset(self.cfg['dataset_path'], train=False, transform=self.val_transform)

        tr_loader = DataLoader(tr_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        print('----------[Load Model]----------')
        model = UNetWithResnet50Encoder(n_classes=3).to(self.device)

        print('----------[Optim and Loss]----------')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=float(self.cfg['lr']))

        print('----------[Run!!]----------')
        summary = SummaryWriter(f"{self.cfg['output_log']}")
        best_score = {'epoch': 0, 'iou': 0, 'acc': 0, 'loss': 0}

        for ep in range(self.cfg['epoch']):
            tr_iou_score = []
            val_iou_score = []
            tr_loss = 0
            val_loss = 0

            model.train()
            for idx, (img, mask) in enumerate(tqdm(tr_loader, desc=f"[Train {ep}/{self.cfg['epoch']}]==>")):
                img = torch.tensor(img, device=self.device, dtype=torch.float32)
                mask = torch.tensor(mask, device=self.device, dtype=torch.float32)

                optimizer.zero_grad()
                logits = model(img)
                loss = criterion(logits, mask.long())
                loss.backward()
                optimizer.step()

                tr_iou_score.extend(iou_pytorch(logits.argmax(1), mask.long()))
                tr_loss += loss.item()

            with torch.no_grad():
                model.eval()
                for idx, (img, mask) in enumerate(tqdm(val_loader, desc=f"[Validation {ep}/{self.cfg['epoch']}]==>")):
                    img = torch.tensor(img, device=self.device, dtype=torch.float32)
                    mask = torch.tensor(mask, device=self.device, dtype=torch.float32)

                    logits = model(img)
                    loss = criterion(logits, mask.long())

                    val_iou_score.extend(iou_pytorch(logits.argmax(1), mask.long()))
                    val_loss += loss.item()

                    mask_values = {
                        0: 0,
                        1: 50,
                        2: 255,
                    }
                    pred = logits.argmax(1)[0]
                    pred = np.vectorize(mask_values.get)(pred.cpu().detach())
                    mask = np.vectorize(mask_values.get)(mask[0].cpu().detach())
                    img = self.to_image_transform(img[0].squeeze()).permute(1, 2, 0).cpu().detach().numpy()

                    self.visualize(img, mask, pred, ep, idx)

                tr_iou_mean = torch.FloatTensor(tr_iou_score).mean()
                tr_loss_mean = tr_loss / len(tr_loader)

                val_iou_mean = torch.FloatTensor(val_iou_score).mean()
                val_loss_mean = val_loss / len(val_loader)

                if best_score['iou'] <= val_iou_mean:
                    best_score['epoch'] = ep
                    best_score['iou'] = val_iou_mean
                    best_score['loss'] = val_loss_mean

                print('\n')
                print('-------------------------------------------------------------------')
                print(f"Epoch: {ep}/50")
                print(f"Train iou: {tr_iou_mean} | Train loss: {tr_loss_mean}")
                print(f"Test iou: {val_iou_mean} | Test loss: {val_loss_mean}")
                print('-------------------------------------------------------------------')
                print(f"Best acc epoch: {best_score['epoch']}")
                print(f"Best iou: {best_score['iou']} | Best loss: {best_score['loss']}")
                print('-------------------------------------------------------------------')

                summary.add_scalar('Train/iou', tr_iou_mean, ep)
                summary.add_scalar('Train/loss', tr_loss_mean, ep)

                summary.add_scalar('Test/iou', val_iou_mean, ep)
                summary.add_scalar('Test/loss', val_loss_mean, ep)

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                        "epoch": ep,
                    },
                    os.path.join(f"{self.cfg['output_ckp']}", f"{ep}.pth"),
                )


if __name__ == '__main__':
    import yaml

    with open('../../configs/seg_configs.yaml') as f:
        config = yaml.safe_load(f)

    tr = Train(config)
    tr.run()
