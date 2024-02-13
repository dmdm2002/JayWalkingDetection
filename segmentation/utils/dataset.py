import glob
import os
import cv2
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

random.seed(1004)


class CustomDataset(Dataset):
    def __init__(self, root: str, train: bool, transform=None):
        super().__init__()
        self.transform = transform
        self.root = root
        data_info = pd.read_csv(f'{root}/train_test_split.csv')

        if train:
            train_df = data_info.groupby('run_type').get_group('train')
            self.img_path = train_df['img_name'].to_numpy()
            self.mask_path = train_df['mask_name'].to_numpy()
        else:
            valid_df = data_info.groupby('run_type').get_group('test')
            self.img_path = valid_df['img_name'].to_numpy()
            self.mask_path = valid_df['mask_name'].to_numpy()

    # def one_hot_encode(self, gt):
    #     semantic_map = []
    #     for colour in self.mask_path:
    #         equality = np.equal(gt, colour)
    #         class_map = np.all(equality, axis=-1)
    #         semantic_map.append(class_map)
    #     semantic_map = np.stack(semantic_map, axis=-1)
    #
    #     return semantic_map

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(f'{self.root}/{self.img_path[idx]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(f'{self.root}/{self.mask_path[idx]}')

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        return torch.from_numpy(img), torch.from_numpy(mask)


if __name__ == '__main__':
    db = CustomDataset('C:/Users/rlawj/sample_DB/Sejong/segmentation', True)

