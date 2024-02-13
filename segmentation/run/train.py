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
    def __init__(self):
        super().__init__()
