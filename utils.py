################################################################################
# Imports
################################################################################

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid

from tqdm.auto import tqdm

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from PIL import Image



torch.manual_seed(0)


################################################################################
# Helpers
################################################################################
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
