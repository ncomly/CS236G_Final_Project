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
from torchvision.models import inception_v3

from tqdm.auto import tqdm

import os
import sys
import glob
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from skimage import color
from PIL import Image

from functools import partial


from easydict import EasyDict as edict
from pprint import pprint

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

def convert_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().squeeze()
    return image_unflat


################################################################################
# Argument Parser
################################################################################
def parse_args():

    # Create a parser
    parser = argparse.ArgumentParser(description="Talking Therapy Dog")
    parser.add_argument('--config', default=None, type=str, help='Configuration file')
    parser.add_argument('--checkpoint', default=None, type=str, help='Model checkpoint file. If one is not provided, training will start from scratch')
    parser.add_argument('--save_path', default='./models/', type=str, help='Top level directory to store model checkpoints')
    parser.add_argument('--train', action='store_true', help='Will train if provided')
    parser.add_argument('--nosave', action='store_true', help='Will not save checkpionts if provided')
    parser.add_argument('--nolandmarks', action='store_true', help='Will not use landmarks if provided')
    parser.add_argument('--iv3', action='store_true', help='Will use InceptionV3 for the cycle loss')

    # Parse the arguments
    args = parser.parse_args()

    # config file
    if not args.config:
        print('No config .json file provided. Defaulting to \'config.json\'')
        args.config = 'config.json'

    # Parse the configurations from the config json file provided
    try:
        if args.config is not None:
            with open(args.config, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    config_args = edict(config_args_dict)
    config_args.train = args.train
    config_args.save_path = args.save_path
    config_args.checkpoint = args.checkpoint
    config_args.save = not args.nosave
    config_args.iv3 = args.iv3
    

    pprint(config_args)
    print("\n")

    return config_args
