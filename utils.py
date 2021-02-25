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
import argparse
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


################################################################################
# Argument Parser
################################################################################
def parse_args():

    # Create a parser
    parser = argparse.ArgumentParser(description="Talking Therapy Dog")
    parser.add_argument('--config', default=None, type=str, help='Configuration file')
    parser.add_argument('--checkpoint', default=None, type=str, help='Model checkpoint file')
    parser.add_argument('--save_path', default='./models/', type=str, help='Top level directory to store model checkpoints')
    parser.add_argument('--train', default=False, type=bool, help='True if training')

    # Parse the arguments
    args = parser.parse_args()

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
    config_args.test = True if config_args.test != 0 else False
    config_args.train = args.train
    config_args.save_path = args.save_path
    config_args.checkpoint = args.checkpoint
    

    pprint(config_args)
    print("\n")

    return config_args
