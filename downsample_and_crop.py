from skimage import io, draw, transform, img_as_ubyte
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm

data_dir = 'C:/Users/nacom/Documents/Datasets/CelebA/img_align_celeba/CelebA_50k/'
# data_dir = 'C:/Users/nacom/Documents/Datasets/DogFaceNet/after_4_bis/'

out_res = 100

output_dir = f'downsampled_{out_res}'


def center_crop(img, res=178):
    H, W, C = img.shape
    
    mid_H = H // 2
    mid_W = W // 2
    start_H = mid_H - res // 2
    start_W = mid_W - res // 2
    stop_H  = mid_H + res // 2
    stop_W  = mid_W + res // 2

    return img[start_H:stop_H, start_W:stop_W, :]

def downsample(img, ratio=2):
    return img[::ratio, ::ratio, :]

def mkdir_p(dir):
    if not os.path.isdir(dir):
        print(f'Directory Does Not Exit\nCreating - \"{dir}\"')
        os.mkdir(dir)
    else:
        print(f'Directory Found! - \"{dir}\"')

def process_celebA(data_dir, output):
    # make the Output folder (if not already there)
    output_dir = os.path.join(data_dir, output)
    mkdir_p(output_dir)

    for subset in ['train', 'test', 'val']:
        input_path  = os.path.join(data_dir, subset)
        output_path = os.path.join(output_dir, subset)
        mkdir_p(output_path)

        for img in tqdm.tqdm(os.listdir(input_path)):
            output_img_path = os.path.join(output_path, img)
            if os.path.isfile(output_img_path):
                continue
            # read in image
            img_data = io.imread(os.path.join(input_path, img))
            # crop
            img_data = center_crop(img_data, res=min(img_data.shape[:2]))

            # resize
            img_data = transform.resize(img_data, (out_res, out_res, 3))
            # save the image
            io.imsave(output_img_path, img_as_ubyte(img_data))

def process_dogs(data_dir, output):
    # make the Output folder (if not already there)
    output_dir = os.path.join(data_dir, output)
    mkdir_p(output_dir)

    dogs = os.listdir(data_dir)
    for dog in tqdm.tqdm(dogs):
        if dog == output or os.path.isfile(os.path.join(data_dir, dog)):
            continue
        dog_path = os.path.join(data_dir, dog)
        for img in os.listdir(dog_path):
            # generate and check output path
            output_path = os.path.join(output_dir, img).replace('.jpg', '_c.jpg')
            if os.path.isfile(output_path):
                continue
            # read in image
            img_data = io.imread(os.path.join(dog_path, img))
            # transform image
            img_data = transform.resize(img_data, (out_res, out_res, 3))
            # save image
            io.imsave(output_path, img_as_ubyte(img_data))

# process_dogs(data_dir, output_dir)
process_celebA(data_dir, output_dir)