import face_alignment
from skimage import io, draw
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm

celebA_dir = 'C:/Users/nacom/Documents/Datasets/CelebA/img_align_celeba/'
dogFace_dir = 'C:/Users/nacom/Documents/Datasets/DogFaceNet/after_4_bis/'

color_map = dict({
    ('jaw', (0,0,.7)),
    ('eyebrow_l', (0.7, 0, 0)),
    ('eyebrow_r', (0.7, 0, 0)),
    ('nose_bridge', (0.5, 0, 0.5)),
    ('nose_base', (0.5, 0, 0.5)),
    ('eye_l', (0, 0.5, 0)),
    ('eye_r', (0, 0.5, 0)),
    ('mouth', (1, 0.7, 0)),
    ('lips', (1, 0.7, 0)),

})

def draw_seg(start, end):
    return draw.line_aa(start[0], start[1], end[0], end[1])

def extract_features(preds):
    features = [
        ('jaw', preds[:17]),
        ('eyebrow_l', preds[17:22]),
        ('eyebrow_r', preds[22:27]),
        ('nose_bridge', preds[27:31]),
        ('nose_base', preds[31:36]),
        ('eye_l', np.vstack((preds[36:42], preds[36]))),
        ('eye_r', np.vstack((preds[42:48], preds[42]))),
        ('mouth', np.vstack((preds[48:61], preds[48]))),
        ('lips', preds[61:])
    ]
    return features

def draw_image(base, features):
    # draw lines
    for f, v in features:
        for i in range(len(v)-1):
            a = v[i].astype(int)
            b = v[i+1].astype(int)
            cc, rr, val = draw_seg(a,b)
            rr = np.clip(rr, 0, 217)
            cc = np.clip(cc, 0, 177)
            base[rr, cc] = val[:, np.newaxis]*color_map[f]*255 + (1-val)[:, np.newaxis]*base[rr,cc]
    return base

def process_single_image(img, landmarks_dir, data_dir, save_csv=True, save_jpg=True):
    # generate output path
    output_path = os.path.join(landmarks_dir, img)
    
    # save as img if true, else as csv
    if save_csv:
        output_path_csv = output_path.replace('.jpg', '_l.csv')
    if save_jpg:
        output_path_jpg = output_path.replace('.jpg', '_l.jpg')

    # # do not process if already in folder
    # if not ((save_csv and not os.path.isfile(output_path_csv)) or 
    #         (save_jpg and not os.path.isfile(output_path_jpg))):
    #     return
    if os.path.isfile(output_path_jpg):
        return

    # read in image
    img_data = io.imread(os.path.join(data_dir, img))
    # get prediction
    preds = fa.get_landmarks(img_data)
    # error checking
    if not preds:
        print('No Face Detected!')
        return
    # remove outter list
    preds = preds[0]

    # if saving as a csv
    # if save_csv and not os.path.isfile(output_path_csv):
    #     np.savetxt(output_path_csv, preds, delimiter=',')


    # if saving as jpg
    # if save_jpg and not os.path.isfile(output_path_jpg):
    # extract facial features
    features = extract_features(preds)

    # white background
    base = np.ones_like(img_data)*255

    # draw the image
    landmarks = draw_image(base, features)

    # save image
    io.imsave(output_path_jpg, landmarks)


def preprocess_celebA(data_dir):

    # make the Landmarks folder (if not already there)
    landmarks_dir = os.path.join(data_dir, 'landmarks')
    if not os.path.isdir(landmarks_dir):
        print(f'Landmarks Directory Does Not Exit\nCreating...')
        os.mkdir(landmarks_dir)
        print(f' - {landmarks_dir}')
    else:
        print(f'Landmarks Directory Found! - {landmarks_dir}')


    # process dataset
    #   - one at a time to allow for failure
    for img in tqdm.tqdm(os.listdir(data_dir)):
        img_path = os.path.join(data_dir,img)
        if os.path.isfile(img_path):
            process_single_image(img, landmarks_dir, data_dir)

def preprocess_dogFace(data_dir, w=178, h=218):
    # make the crops folder (if not already there)
    crops_dir = os.path.join(data_dir, 'crops')
    if not os.path.isdir(crops_dir):
        print(f'Crops Directory Does Not Exit\nCreating...')
        os.mkdir(crops_dir)
        print(f' - {crops_dir}')
    else:
        print(f'Crops Directory Found! - {crops_dir}')

    # center points
    # hardcoding is bad! but easier...
    c_w = (224//2)-1
    c_h = (224//2)-1


    dogs = os.listdir(data_dir)
    for dog in tqdm.tqdm(dogs):
        if dog == 'crops':
            continue
        dog_path = os.path.join(data_dir, dog)
        for img in os.listdir(dog_path):
            # generate and check output path
            output_path = os.path.join(crops_dir, img).replace('.jpg', '_c.jpg')
            if os.path.isfile(output_path):
                continue
            # read and crop data
            img_data = io.imread(os.path.join(dog_path, img))
            img_data = img_data[ c_h - h//2:c_h + h//2,c_w - w//2:c_w + w//2,:]

            # save image
            io.imsave(output_path, img_data)


if __name__ == '__main__':
    # create face aligner
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
    preprocess_celebA(celebA_dir)
    preprocess_dogFace(dogFace_dir)

