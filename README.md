# Talking Therapy Dog: Unpaired Image-to-Image Translation between Disimilar Image Domains
Stanford CS 236G Final Project

## Setup
Follow these instructions to use this repo.
#### Environment
First, clone the repo, create a new environment and activate it.

With Conda:
```
  $> conda create --name [ENVNAME] python=3.8
    ...
  $> conda activate [ENVNAME] 
```

Then install all required packages:
```
  $> pip install -r requirements.txt
```

#### Dataset
Download the dataset by running the provided script:
```
  $> ./download_dataset.sh
```
This will download both the CelebA (w/ Landmarks) and DogFaceNet (cropped) Datasets to the `./data` folder

## Usage
For the base implementation of CycleGAN, first follow the setup above to prepare the environment and download the dataset.

The full model is compartmentalized into multiple aptly named files. The full CycleGAN model is run via `main.py`. 

There are multiple arguments for `main.py`, but the only required argument to train is `--train`. So to start training a new model run:
```
  $> python main.py --train
```

As of now, test mode has not been implemented

In addition, the main function takes in a configuration file like `config.json` which specifies data location, number of epochs, learning rate, loss weights, etc.
The config file can be specified via `--config [CONFIG JSON]`.
