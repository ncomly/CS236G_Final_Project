# Talking Therapy Dog
## Unpaired Image-to-Image Translation between Disimilar Image Domains
Stanford CS 236G Final Project

Nick Comly


### Setup
Follow these instructions to use this repo.
#### Environment
First, clone the repo, create a new environment and activate it.

With Conda:
```
  $> conda create --env ENVNAME python=3.8
    ...
  $> conda activate ENVNAME 
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
