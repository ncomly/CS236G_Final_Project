echo "Downloading Dataset for Talking Therapy Dog Project"
mkdir -p data/
cd data/

echo "Downloading DogFaceNet Dataset"
wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1bphRTkOb3m-fxyi7lvo9ps_7vO85PCeJ" -O "DogFaceNet_crops.tar"

echo "Downloading CelebA w/ Landmarks Dataset"

wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1FdfOkdYWmuahpPOqc0dglV8nvQ5vfCs8" -O "CelebA_with_Landmarks.tar"
tar -xvf CelebA_with_Landmarks.tar