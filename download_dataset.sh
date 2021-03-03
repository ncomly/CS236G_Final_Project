echo "Download Dataset for Talking Therapy Dog Project"
mkdir -p data/
cd data/

echo "DogFaceNet Dataset"
# gdown "https://drive.google.com/u/1/uc?id=1bphRTkOb3m-fxyi7lvo9ps_7vO85PCeJ" -O "DogFaceNet_crops.tar"

gdown "https://drive.google.com/u/1/uc?id=1x3M7ozNIiSLnG2XdD4ZhLU8N7-w0QQH4" -O "DogFaceNet_crops.tar"
tar -xvf DogFaceNet_crops.tar

echo "CelebA w/ Landmarks Dataset"

gdown "https://drive.google.com/u/1/uc?id=1FdfOkdYWmuahpPOqc0dglV8nvQ5vfCs8" -O "CelebA_with_Landmarks.tar"

tar -xvf CelebA_with_Landmarks.tar
