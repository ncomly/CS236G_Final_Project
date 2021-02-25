# Inspired by CS 236G Coursera Course Content
# Inspired by https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/datasets.py
from utils import *
class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train', a_subroot='.', b_subroot='.'):
        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(root, a_subroot, mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, b_subroot, mode) + '/*.*'))
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
        assert len(self.files_A) > 0, "Make sure you downloaded the dataset images!"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))
        if item_A.shape[0] != 3: 
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3: 
            item_B = item_B.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()
        # Old versions of PyTorch didn't support normalization for different-channeled images
        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))