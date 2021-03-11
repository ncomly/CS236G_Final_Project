# Inspired by CS 236G Coursera Course Content
# Inspired by https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/datasets.py
from utils import *
class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train', a_subroot='.', b_subroot='.', l_subroot=None):
        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(root, a_subroot, mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, b_subroot, mode) + '/*.*'))
        if l_subroot:
            self.files_L = sorted(glob.glob(os.path.join(root, l_subroot, mode) + '/*.*'))
        else:
            self.file_L = None
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
        assert len(self.files_A) > 0, "Make sure you downloaded the dataset images!"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))
        item_L = self.transform(Image.open(self.files_L[self.randperm[index]])) 
        
        if item_A.shape[0] != 3: 
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3: 
            item_B = item_B.repeat(3, 1, 1)
        if item_L.shape[0] != 3: 
            item_L = item_B.repeat(3, 1, 1)
                
        if index == len(self) - 1:
            self.new_perm()
        return (item_A - 0.5) * 2, (item_B - 0.5) * 2, (item_L - 0.5) * 2
                



    def __len__(self):
        return min(len(self.files_A), len(self.files_B))