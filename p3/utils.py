import torch, os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np

class SegDataset(Dataset):
    def __init__(self, root_path='./data', train=True, transform=None):
        super(SegDataset, self).__init__()
        self.transform = transform

        try:
            if train:
                self.data = torch.load(os.path.join(root_path, "train_data.pth")).float() / 255.
                self.target = torch.load(os.path.join(root_path, "train_target.pth")).long()
            else:
                self.data = torch.load(os.path.join(root_path, "test_data.pth")).float() / 255.
                self.target = torch.load(os.path.join(root_path, "test_target.pth")).long()
        except:
            print("Error when loading data: {train, test}_{data, target}.pth should be in data folder.")
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, target
