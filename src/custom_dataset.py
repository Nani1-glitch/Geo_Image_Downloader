import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.valid_indices = self._get_valid_indices()
        print(f"Number of valid samples: {len(self.valid_indices)}")

    def _get_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.data_frame)):
            img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
            if os.path.exists(img_name):
                valid_indices.append(idx)
            else:
                print(f"File not found: {img_name}")
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        lat = self.data_frame.iloc[idx, 1]
        lon = self.data_frame.iloc[idx, 2]
        return image, torch.tensor([lat, lon])
