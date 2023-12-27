import torch
from torch.utils.data import Dataset


class AIOrNotDataset(Dataset):
    """AI or not dataset from https://huggingface.co/datasets/competitions/aiornot"""

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Assuming 'image' column contains image data in bytes format
        image = self.data[idx]["image"]
        label = self.data[idx]["label"]

        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label, dtype=torch.float32)

        return image, label
