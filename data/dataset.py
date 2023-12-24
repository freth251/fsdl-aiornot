import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import io

class ParquetDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.dataframe = self.load_data()

    def load_data(self):
        # Load all parquet files and concatenate into a single DataFrame
        files = [f for f in os.listdir(self.directory) if f.endswith('.parquet')]
        df_list = [pd.read_parquet(os.path.join(self.directory, f)) for f in files]
        return pd.concat(df_list, ignore_index=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Assuming 'image' column contains image data in bytes format
        image = Image.open(io.BytesIO(self.dataframe.iloc[idx]['image']['bytes']))
        label = self.dataframe.iloc[idx]['label']
        label = torch.tensor(label)
        label = label.to(torch.float32)
        if self.transform:
            image = self.transform(image)

        return image, label

