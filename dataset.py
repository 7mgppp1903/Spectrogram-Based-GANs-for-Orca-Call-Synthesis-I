import torch
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Running on:", device)

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class OrcaSpecDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx]).astype(np.float32)
        spec = torch.tensor(spec).unsqueeze(0)
        return spec

def get_loader(path, batch_size=8):
    dataset = OrcaSpecDataset(path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader




