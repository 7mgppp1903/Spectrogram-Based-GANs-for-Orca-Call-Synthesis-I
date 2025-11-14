import os
import numpy as np

folders = [
    "/Users/miilee/OrcaCallGAN/Data/sliced_train",
    "/Users/miilee/OrcaCallGAN/Data/sliced_test"
]

for folder in folders:
    for f in os.listdir(folder):
        if f.endswith(".npy"):
            arr = np.load(os.path.join(folder, f))
            if arr.shape != (128, 128):
                os.remove(os.path.join(folder, f))