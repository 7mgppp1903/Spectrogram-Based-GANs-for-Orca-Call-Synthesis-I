import os
import numpy as np

input_dir = "/Users/miilee/PycharmProjects/OrcaCallGAN/data/SpectogramTrain"
output_dir = "/data/sliced_train"
os.makedirs(output_dir, exist_ok=True)

window_size = 128
step = 128  # you can also try step=64 for overlapping slices

for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        S = np.load(os.path.join(input_dir, filename))

        # Slide across time dimension
        for i in range(0, S.shape[1] - window_size, step):
            slice_ = S[:, i:i+window_size]

            # Normalize to [-1, 1]
            slice_ = (slice_ - slice_.min()) / (slice_.max() - slice_.min())
            slice_ = (slice_ * 2) - 1

            out_name = f"{filename[:-4]}_{i}.npy"
            np.save(os.path.join(output_dir, out_name), slice_)
