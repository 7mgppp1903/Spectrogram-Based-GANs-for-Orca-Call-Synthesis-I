import os
import numpy as np

input_dir = "/Users/miilee/PycharmProjects/OrcaCallGAN/data/SpectrogramTest"
output_dir = "/Users/miilee/PycharmProjects/OrcaCallGAN/data/sliced_test"
os.makedirs(output_dir, exist_ok=True)

window_size = 128
step = 128  # keep same step size as train

for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        S = np.load(os.path.join(input_dir, filename))

        for i in range(0, S.shape[1] - window_size, step):
            slice_ = S[:, i:i+window_size]

            # normalize to [-1, 1]
            slice_ = (slice_ - slice_.min()) / (slice_.max() - slice_.min())
            slice_ = (slice_ * 2) - 1

            np.save(os.path.join(output_dir, f"{filename[:-4]}_{i}.npy"), slice_)

print("Test spectrograms sliced and normalized!")
