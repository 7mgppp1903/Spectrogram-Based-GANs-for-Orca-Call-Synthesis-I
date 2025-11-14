import matplotlib.pyplot as plt
import numpy as np

for i in range(8):
    spec = np.load(f"Results/epoch10_sample_{i}.npy")
    plt.figure(figsize=(4,4))
    plt.imshow(spec, cmap="inferno", aspect='auto', origin='lower')
    plt.title(f"Epoch 10 Sample {i}")
    plt.show()
