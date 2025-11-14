import numpy as np
import matplotlib.pyplot as plt

g = np.load("Results/g_losses.npy")
d = np.load("Results/d_losses.npy")

plt.figure(figsize=(6,4))
plt.plot(g, label="Generator Loss")
plt.plot(d, label="Discriminator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("GAN Training Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("Results/loss_curve.png", dpi=300)
plt.show()
