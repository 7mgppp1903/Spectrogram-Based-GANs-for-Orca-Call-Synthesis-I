import torch
import numpy as np
from model import SpecGANGenerator

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

G = SpecGANGenerator().to(device)
G.load_state_dict(torch.load("checkpoints/G_epoch_fixed_10.pt", map_location=device))
G.eval()

for i in range(8):
    z = torch.randn(1, 100, 1, 1, device=device)
    with torch.no_grad():
        fake = G(z).cpu().squeeze().numpy()
    np.save(f"Results/epoch10_sample_{i}.npy", fake)

print("✅ Generated 8 samples in Results/")
