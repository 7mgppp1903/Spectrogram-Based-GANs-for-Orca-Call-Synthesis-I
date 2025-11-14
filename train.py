import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_loader
from model import SpecGANGenerator, SpecGANDiscriminator

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

train_loader = get_loader("/Users/miilee/OrcaCallGAN/Data/sliced_train", batch_size=8)

G = SpecGANGenerator().to(device)
D = SpecGANDiscriminator().to(device)

criterion = nn.BCELoss()

# Generator learns faster
optG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
# Discriminator slowed down to prevent collapse
optD = optim.Adam(D.parameters(), lr=0.00001, betas=(0.5, 0.999))

latent_dim = 100
epochs = 10

g_losses = []
d_losses = []

for epoch in range(epochs):
    for real in train_loader:
        real = real.to(device)
        b = real.size(0)

        # Label smoothing
        valid = torch.full((b, 1), 0.9, device=device)
        fake  = torch.full((b, 1), 0.1, device=device)

        # === Train Generator ===
        optG.zero_grad()
        z = torch.randn(b, latent_dim, 1, 1, device=device)
        gen = G(z)

        # Noise regularization (prevents collapse)
        gen = gen + 0.01 * torch.randn_like(gen)

        g_loss = criterion(D(gen).view(-1, 1), valid)
        g_loss.backward()
        optG.step()

        # === Train Discriminator (slower) ===
        optD.zero_grad()

        real = real + 0.01 * torch.randn_like(real)

        real_loss = criterion(D(real).view(-1, 1), valid)
        fake_loss = criterion(D(gen.detach()).view(-1, 1), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optD.step()

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())

    # Save checkpoints every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(G.state_dict(), f"/Users/miilee/OrcaCallGAN/checkpoints/G_epoch_fixed_{epoch+1}.pt")


import numpy as np
np.save("Results/g_losses.npy", np.array(g_losses))
np.save("Results/d_losses.npy", np.array(d_losses))
