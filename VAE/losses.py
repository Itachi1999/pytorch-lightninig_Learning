import torch.nn as nn
import torch


class VAE_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, recon, data, mu, log_var):
        bce_loss = self.bce_loss(recon, data)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return bce_loss + KLD