# loss function for training

import torch
import torch.nn as nn

def loss_function(x_reconstructed, x, mu, logvar, beta=1.0):
    reconstruction_loss = nn.MSELoss()(x_reconstructed, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + beta * kl_loss
