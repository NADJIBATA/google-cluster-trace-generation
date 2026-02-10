"""
Loss utilities for VAE training.

Includes a simple MSE reconstruction loss + KL divergence
and a small helper to track running averages during training.
"""

import torch
import torch.nn.functional as F


def vae_loss(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
	"""Compute VAE loss: reconstruction (MSE) + beta * KL divergence.

	Returns: (total_loss, recon_loss, kl_div)
	recon_loss and kl_div are returned as tensor scalars (averaged per-batch).
	"""
	# Reconstruction loss (MSE) averaged per sample
	# sum over features then mean over batch
	recon = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)

	# KL divergence between N(mu, var) and N(0,1)
	# sum over latent dims then mean over batch
	kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
	kl = kl.mean()

	total = recon + beta * kl
	return total, recon, kl


class VAELossTracker:
	"""Simple accumulator for loss components returning averages."""

	def __init__(self):
		self.total = 0.0
		self.recon = 0.0
		self.kl = 0.0
		self.count = 0

	def update(self, total_loss: float, recon_loss: float, kl: float):
		self.total += total_loss
		self.recon += recon_loss
		self.kl += kl
		self.count += 1

	def get_average(self):
		if self.count == 0:
			return {'total': 0.0, 'recon': 0.0, 'kl': 0.0}
		return {
			'total': self.total / self.count,
			'recon': self.recon / self.count,
			'kl': self.kl / self.count
		}


__all__ = ['vae_loss', 'VAELossTracker']
