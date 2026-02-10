"""
Lightweight VAE implementation used by training script.

Provides a simple MLP encoder/decoder that works with flattened
sequence inputs (shape: batch, input_dim).

API:
- VAE(input_dim, latent_dim, hidden_dims, activation, dropout)
- forward(x) -> x_recon, mu, log_var
- count_parameters() -> int
"""

from typing import List
import torch
import torch.nn as nn


def _make_mlp(in_dim: int, layer_dims: List[int], activation: str = 'relu', dropout: float = 0.0):
	layers = []
	act = nn.ReLU if activation == 'relu' else nn.LeakyReLU
	prev = in_dim
	for d in layer_dims:
		layers.append(nn.Linear(prev, d))
		layers.append(act())
		if dropout and dropout > 0:
			layers.append(nn.Dropout(dropout))
		prev = d
	return nn.Sequential(*layers)


class VAE(nn.Module):
	"""Simple fully-connected VAE for flattened sequences."""

	def __init__(self, input_dim: int = 100, latent_dim: int = 32, hidden_dims: List[int] = None,
				 activation: str = 'relu', dropout: float = 0.0):
		super().__init__()
		if hidden_dims is None:
			hidden_dims = [256, 128]

		# Encoder
		self.encoder_mlp = _make_mlp(input_dim, hidden_dims, activation, dropout)
		last_enc = hidden_dims[-1] if len(hidden_dims) > 0 else input_dim
		self.fc_mu = nn.Linear(last_enc, latent_dim)
		self.fc_logvar = nn.Linear(last_enc, latent_dim)

		# Decoder
		# Mirror hidden dims for decoder
		dec_hidden = list(reversed(hidden_dims))
		self.decoder_input = nn.Linear(latent_dim, dec_hidden[0] if len(dec_hidden) > 0 else input_dim)
		self.decoder_mlp = _make_mlp(dec_hidden[0] if len(dec_hidden) > 0 else input_dim,
									 dec_hidden[1:] if len(dec_hidden) > 1 else [], activation, dropout)
		self.final_layer = nn.Linear(dec_hidden[-1] if len(dec_hidden) > 0 else input_dim, input_dim)

	def encode(self, x: torch.Tensor):
		h = self.encoder_mlp(x)
		mu = self.fc_mu(h)
		logvar = self.fc_logvar(h)
		return mu, logvar

	def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def decode(self, z: torch.Tensor):
		h = self.decoder_input(z)
		h = self.decoder_mlp(h)
		x_recon = self.final_layer(h)
		return x_recon

	def forward(self, x: torch.Tensor):
		"""Forward returns reconstruction, mu and logvar."""
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		x_recon = self.decode(z)
		return x_recon, mu, logvar

	def count_parameters(self) -> int:
		return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ['VAE']
