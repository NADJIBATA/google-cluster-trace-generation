"""
VAE avec LSTM pour séries temporelles.
Architecture adaptée aux workloads temporels.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

class LSTMVAE(nn.Module):
    """
    VAE basé sur LSTM pour séries temporelles.
    
    Architecture:
        Encoder: LSTM → Dense → (μ, log_σ²)
        Decoder: Dense → LSTM → Dense → Reconstruction
    """
    
    def __init__(
        self,
        input_size: int = 1,
        sequence_length: int = 288,
        hidden_size: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Args:
            input_size: Nombre de features par timestep (1 pour univarié)
            sequence_length: Longueur de la séquence
            hidden_size: Taille des couches cachées LSTM
            latent_dim: Dimension de l'espace latent
            num_layers: Nombre de couches LSTM
            dropout: Taux de dropout
            bidirectional: Utiliser LSTM bidirectionnel
        """
        super().__init__()
            
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # ============= ENCODER =============
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Couches pour μ et log_σ²
        encoder_output_size = hidden_size * self.num_directions
        self.fc_mu = nn.Linear(encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_size, latent_dim)
        
        # ============= DECODER =============
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size)
        
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_output = nn.Linear(hidden_size, input_size)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode l'input en paramètres de distribution latente.
        
        Args:
            x: (batch_size, sequence_length, input_size)
        
        Returns:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        # LSTM encoder
        # output: (batch_size, seq_len, hidden_size * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_size)
        lstm_out, (hidden, cell) = self.encoder_lstm(x)
        
        # Utiliser le dernier état caché
        if self.bidirectional:
            # Concaténer forward et backward du dernier layer
            hidden_last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden_last = hidden[-1]
        
        # Calculer μ et log(σ²)
        mu = self.fc_mu(hidden_last)
        logvar = self.fc_logvar(hidden_last)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, où ε ~ N(0, 1)
        
        Args:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        
        Returns:
            z: (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Décode z en séquence reconstruite.
        
        Args:
            z: (batch_size, latent_dim)
        
        Returns:
            reconstructed: (batch_size, sequence_length, input_size)
        """
        batch_size = z.size(0)
        
        # Projeter z dans l'espace du LSTM
        hidden = self.latent_to_hidden(z)  # (batch_size, hidden_size)
        hidden = torch.tanh(hidden)
        
        # Répéter pour chaque pas de temps
        # (batch_size, sequence_length, hidden_size)
        decoder_input = hidden.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # LSTM decoder
        lstm_out, _ = self.decoder_lstm(decoder_input)
        
        # Projeter vers la dimension de sortie
        reconstructed = self.fc_output(lstm_out)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass complet du VAE.
        
        Args:
            x: (batch_size, sequence_length, input_size)
        
        Returns:
            reconstructed: (batch_size, sequence_length, input_size)
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        # Encoder
        mu, logvar = self.encode(x)
        
        # Reparameterization
        z = self.reparameterize(mu, logvar)
        
        # Decoder
        reconstructed = self.decode(z)
        
        return reconstructed, mu, logvar
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Génère de nouvelles séquences en samplant l'espace latent.
        
        Args:
            num_samples: nombre de séquences à générer
            device: 'cpu' ou 'cuda'
        
        Returns:
            samples: (num_samples, sequence_length, input_size)
        """
        with torch.no_grad():
            # Sampler z depuis N(0, 1)
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Décoder
            samples = self.decode(z)
        
        return samples
    
    def count_parameters(self) -> int:
        """Compte le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_vae_config(config_type: str = 'medium') -> dict:
    """
    Retourne une configuration prédéfinie.
    
    Args:
        config_type: 'small', 'medium', 'large'
    
    Returns:
        dict avec paramètres du modèle
    """
    configs = {
        'small': {
            'hidden_size': 64,
            'latent_dim': 16,
            'num_layers': 1,
            'dropout': 0.1,
            'bidirectional': False
        },
        'medium': {
            'hidden_size': 128,
            'latent_dim': 32,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': False
        },
        'large': {
            'hidden_size': 256,
            'latent_dim': 64,
            'num_layers': 3,
            'dropout': 0.3,
            'bidirectional': True
        }
    }
    
    return configs.get(config_type, configs['medium'])


# Test du modèle
if __name__ == "__main__":
    
    print("=" * 70)
    print("🧪 TEST DU LSTM-VAE")
    print("=" * 70)
    
    # Paramètres de test
    batch_size = 16
    sequence_length = 288  # 24h avec Δt=5min
    input_size = 1
    
    # Créer données de test
    x = torch.randn(batch_size, sequence_length, input_size)
    print(f"\n📊 Données de test : {x.shape}")
    
    # Tester chaque configuration
    for config_name in ['small', 'medium', 'large']:
        print(f"\n{'='*70}")
        print(f"Testing {config_name.upper()} VAE")
        print("="*70)
        
        config = get_vae_config(config_name)
        
        model = LSTMVAE(
            input_size=input_size,
            sequence_length=sequence_length,
            **config
        )
        
        n_params = model.count_parameters()
        print(f"📊 Nombre de paramètres : {n_params:,}")
        
        # Forward pass
        x_recon, mu, logvar = model(x)
        
        print(f"✅ Forward pass réussi")
        print(f"   Input shape        : {x.shape}")
        print(f"   Reconstructed shape: {x_recon.shape}")
        print(f"   Mu shape           : {mu.shape}")
        print(f"   Logvar shape       : {logvar.shape}")
        
        # Test de génération
        samples = model.sample(num_samples=5)
        print(f"✅ Génération réussie : {samples.shape}")
    
    print("\n" + "=" * 70)
    print("✅ TOUS LES TESTS RÉUSSIS")
    print("=" * 70)