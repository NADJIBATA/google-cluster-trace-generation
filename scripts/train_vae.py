"""
Entraînement du VAE
===================

Script pour entraîner le VAE sur les séquences de workload.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ajouter le chemin
sys.path.append('.')

from src.models.vae_base import VAE
from src.training.losses import vae_loss, VAELossTracker

print("="*70)
print("🚀 ENTRAÎNEMENT DU VAE")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Données
    'data_dir': 'data/processed/sequences',
    
    # Architecture VAE
    'input_dim': 100,           # Longueur séquence × n_features
    'latent_dim': 32,           # Dimension espace latent
    'hidden_dims': [256, 128],  # Couches cachées encoder/decoder
    'activation': 'relu',
    'dropout': 0.1,
    
    # Entraînement
    'n_epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'beta': 1.0,                # Poids KL divergence (β-VAE)
    
    # Sauvegarde
    'checkpoint_dir': 'checkpoints',
    'save_every': 10,           # Sauvegarder tous les N epochs
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"\n⚙️  Configuration:")
print(f"   Device: {CONFIG['device']}")
print(f"   Latent dim: {CONFIG['latent_dim']}")
print(f"   Batch size: {CONFIG['batch_size']}")
print(f"   Epochs: {CONFIG['n_epochs']}")
print(f"   Learning rate: {CONFIG['learning_rate']}")
print(f"   Beta (KL weight): {CONFIG['beta']}")

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

print(f"\n{'='*70}")
print("📂 Chargement des données")
print("="*70)

data_dir = Path(CONFIG['data_dir'])

# Charger séquences
train_data = np.load(data_dir / 'train.npy')
val_data = np.load(data_dir / 'val.npy')

print(f"✓ Train: {train_data.shape}")
print(f"✓ Val:   {val_data.shape}")

# Flatten pour VAE (batch, seq_len*features)
train_flat = train_data.reshape(len(train_data), -1)
val_flat = val_data.reshape(len(val_data), -1)

print(f"\nAprès flatten:")
print(f"  Train: {train_flat.shape}")
print(f"  Val:   {val_flat.shape}")

# Créer DataLoaders
train_dataset = TensorDataset(torch.FloatTensor(train_flat))
val_dataset = TensorDataset(torch.FloatTensor(val_flat))

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=False
)

print(f"\n✓ DataLoaders créés:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")

# ============================================================================
# CRÉATION DU MODÈLE
# ============================================================================

print(f"\n{'='*70}")
print("🏗️  Création du modèle VAE")
print("="*70)

model = VAE(
    input_dim=CONFIG['input_dim'],
    latent_dim=CONFIG['latent_dim'],
    hidden_dims=CONFIG['hidden_dims'],
    activation=CONFIG['activation'],
    dropout=CONFIG['dropout']
)

model = model.to(CONFIG['device'])

print(f"✓ Modèle créé:")
print(f"  Input dim:  {CONFIG['input_dim']}")
print(f"  Latent dim: {CONFIG['latent_dim']}")
print(f"  Hidden:     {CONFIG['hidden_dims']}")
print(f"  Paramètres: {model.count_parameters():,}")

# ============================================================================
# OPTIMISEUR
# ============================================================================

optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

print(f"\n✓ Optimiseur: Adam (lr={CONFIG['learning_rate']})")

# ============================================================================
# ENTRAÎNEMENT
# ============================================================================

print(f"\n{'='*70}")
print("🔥 Entraînement")
print("="*70)

# Tracking
history = {
    'train_loss': [],
    'train_recon': [],
    'train_kl': [],
    'val_loss': [],
    'val_recon': [],
    'val_kl': []
}

best_val_loss = float('inf')

# Créer dossier checkpoints
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)

for epoch in range(CONFIG['n_epochs']):
    
    # ========== TRAINING ==========
    model.train()
    train_tracker = VAELossTracker()
    
    for batch in train_loader:
        x = batch[0].to(CONFIG['device'])
        
        # Forward
        x_recon, mu, log_var = model(x)
        
        # Loss
        loss, recon_loss, kl_div = vae_loss(
            x_recon, x, mu, log_var, 
            beta=CONFIG['beta']
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track
        train_tracker.update(
            loss.item(), 
            recon_loss.item(), 
            kl_div.item()
        )
    
    train_losses = train_tracker.get_average()
    
    # ========== VALIDATION ==========
    model.eval()
    val_tracker = VAELossTracker()
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(CONFIG['device'])
            
            x_recon, mu, log_var = model(x)
            loss, recon_loss, kl_div = vae_loss(
                x_recon, x, mu, log_var,
                beta=CONFIG['beta']
            )
            
            val_tracker.update(
                loss.item(),
                recon_loss.item(),
                kl_div.item()
            )
    
    val_losses = val_tracker.get_average()
    
    # ========== LOGGING ==========
    history['train_loss'].append(train_losses['total'])
    history['train_recon'].append(train_losses['recon'])
    history['train_kl'].append(train_losses['kl'])
    history['val_loss'].append(val_losses['total'])
    history['val_recon'].append(val_losses['recon'])
    history['val_kl'].append(val_losses['kl'])
    
    # Print
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{CONFIG['n_epochs']} | "
              f"Train Loss: {train_losses['total']:.4f} "
              f"(R: {train_losses['recon']:.4f}, KL: {train_losses['kl']:.4f}) | "
              f"Val Loss: {val_losses['total']:.4f}")
    
    # ========== CHECKPOINTING ==========
    
    # Sauvegarder meilleur modèle
    if val_losses['total'] < best_val_loss:
        best_val_loss = val_losses['total']
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'config': CONFIG
        }, Path(CONFIG['checkpoint_dir']) / 'vae_best.pth')
    
    # Sauvegarder périodiquement
    if (epoch + 1) % CONFIG['save_every'] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'config': CONFIG
        }, Path(CONFIG['checkpoint_dir']) / f'vae_epoch_{epoch+1}.pth')

print(f"\n{'='*70}")
print("✅ Entraînement terminé !")
print(f"Meilleure val loss: {best_val_loss:.4f}")
print("="*70)

# ============================================================================
# VISUALISATION
# ============================================================================

print(f"\n📊 Génération des visualisations...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Total loss
axes[0].plot(history['train_loss'], label='Train', linewidth=2)
axes[0].plot(history['val_loss'], label='Val', linewidth=2)
axes[0].set_title('Total Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Reconstruction loss
axes[1].plot(history['train_recon'], label='Train', linewidth=2)
axes[1].plot(history['val_recon'], label='Val', linewidth=2)
axes[1].set_title('Reconstruction Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# KL divergence
axes[2].plot(history['train_kl'], label='Train', linewidth=2)
axes[2].plot(history['val_kl'], label='Val', linewidth=2)
axes[2].set_title('KL Divergence')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('KL')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(CONFIG['checkpoint_dir']) / 'training_history.png', dpi=150)
print(f"✓ Sauvegardé: {CONFIG['checkpoint_dir']}/training_history.png")
plt.close()

# Sauvegarder historique
with open(Path(CONFIG['checkpoint_dir']) / 'history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*70}")
print("🎉 TOUT EST PRÊT !")
print("="*70)
print(f"\n📁 Fichiers créés:")
print(f"  {CONFIG['checkpoint_dir']}/vae_best.pth")
print(f"  {CONFIG['checkpoint_dir']}/training_history.png")
print(f"  {CONFIG['checkpoint_dir']}/history.json")

print(f"\n🚀 Prochaine étape:")
print(f"  Générer des scénarios avec: python scripts/generate_scenarios.py")

print(f"\n{'='*70}")