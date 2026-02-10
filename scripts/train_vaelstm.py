"""
Entraînement du LSTM-VAE avec vérifications de données
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

sys.path.append('.')

from src.models.vae_lstm import LSTMVAE, get_vae_config
from src.training.losses import vae_loss, VAELossTracker

print("="*70)
print("🚀 ENTRAÎNEMENT DU LSTM-VAE")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Données
    'data_dir': 'data/processed/sequences',
    
    # Architecture VAE
    'sequence_length': 288,      # 24h avec Δt=5min
    'input_size': 1,             # Features par timestep
    'model_size': 'medium',      # 'small', 'medium', 'large'
    
    # Entraînement
    'n_epochs': 200,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'beta': 1.0,                 # Poids KL divergence
    'beta_warmup_epochs': 20,    # β-warmup progressif
    
    # Early stopping
    'patience': 20,
    'min_delta': 1e-4,
    
    # Sauvegarde
    'checkpoint_dir': 'checkpoints',
    'save_every': 10,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"\n⚙️  Configuration:")
print(f"   Device: {CONFIG['device']}")
print(f"   Modèle: {CONFIG['model_size']}")
print(f"   Séquence: {CONFIG['sequence_length']} timesteps")
print(f"   Batch size: {CONFIG['batch_size']}")
print(f"   Epochs: {CONFIG['n_epochs']}")
print(f"   Learning rate: {CONFIG['learning_rate']}")
print(f"   Beta: {CONFIG['beta']} (warmup: {CONFIG['beta_warmup_epochs']} epochs)")

# ============================================================================
# CHARGEMENT DES DONNÉES AVEC VÉRIFICATIONS
# ============================================================================

print(f"\n{'='*70}")
print("📂 Chargement et vérification des données")
print("="*70)

data_dir = Path(CONFIG['data_dir'])

if not data_dir.exists():
    raise FileNotFoundError(
        f"❌ Dossier non trouvé : {data_dir}\n"
        f"Assurez-vous d'avoir généré les séquences d'abord."
    )

# Charger séquences
train_data = np.load(data_dir / 'train.npy')
val_data = np.load(data_dir / 'val.npy')

print(f"✓ Train: {train_data.shape}")
print(f"✓ Val:   {val_data.shape}")

# VÉRIFICATIONS CRITIQUES
print(f"\n🔍 Vérifications critiques:")

# 1. Nombre d'échantillons
n_train = len(train_data)
n_val = len(val_data)

print(f"   Nombre d'échantillons:")
print(f"      Train: {n_train}")
print(f"      Val:   {n_val}")

if n_train < 1000:
    print(f"   ⚠️  ATTENTION : Seulement {n_train} échantillons d'entraînement !")
    print(f"      C'est très peu. Vous devriez avoir au moins 5000-10000 séquences.")
    print(f"      Vérifiez la création des séquences (fenêtre glissante ?).")

# 2. Forme des données
if len(train_data.shape) == 2:
    print(f"\n   ⚠️  Les données sont 2D : {train_data.shape}")
    print(f"      Reshape en 3D : (samples, sequence_length, features)")
    
    # Inférer la forme
    total_size = train_data.shape[1]
    if total_size % CONFIG['sequence_length'] == 0:
        n_features = total_size // CONFIG['sequence_length']
        print(f"      Séquence détectée : length={CONFIG['sequence_length']}, features={n_features}")
        
        train_data = train_data.reshape(-1, CONFIG['sequence_length'], n_features)
        val_data = val_data.reshape(-1, CONFIG['sequence_length'], n_features)
        CONFIG['input_size'] = n_features
    else:
        raise ValueError(
            f"Impossible de reshape {train_data.shape} avec sequence_length={CONFIG['sequence_length']}"
        )

print(f"\n   Forme finale:")
print(f"      Train: {train_data.shape}")
print(f"      Val:   {val_data.shape}")

# 3. Statistiques des données
print(f"\n   Statistiques:")
print(f"      Min:    {train_data.min():.4f}")
print(f"      Max:    {train_data.max():.4f}")
print(f"      Mean:   {train_data.mean():.4f}")
print(f"      Std:    {train_data.std():.4f}")

if train_data.min() == train_data.max():
    raise ValueError("❌ Toutes les valeurs sont identiques ! Problème de normalisation.")

# 4. Vérifier les NaN
if np.isnan(train_data).any():
    raise ValueError("❌ Données contiennent des NaN !")

# ============================================================================
# CRÉER DATALOADERS
# ============================================================================

print(f"\n{'='*70}")
print("📦 Création des DataLoaders")
print("="*70)

train_dataset = TensorDataset(torch.FloatTensor(train_data))
val_dataset = TensorDataset(torch.FloatTensor(val_data))

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True,
    num_workers=0  # Pour éviter les problèmes sur Windows
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=False,
    num_workers=0
)

print(f"✓ DataLoaders créés:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches:   {len(val_loader)}")

# ============================================================================
# CRÉATION DU MODÈLE
# ============================================================================

print(f"\n{'='*70}")
print("🏗️  Création du modèle LSTM-VAE")
print("="*70)

vae_config = get_vae_config(CONFIG['model_size'])

model = LSTMVAE(
    input_size=CONFIG['input_size'],
    sequence_length=CONFIG['sequence_length'],
    **vae_config
)

model = model.to(CONFIG['device'])

print(f"✓ Modèle créé:")
print(f"   Type:       LSTM-VAE ({CONFIG['model_size']})")
print(f"   Séquence:   {CONFIG['sequence_length']}")
print(f"   Input size: {CONFIG['input_size']}")
print(f"   Latent dim: {vae_config['latent_dim']}")
print(f"   Hidden:     {vae_config['hidden_size']}")
print(f"   Layers:     {vae_config['num_layers']}")
print(f"   Paramètres: {model.count_parameters():,}")

# ============================================================================
# OPTIMISEUR ET SCHEDULER
# ============================================================================

optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

print(f"\n✓ Optimiseur: Adam (lr={CONFIG['learning_rate']})")
print(f"✓ Scheduler: ReduceLROnPlateau")

# ============================================================================
# FONCTION β-WARMUP
# ============================================================================

def get_beta(epoch, config):
    """Calcule β avec warmup progressif."""
    if epoch < config['beta_warmup_epochs']:
        return config['beta'] * (epoch / config['beta_warmup_epochs'])
    return config['beta']

# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(
    patience=CONFIG['patience'], 
    min_delta=CONFIG['min_delta']
)

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
    'val_kl': [],
    'beta': []
}

best_val_loss = float('inf')

# Créer dossier checkpoints
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)

for epoch in range(CONFIG['n_epochs']):
    
    beta = get_beta(epoch, CONFIG)
    
    # ========== TRAINING ==========
    model.train()
    train_tracker = VAELossTracker()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['n_epochs']}")
    for batch in pbar:
        x = batch[0].to(CONFIG['device'])
        
        # Forward
        x_recon, mu, log_var = model(x)
        
        # Loss (doit être calculée avec les bonnes dimensions)
        # Flatten pour la loss
        x_flat = x.reshape(x.size(0), -1)
        x_recon_flat = x_recon.reshape(x_recon.size(0), -1)
        
        loss, recon_loss, kl_div = vae_loss(
            x_recon_flat, x_flat, mu, log_var, 
            beta=beta
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track
        train_tracker.update(
            loss.item(), 
            recon_loss.item(), 
            kl_div.item()
        )
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.2f}",
            'β': f"{beta:.3f}"
        })
    
    train_losses = train_tracker.get_average()
    
    # ========== VALIDATION ==========
    model.eval()
    val_tracker = VAELossTracker()
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(CONFIG['device'])
            
            x_recon, mu, log_var = model(x)
            
            x_flat = x.reshape(x.size(0), -1)
            x_recon_flat = x_recon.reshape(x_recon.size(0), -1)
            
            loss, recon_loss, kl_div = vae_loss(
                x_recon_flat, x_flat, mu, log_var,
                beta=beta
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
    history['beta'].append(beta)
    
    # Scheduler step
    scheduler.step(val_losses['total'])
    
    # Print
    print(f"\nEpoch {epoch+1:3d}/{CONFIG['n_epochs']} | "
          f"β={beta:.3f} | "
          f"Train: {train_losses['total']:.2f} "
          f"(R:{train_losses['recon']:.2f}, KL:{train_losses['kl']:.2f}) | "
          f"Val: {val_losses['total']:.2f} "
          f"(R:{val_losses['recon']:.2f}, KL:{val_losses['kl']:.2f})")
    
    # ========== CHECKPOINTING ==========
    
    # Sauvegarder meilleur modèle
    if val_losses['total'] < best_val_loss:
        best_val_loss = val_losses['total']
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'config': CONFIG,
            'vae_config': vae_config
        }, Path(CONFIG['checkpoint_dir']) / 'lstm_vae_best.pth')
        
        print(f"   ⭐ Nouveau meilleur modèle sauvegardé !")
    
    # Early stopping
    early_stopping(val_losses['total'])
    if early_stopping.early_stop:
        print(f"\n⚠️  Early stopping déclenché à l'epoch {epoch+1}")
        break
    
    # Sauvegarder périodiquement
    if (epoch + 1) % CONFIG['save_every'] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'config': CONFIG,
            'vae_config': vae_config
        }, Path(CONFIG['checkpoint_dir']) / f'lstm_vae_epoch_{epoch+1}.pth')

print(f"\n{'='*70}")
print("✅ Entraînement terminé !")
print(f"Meilleure val loss: {best_val_loss:.4f}")
print("="*70)

# ============================================================================
# VISUALISATION
# ============================================================================

print(f"\n📊 Génération des visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Total loss
ax = axes[0, 0]
ax.plot(history['train_loss'], label='Train', linewidth=2)
ax.plot(history['val_loss'], label='Val', linewidth=2)
ax.set_title('Total Loss', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Reconstruction loss
ax = axes[0, 1]
ax.plot(history['train_recon'], label='Train', linewidth=2)
ax.plot(history['val_recon'], label='Val', linewidth=2)
ax.set_title('Reconstruction Loss', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# KL divergence
ax = axes[1, 0]
ax.plot(history['train_kl'], label='Train', linewidth=2)
ax.plot(history['val_kl'], label='Val', linewidth=2)
ax.set_title('KL Divergence', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('KL')
ax.legend()
ax.grid(True, alpha=0.3)

# Beta schedule
ax = axes[1, 1]
ax.plot(history['beta'], linewidth=2, color='purple')
ax.set_title('β Schedule', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('β')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(CONFIG['checkpoint_dir']) / 'lstm_vae_training_history.png', dpi=150)
print(f"✓ Sauvegardé: {CONFIG['checkpoint_dir']}/lstm_vae_training_history.png")
plt.close()

# Sauvegarder historique
with open(Path(CONFIG['checkpoint_dir']) / 'lstm_vae_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*70}")
print("🎉 TOUT EST PRÊT !")
print("="*70)
print(f"\n📁 Fichiers créés:")
print(f"  {CONFIG['checkpoint_dir']}/lstm_vae_best.pth")
print(f"  {CONFIG['checkpoint_dir']}/lstm_vae_training_history.png")
print(f"  {CONFIG['checkpoint_dir']}/lstm_vae_history.json")

print(f"\n📊 Résumé:")
print(f"   Échantillons train: {n_train}")
print(f"   Échantillons val:   {n_val}")
print(f"   Epochs effectués:   {len(history['train_loss'])}")
print(f"   Meilleure val loss: {best_val_loss:.4f}")

print(f"\n🚀 Prochaine étape:")
print(f"  Générer des scénarios avec: python scripts/generate_scenarios_lstm.py")

print(f"\n{'='*70}")