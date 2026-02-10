"""
Génération de Scénarios avec LSTM-VAE
======================================

Ce script utilise le LSTM-VAE entraîné pour générer différents
types de scénarios de workload.

Types de scénarios:
- Nominal: Comportement moyen typique
- Optimiste: Charge faible
- Pessimiste: Charge élevée
- Aléatoire: Variété de comportements
- Stress: Situations extrêmes
"""

import numpy as np
import torch
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys

sys.path.append('.')

from src.models.vae_lstm import LSTMVAE

print("="*70)
print("🎨 GÉNÉRATION DE SCÉNARIOS AVEC LSTM-VAE")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Modèle
    'checkpoint_path': 'checkpoints/lstm_vae_best.pth',
    'scaler_path': 'data/processed/sequences/scaler.pkl',
    
    # Génération
    'num_random_scenarios': 10,
    'num_stress_scenarios': 10,
    
    # Output
    'output_dir': 'data/generated',
    'figures_dir': 'results/figures/scenarios',
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"\n⚙️  Configuration:")
print(f"   Device: {CONFIG['device']}")
print(f"   Checkpoint: {CONFIG['checkpoint_path']}")

# ============================================================================
# CHARGEMENT DU MODÈLE
# ============================================================================

print(f"\n{'='*70}")
print("📂 Chargement du modèle")
print("="*70)

# Charger le checkpoint
checkpoint_path = Path(CONFIG['checkpoint_path'])
if not checkpoint_path.exists():
    print(f"❌ Checkpoint non trouvé : {checkpoint_path}")
    print(f"\n💡 Assurez-vous d'avoir entraîné le modèle d'abord:")
    print(f"   python scripts/train_lstm_vae.py")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])

print(f"✓ Checkpoint chargé")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Val loss: {checkpoint['val_loss']:.4f}")

# Reconstruire le modèle
model_config = checkpoint['vae_config']
model = LSTMVAE(
    input_size=1,  # Univarié
    sequence_length=checkpoint['config']['sequence_length'],
    hidden_size=model_config['hidden_size'],
    latent_dim=model_config['latent_dim'],
    num_layers=model_config['num_layers'],
    dropout=model_config['dropout'],
    bidirectional=model_config.get('bidirectional', False)
)

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(CONFIG['device'])
model.eval()

print(f"\n✓ Modèle reconstruit:")
print(f"  Séquence length: {checkpoint['config']['sequence_length']}")
print(f"  Latent dim: {model_config['latent_dim']}")
print(f"  Hidden size: {model_config['hidden_size']}")
print(f"  Paramètres: {model.count_parameters():,}")

# ============================================================================
# CHARGEMENT DU SCALER
# ============================================================================

print(f"\n{'='*70}")
print("📏 Chargement du scaler")
print("="*70)

scaler_path = Path(CONFIG['scaler_path'])
if not scaler_path.exists():
    print(f"❌ Scaler non trouvé : {scaler_path}")
    sys.exit(1)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

print(f"✓ Scaler chargé")
print(f"  Mean: {scaler.mean_[0]:.4f}")
print(f"  Std: {np.sqrt(scaler.var_[0]):.4f}")

# ============================================================================
# ANALYSER L'ESPACE LATENT (sur données d'entraînement)
# ============================================================================

print(f"\n{'='*70}")
print("🔍 Analyse de l'espace latent")
print("="*70)

# Charger les données d'entraînement pour analyser l'espace latent
train_data = np.load('data/processed/sequences/train.npy')
print(f"✓ Données d'entraînement chargées: {train_data.shape}")

# Encoder toutes les séquences d'entraînement
print(f"Encodage de {len(train_data)} séquences...")
latent_codes = []

with torch.no_grad():
    batch_size = 32
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        batch_tensor = torch.FloatTensor(batch).to(CONFIG['device'])
        mu, logvar = model.encode(batch_tensor)
        latent_codes.append(mu.cpu().numpy())

latent_codes = np.concatenate(latent_codes, axis=0)
print(f"✓ Espace latent analysé: {latent_codes.shape}")

# Statistiques de l'espace latent
latent_mean = latent_codes.mean(axis=0)
latent_std = latent_codes.std(axis=0)
latent_min = latent_codes.min(axis=0)
latent_max = latent_codes.max(axis=0)

print(f"\nStatistiques espace latent:")
print(f"  Mean: {latent_mean.mean():.4f}")
print(f"  Std:  {latent_std.mean():.4f}")
print(f"  Min:  {latent_min.min():.4f}")
print(f"  Max:  {latent_max.max():.4f}")

# Identifier les dimensions avec le plus de variance
variance = latent_codes.var(axis=0)
top_variance_dims = np.argsort(variance)[::-1][:5]
print(f"\nTop 5 dimensions par variance: {top_variance_dims}")

# ============================================================================
# GÉNÉRATION DES SCÉNARIOS
# ============================================================================

print(f"\n{'='*70}")
print("🎨 Génération des scénarios")
print("="*70)

def denormalize_sequence(seq_normalized, scaler):
    """Dénormalise une séquence."""
    seq_reshaped = seq_normalized.reshape(-1, 1)
    seq_denorm = scaler.inverse_transform(seq_reshaped)
    return seq_denorm.reshape(seq_normalized.shape)

def generate_from_latent(z, model, scaler, apply_constraints=True):
    """
    Génère une séquence depuis un code latent.
    
    Args:
        z: Code latent (latent_dim,)
        model: Modèle VAE
        scaler: Scaler pour dénormalisation
        apply_constraints: Appliquer contraintes (positif, entier)
    
    Returns:
        Séquence dénormalisée
    """
    with torch.no_grad():
        z_tensor = torch.FloatTensor(z).unsqueeze(0).to(CONFIG['device'])
        seq_norm = model.decode(z_tensor)
        seq_norm = seq_norm.cpu().numpy()[0]  # (seq_len, 1)
    
    # Dénormaliser
    seq = denormalize_sequence(seq_norm, scaler)
    
    # Appliquer contraintes
    if apply_constraints:
        seq = np.maximum(seq, 0)  # Pas de valeurs négatives
        seq = np.round(seq)       # Entiers (nombre de jobs)
    
    return seq

# Dictionnaire pour stocker les scénarios
scenarios = {}

# ============================================================================
# 1. SCÉNARIO NOMINAL
# ============================================================================

print(f"\n1️⃣  Scénario NOMINAL (comportement moyen)")

z_nominal = latent_mean.copy()
scenario_nominal = generate_from_latent(z_nominal, model, scaler)

scenarios['nominal'] = scenario_nominal
print(f"   ✓ Généré: shape={scenario_nominal.shape}")
print(f"     Mean: {scenario_nominal.mean():.2f}, Min: {scenario_nominal.min():.0f}, Max: {scenario_nominal.max():.0f}")

# ============================================================================
# 2. SCÉNARIO OPTIMISTE (charge faible)
# ============================================================================

print(f"\n2️⃣  Scénario OPTIMISTE (charge faible)")

# Utiliser percentile bas de l'espace latent
z_optimistic = np.percentile(latent_codes, 10, axis=0)
scenario_optimistic = generate_from_latent(z_optimistic, model, scaler)

scenarios['optimistic'] = scenario_optimistic
print(f"   ✓ Généré: shape={scenario_optimistic.shape}")
print(f"     Mean: {scenario_optimistic.mean():.2f}, Min: {scenario_optimistic.min():.0f}, Max: {scenario_optimistic.max():.0f}")

# ============================================================================
# 3. SCÉNARIO PESSIMISTE (charge élevée)
# ============================================================================

print(f"\n3️⃣  Scénario PESSIMISTE (charge élevée)")

# Utiliser percentile haut de l'espace latent
z_pessimistic = np.percentile(latent_codes, 90, axis=0)
scenario_pessimistic = generate_from_latent(z_pessimistic, model, scaler)

scenarios['pessimistic'] = scenario_pessimistic
print(f"   ✓ Généré: shape={scenario_pessimistic.shape}")
print(f"     Mean: {scenario_pessimistic.mean():.2f}, Min: {scenario_pessimistic.min():.0f}, Max: {scenario_pessimistic.max():.0f}")

# ============================================================================
# 4. SCÉNARIOS ALÉATOIRES (variété)
# ============================================================================

print(f"\n4️⃣  Scénarios ALÉATOIRES ({CONFIG['num_random_scenarios']} variantes)")

random_scenarios = []
for i in range(CONFIG['num_random_scenarios']):
    # Sampler depuis une gaussienne N(0, 1)
    z_random = np.random.randn(model.latent_dim)
    scenario_random = generate_from_latent(z_random, model, scaler)
    random_scenarios.append(scenario_random)

scenarios['random'] = np.array(random_scenarios)
print(f"   ✓ Généré: {len(random_scenarios)} scénarios")
print(f"     Mean (avg): {np.mean([s.mean() for s in random_scenarios]):.2f}")

# ============================================================================
# 5. SCÉNARIOS STRESS (extrêmes)
# ============================================================================

print(f"\n5️⃣  Scénarios STRESS ({CONFIG['num_stress_scenarios']} variantes)")

stress_scenarios = []
for i in range(CONFIG['num_stress_scenarios']):
    # Créer un vecteur latent avec des valeurs extrêmes
    # sur les dimensions avec le plus de variance
    z_stress = latent_mean.copy()
    
    # Prendre alternativement des valeurs extrêmes
    for j, dim in enumerate(top_variance_dims):
        if i % 2 == 0:
            # Extrême haut
            z_stress[dim] = latent_max[dim] + np.random.rand() * latent_std[dim]
        else:
            # Extrême bas
            z_stress[dim] = latent_min[dim] - np.random.rand() * latent_std[dim]
    
    scenario_stress = generate_from_latent(z_stress, model, scaler)
    stress_scenarios.append(scenario_stress)

scenarios['stress'] = np.array(stress_scenarios)
print(f"   ✓ Généré: {len(stress_scenarios)} scénarios")
print(f"     Mean (avg): {np.mean([s.mean() for s in stress_scenarios]):.2f}")

# ============================================================================
# SAUVEGARDE
# ============================================================================

print(f"\n{'='*70}")
print("💾 Sauvegarde des scénarios")
print("="*70)

output_dir = Path(CONFIG['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

# Sauvegarder chaque type de scénario
np.save(output_dir / 'scenario_nominal.npy', scenarios['nominal'])
np.save(output_dir / 'scenario_optimistic.npy', scenarios['optimistic'])
np.save(output_dir / 'scenario_pessimistic.npy', scenarios['pessimistic'])
np.save(output_dir / 'scenarios_random.npy', scenarios['random'])
np.save(output_dir / 'scenarios_stress.npy', scenarios['stress'])

print(f"✓ Scénarios sauvegardés dans {output_dir}")

# Sauvegarder en CSV pour analyse facile
for name, scenario in [('nominal', scenarios['nominal']), 
                       ('optimistic', scenarios['optimistic']),
                       ('pessimistic', scenarios['pessimistic'])]:
    import pandas as pd
    df = pd.DataFrame({
        'timestep': range(len(scenario)),
        'value': scenario.flatten()
    })
    df.to_csv(output_dir / f'scenario_{name}.csv', index=False)
    print(f"✓ {name}.csv")

# Statistiques
stats = {
    'nominal': {
        'mean': float(scenarios['nominal'].mean()),
        'std': float(scenarios['nominal'].std()),
        'min': float(scenarios['nominal'].min()),
        'max': float(scenarios['nominal'].max()),
    },
    'optimistic': {
        'mean': float(scenarios['optimistic'].mean()),
        'std': float(scenarios['optimistic'].std()),
        'min': float(scenarios['optimistic'].min()),
        'max': float(scenarios['optimistic'].max()),
    },
    'pessimistic': {
        'mean': float(scenarios['pessimistic'].mean()),
        'std': float(scenarios['pessimistic'].std()),
        'min': float(scenarios['pessimistic'].min()),
        'max': float(scenarios['pessimistic'].max()),
    },
    'random': {
        'mean': float(np.mean([s.mean() for s in scenarios['random']])),
        'std': float(np.mean([s.std() for s in scenarios['random']])),
    },
    'stress': {
        'mean': float(np.mean([s.mean() for s in scenarios['stress']])),
        'std': float(np.mean([s.std() for s in scenarios['stress']])),
    }
}

with open(output_dir / 'scenarios_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"✓ scenarios_statistics.json")

# ============================================================================
# VISUALISATION
# ============================================================================

print(f"\n{'='*70}")
print("📊 Création des visualisations")
print("="*70)

figures_dir = Path(CONFIG['figures_dir'])
figures_dir.mkdir(parents=True, exist_ok=True)

# Figure 1: Comparaison Nominal/Optimiste/Pessimiste
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

timesteps = np.arange(len(scenarios['nominal']))

# Nominal
ax = axes[0]
ax.plot(timesteps, scenarios['nominal'], linewidth=2, color='blue', label='Nominal')
ax.fill_between(timesteps, 0, scenarios['nominal'].flatten(), alpha=0.3, color='blue')
ax.set_title('Scénario NOMINAL (comportement moyen)', fontsize=12, fontweight='bold')
ax.set_ylabel('Nombre d\'arrivées')
ax.grid(True, alpha=0.3)
ax.legend()

# Optimiste
ax = axes[1]
ax.plot(timesteps, scenarios['optimistic'], linewidth=2, color='green', label='Optimiste')
ax.fill_between(timesteps, 0, scenarios['optimistic'].flatten(), alpha=0.3, color='green')
ax.set_title('Scénario OPTIMISTE (charge faible)', fontsize=12, fontweight='bold')
ax.set_ylabel('Nombre d\'arrivées')
ax.grid(True, alpha=0.3)
ax.legend()

# Pessimiste
ax = axes[2]
ax.plot(timesteps, scenarios['pessimistic'], linewidth=2, color='red', label='Pessimiste')
ax.fill_between(timesteps, 0, scenarios['pessimistic'].flatten(), alpha=0.3, color='red')
ax.set_title('Scénario PESSIMISTE (charge élevée)', fontsize=12, fontweight='bold')
ax.set_ylabel('Nombre d\'arrivées')
ax.set_xlabel('Timestep (5 minutes)')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(figures_dir / 'scenarios_main.png', dpi=300, bbox_inches='tight')
print(f"✓ scenarios_main.png")
plt.close()

# Figure 2: Scénarios aléatoires
fig, ax = plt.subplots(figsize=(16, 6))

for i, scenario in enumerate(scenarios['random'][:5]):  # Afficher 5 premiers
    ax.plot(timesteps, scenario, linewidth=1, alpha=0.7, label=f'Random {i+1}')

ax.set_title('Scénarios ALÉATOIRES (échantillon de 5)', fontsize=12, fontweight='bold')
ax.set_xlabel('Timestep (5 minutes)')
ax.set_ylabel('Nombre d\'arrivées')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(figures_dir / 'scenarios_random.png', dpi=300, bbox_inches='tight')
print(f"✓ scenarios_random.png")
plt.close()

# Figure 3: Scénarios stress
fig, ax = plt.subplots(figsize=(16, 6))

for i, scenario in enumerate(scenarios['stress'][:5]):  # Afficher 5 premiers
    ax.plot(timesteps, scenario, linewidth=1, alpha=0.7, label=f'Stress {i+1}')

ax.set_title('Scénarios STRESS (échantillon de 5)', fontsize=12, fontweight='bold')
ax.set_xlabel('Timestep (5 minutes)')
ax.set_ylabel('Nombre d\'arrivées')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(figures_dir / 'scenarios_stress.png', dpi=300, bbox_inches='tight')
print(f"✓ scenarios_stress.png")
plt.close()

# Figure 4: Comparaison distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Nominal
axes[0, 0].hist(scenarios['nominal'].flatten(), bins=30, alpha=0.7, edgecolor='black', color='blue')
axes[0, 0].set_title('Distribution NOMINAL', fontweight='bold')
axes[0, 0].set_xlabel('Nombre d\'arrivées')
axes[0, 0].set_ylabel('Fréquence')
axes[0, 0].grid(True, alpha=0.3)

# Optimiste
axes[0, 1].hist(scenarios['optimistic'].flatten(), bins=30, alpha=0.7, edgecolor='black', color='green')
axes[0, 1].set_title('Distribution OPTIMISTE', fontweight='bold')
axes[0, 1].set_xlabel('Nombre d\'arrivées')
axes[0, 1].set_ylabel('Fréquence')
axes[0, 1].grid(True, alpha=0.3)

# Pessimiste
axes[1, 0].hist(scenarios['pessimistic'].flatten(), bins=30, alpha=0.7, edgecolor='black', color='red')
axes[1, 0].set_title('Distribution PESSIMISTE', fontweight='bold')
axes[1, 0].set_xlabel('Nombre d\'arrivées')
axes[1, 0].set_ylabel('Fréquence')
axes[1, 0].grid(True, alpha=0.3)

# Comparaison
axes[1, 1].hist(scenarios['nominal'].flatten(), bins=30, alpha=0.5, label='Nominal', color='blue')
axes[1, 1].hist(scenarios['optimistic'].flatten(), bins=30, alpha=0.5, label='Optimiste', color='green')
axes[1, 1].hist(scenarios['pessimistic'].flatten(), bins=30, alpha=0.5, label='Pessimiste', color='red')
axes[1, 1].set_title('Comparaison des Distributions', fontweight='bold')
axes[1, 1].set_xlabel('Nombre d\'arrivées')
axes[1, 1].set_ylabel('Fréquence')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'scenarios_distributions.png', dpi=300, bbox_inches='tight')
print(f"✓ scenarios_distributions.png")
plt.close()

# ============================================================================
# RÉSUMÉ
# ============================================================================

print(f"\n{'='*70}")
print("🎉 GÉNÉRATION TERMINÉE AVEC SUCCÈS !")
print("="*70)

print(f"\n📊 Résumé des scénarios générés:")
print(f"\n  1. NOMINAL:")
print(f"     Mean: {stats['nominal']['mean']:.2f}")
print(f"     Range: [{stats['nominal']['min']:.0f}, {stats['nominal']['max']:.0f}]")

print(f"\n  2. OPTIMISTE:")
print(f"     Mean: {stats['optimistic']['mean']:.2f}")
print(f"     Range: [{stats['optimistic']['min']:.0f}, {stats['optimistic']['max']:.0f}]")
print(f"     Réduction: {(1 - stats['optimistic']['mean']/stats['nominal']['mean'])*100:.1f}%")

print(f"\n  3. PESSIMISTE:")
print(f"     Mean: {stats['pessimistic']['mean']:.2f}")
print(f"     Range: [{stats['pessimistic']['min']:.0f}, {stats['pessimistic']['max']:.0f}]")
print(f"     Augmentation: {(stats['pessimistic']['mean']/stats['nominal']['mean'] - 1)*100:.1f}%")

print(f"\n  4. ALÉATOIRES: {CONFIG['num_random_scenarios']} scénarios")
print(f"     Mean (avg): {stats['random']['mean']:.2f}")

print(f"\n  5. STRESS: {CONFIG['num_stress_scenarios']} scénarios")
print(f"     Mean (avg): {stats['stress']['mean']:.2f}")

print(f"\n📁 Fichiers créés:")
print(f"  {output_dir}/")
print(f"    - scenario_nominal.npy/.csv")
print(f"    - scenario_optimistic.npy/.csv")
print(f"    - scenario_pessimistic.npy/.csv")
print(f"    - scenarios_random.npy ({CONFIG['num_random_scenarios']} scénarios)")
print(f"    - scenarios_stress.npy ({CONFIG['num_stress_scenarios']} scénarios)")
print(f"    - scenarios_statistics.json")

print(f"\n  {figures_dir}/")
print(f"    - scenarios_main.png")
print(f"    - scenarios_random.png")
print(f"    - scenarios_stress.png")
print(f"    - scenarios_distributions.png")

print(f"\n🚀 Prochaines étapes:")
print(f"  1. Examiner les visualisations dans {figures_dir}/")
print(f"  2. Analyser les statistiques dans scenarios_statistics.json")
print(f"  3. Évaluer la qualité avec des métriques (KS test, Wasserstein, etc.)")
print(f"  4. Comparer avec les données réelles")

print(f"\n{'='*70}")