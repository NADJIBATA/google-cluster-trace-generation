"""
Préparation Complète des Données - VERSION CORRIGÉE
====================================================

Ce script corrige TOUS les problèmes :
1. Clipping des outliers
2. Création de BEAUCOUP de séquences (stride=1)
3. Normalisation correcte
4. Validation complète
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("="*70)
print("🔧 PRÉPARATION COMPLÈTE DES DONNÉES - VERSION CORRIGÉE")
print("="*70)

# ============================================================================
# CONFIGURATION OPTIMISÉE
# ============================================================================

CONFIG = {
    # Fichier source
    'input_file': 'data/processed/time_series_dt5min.csv',
    
    # Séquences - OPTIMISÉ POUR AVOIR BEAUCOUP DE DONNÉES
    'sequence_length': 100,     # 100 × 5min = 8.33h
    'stride': 1,                # ⭐ stride=1 pour MAXIMUM de séquences !
    
    # Outliers
    'clip_outliers': True,
    'clip_percentile': 99,      # Clipper au 99ème percentile
    
    # Split
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    
    # Random
    'random_seed': 42
}

print(f"\n⚙️  Configuration:")
print(f"   Séquence: {CONFIG['sequence_length']} timesteps ({CONFIG['sequence_length']*5/60:.1f}h)")
print(f"   Stride: {CONFIG['stride']} (overlap maximal)")
print(f"   Clipping: P{CONFIG['clip_percentile']}")

# ============================================================================
# ÉTAPE 1 : CHARGER
# ============================================================================

print(f"\n{'='*70}")
print("📂 ÉTAPE 1 : Chargement")
print("="*70)

df = pd.read_csv(CONFIG['input_file'], index_col=0, parse_dates=True)

# Détecter colonne
if 'arrival_rate' in df.columns:
    col = 'arrival_rate'
elif 'job_count' in df.columns:
    col = 'job_count'
else:
    col = df.select_dtypes(include=[np.number]).columns[0]

data = df[col].values

print(f"✓ Chargé: {len(data):,} timesteps")
print(f"  Période: {df.index.min()} → {df.index.max()}")
print(f"  Durée: {(df.index.max() - df.index.min()).days} jours")

print(f"\n  Statistiques brutes:")
print(f"    Min:     {data.min():.2f}")
print(f"    Max:     {data.max():.2f}")
print(f"    Mean:    {data.mean():.2f}")
print(f"    Median:  {np.median(data):.2f}")
print(f"    Std:     {data.std():.2f}")

# ============================================================================
# ÉTAPE 2 : CLIPPING DES OUTLIERS
# ============================================================================

print(f"\n{'='*70}")
print("✂️  ÉTAPE 2 : Clipping des outliers")
print("="*70)

if CONFIG['clip_outliers']:
    # Calculer percentiles
    p_low = 100 - CONFIG['clip_percentile']
    p_high = CONFIG['clip_percentile']
    
    lower_bound = np.percentile(data, p_low)
    upper_bound = np.percentile(data, p_high)
    
    print(f"Percentiles:")
    print(f"  P{p_low}:  {lower_bound:.2f}")
    print(f"  P{p_high}: {upper_bound:.2f}")
    
    # Détecter outliers
    n_high = (data > upper_bound).sum()
    n_low = (data < lower_bound).sum()
    
    print(f"\nOutliers détectés:")
    print(f"  Hauts: {n_high} ({n_high/len(data)*100:.2f}%)")
    print(f"  Bas:   {n_low} ({n_low/len(data)*100:.2f}%)")
    
    if n_high > 0 or n_low > 0:
        # Clipper
        data_clipped = np.clip(data, lower_bound, upper_bound)
        
        print(f"\n✓ Clipping appliqué:")
        print(f"  Avant: [{data.min():.0f}, {data.max():.0f}]")
        print(f"  Après: [{data_clipped.min():.0f}, {data_clipped.max():.0f}]")
        
        data = data_clipped
    else:
        print(f"\n✓ Aucun outlier à clipper")

# ============================================================================
# ÉTAPE 3 : CRÉER SÉQUENCES
# ============================================================================

print(f"\n{'='*70}")
print("🔀 ÉTAPE 3 : Création des séquences")
print("="*70)

def create_sequences(data, seq_length, stride):
    """Fenêtre glissante optimisée."""
    sequences = []
    for i in range(0, len(data) - seq_length + 1, stride):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences).reshape(-1, seq_length, 1)

sequences = create_sequences(data, CONFIG['sequence_length'], CONFIG['stride'])

print(f"✓ Séquences créées:")
print(f"  Shape: {sequences.shape}")
print(f"  N séquences: {sequences.shape[0]:,}")

# Estimation
n_expected = (len(data) - CONFIG['sequence_length']) // CONFIG['stride'] + 1
print(f"  Attendu: {n_expected:,}")

# Vérification quantité
if sequences.shape[0] < 500:
    print(f"\n  ⚠️  Seulement {sequences.shape[0]} séquences !")
    print(f"     Recommandation: Réduire stride ou augmenter données")
elif sequences.shape[0] < 5000:
    print(f"\n  ℹ️  {sequences.shape[0]} séquences : acceptable")
else:
    print(f"\n  ✅ {sequences.shape[0]:,} séquences : excellent !")

# ============================================================================
# ÉTAPE 4 : NORMALISATION
# ============================================================================

print(f"\n{'='*70}")
print("📏 ÉTAPE 4 : Normalisation")
print("="*70)

scaler = StandardScaler()

# Flatten
n_seq, seq_len, n_feat = sequences.shape
sequences_flat = sequences.reshape(-1, n_feat)

print(f"Avant normalisation:")
print(f"  Min:  {sequences_flat.min():.2f}")
print(f"  Max:  {sequences_flat.max():.2f}")
print(f"  Mean: {sequences_flat.mean():.2f}")
print(f"  Std:  {sequences_flat.std():.2f}")

# Normaliser
sequences_norm_flat = scaler.fit_transform(sequences_flat)
sequences_norm = sequences_norm_flat.reshape(n_seq, seq_len, n_feat)

print(f"\nAprès normalisation:")
print(f"  Min:  {sequences_norm_flat.min():.4f}")
print(f"  Max:  {sequences_norm_flat.max():.4f}")
print(f"  Mean: {sequences_norm_flat.mean():.6f}")
print(f"  Std:  {sequences_norm_flat.std():.6f}")

# VALIDATION CRITIQUE
if abs(sequences_norm_flat.max()) > 5:
    print(f"\n  ❌ ERREUR : Max = {sequences_norm_flat.max():.2f} (devrait être < 5)")
    print(f"     → Outliers pas bien clippés !")
    print(f"     → Augmentez clip_percentile ou vérifiez les données")
elif abs(sequences_norm_flat.max()) > 3.5:
    print(f"\n  ⚠️  Max = {sequences_norm_flat.max():.2f} (un peu élevé)")
    print(f"     Devrait être ~3 max")
else:
    print(f"\n  ✅ Normalisation OK ! (Min/Max dans [-3, +3])")

# ============================================================================
# ÉTAPE 5 : SPLIT
# ============================================================================

print(f"\n{'='*70}")
print("🔀 ÉTAPE 5 : Split train/val/test")
print("="*70)

# Shuffle
np.random.seed(CONFIG['random_seed'])
indices = np.random.permutation(len(sequences_norm))

# Split
n_total = len(sequences_norm)
n_train = int(n_total * CONFIG['train_ratio'])
n_val = int(n_total * CONFIG['val_ratio'])

train_indices = indices[:n_train]
val_indices = indices[n_train:n_train + n_val]
test_indices = indices[n_train + n_val:]

train = sequences_norm[train_indices]
val = sequences_norm[val_indices]
test = sequences_norm[test_indices]

print(f"✓ Split:")
print(f"  Train: {train.shape} ({len(train):,} séq)")
print(f"  Val:   {val.shape} ({len(val):,} séq)")
print(f"  Test:  {test.shape} ({len(test):,} séq)")

# Vérification
if len(train) < 500:
    print(f"\n  ❌ Train trop petit ({len(train)} séq)")
elif len(train) < 5000:
    print(f"\n  ⚠️  Train acceptable ({len(train):,} séq)")
else:
    print(f"\n  ✅ Train suffisant ({len(train):,} séq)")

# ============================================================================
# ÉTAPE 6 : SAUVEGARDE
# ============================================================================

print(f"\n{'='*70}")
print("💾 ÉTAPE 6 : Sauvegarde")
print("="*70)

output_dir = Path("data/processed/sequences")
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / 'train.npy', train)
np.save(output_dir / 'val.npy', val)
np.save(output_dir / 'test.npy', test)

with open(output_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

import json
config_save = {
    **CONFIG,
    'n_sequences': int(n_total),
    'n_train': int(len(train)),
    'n_val': int(len(val)),
    'n_test': int(len(test))
}

with open(output_dir / 'config.json', 'w') as f:
    json.dump(config_save, f, indent=2)

print(f"✓ Fichiers sauvegardés dans {output_dir}/")

# ============================================================================
# ÉTAPE 7 : VISUALISATIONS
# ============================================================================

print(f"\n{'='*70}")
print("📊 ÉTAPE 7 : Visualisations")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution normalisée
axes[0, 0].hist(sequences_norm.flatten(), bins=100, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Mean=0')
axes[0, 0].axvline(-3, color='orange', linestyle='--', alpha=0.5, label='±3σ')
axes[0, 0].axvline(3, color='orange', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Distribution Normalisée', fontweight='bold')
axes[0, 0].set_xlabel('Valeur')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Exemples de séquences
for i in range(5):
    axes[0, 1].plot(train[i, :, 0], alpha=0.6, linewidth=1)
axes[0, 1].set_title('5 Séquences d\'Entraînement', fontweight='bold')
axes[0, 1].set_xlabel('Position')
axes[0, 1].set_ylabel('Valeur Normalisée')
axes[0, 1].grid(True, alpha=0.3)

# 3. Stats par séquence
means = train.mean(axis=1).flatten()
stds = train.std(axis=1).flatten()

axes[1, 0].scatter(means, stds, alpha=0.3, s=10)
axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
axes[1, 0].axhline(1, color='red', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Mean vs Std par Séquence', fontweight='bold')
axes[1, 0].set_xlabel('Mean')
axes[1, 0].set_ylabel('Std')
axes[1, 0].grid(True, alpha=0.3)

# 4. Distribution des longueurs
axes[1, 1].text(0.5, 0.7, f"✓ Séquences créées", ha='center', fontsize=14, fontweight='bold')
axes[1, 1].text(0.5, 0.5, f"Total: {n_total:,}", ha='center', fontsize=12)
axes[1, 1].text(0.5, 0.4, f"Train: {len(train):,}", ha='center', fontsize=10)
axes[1, 1].text(0.5, 0.3, f"Val: {len(val):,}", ha='center', fontsize=10)
axes[1, 1].text(0.5, 0.2, f"Test: {len(test):,}", ha='center', fontsize=10)

# Validation status
if abs(sequences_norm_flat.max()) <= 3.5 and len(train) >= 500:
    axes[1, 1].text(0.5, 0.05, "✅ PRÊT POUR ENTRAÎNEMENT", 
                   ha='center', fontsize=12, color='green', fontweight='bold')
else:
    axes[1, 1].text(0.5, 0.05, "⚠️ Vérifier les données", 
                   ha='center', fontsize=12, color='orange', fontweight='bold')

axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'preparation_summary.png', dpi=150)
print(f"✓ Sauvegardé: {output_dir / 'preparation_summary.png'}")
plt.close()

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

print(f"\n{'='*70}")
print("🎉 PRÉPARATION TERMINÉE !")
print("="*70)

print(f"\n📊 Résumé:")
print(f"  Sources:         {len(data):,} timesteps")
print(f"  Séquences:       {n_total:,}")
print(f"  Train/Val/Test:  {len(train):,}/{len(val):,}/{len(test):,}")

print(f"\n🔍 Validation:")
print(f"  Normalisation:   ", end="")
if abs(sequences_norm_flat.max()) <= 3.5:
    print("✅ OK")
else:
    print(f"❌ Max={sequences_norm_flat.max():.2f} (trop élevé)")

print(f"  Quantité data:   ", end="")
if len(train) >= 5000:
    print(f"✅ Excellent ({len(train):,} séq)")
elif len(train) >= 500:
    print(f"⚠️  OK ({len(train):,} séq)")
else:
    print(f"❌ Insuffisant ({len(train):,} séq)")

print(f"\n🚀 Prochaine étape:")
if abs(sequences_norm_flat.max()) <= 3.5 and len(train) >= 500:
    print(f"  ✅ Lancez l'entraînement:")
    print(f"     python scripts/train_vae.py")
else:
    print(f"  ⚠️  Problèmes détectés - corrigez d'abord")

print(f"\n{'='*70}")