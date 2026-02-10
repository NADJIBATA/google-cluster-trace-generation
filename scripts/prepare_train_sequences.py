"""
Préparation des Séquences pour le VAE
======================================

Ce script prend vos séries temporelles et crée des séquences 
prêtes pour l'entraînement du VAE.

Étapes:
1. Charger les données de séries temporelles
2. Créer des séquences avec fenêtre glissante
3. Normaliser les données
4. Split train/val/test
5. Sauvegarder tout
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys

# Ajouter le chemin du module
sys.path.append('.')

print("="*70)
print("🔧 PRÉPARATION DES SÉQUENCES POUR LE VAE")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'sequence_length': 288,      # Longueur des séquences (288 × 5min = 24h)
    'stride': 12,                # Pas de glissement (12 × 5min = 1h entre chaque séquence)
    'train_ratio': 0.7,          # 70% pour entraînement
    'val_ratio': 0.15,           # 15% pour validation
    'test_ratio': 0.15,          # 15% pour test
    'random_seed': 42,           # Pour reproductibilité
    
    # Input file - MODIFIEZ SI NÉCESSAIRE
    # Use relative path (no leading slash) so the script finds files in the repo
    'input_file': 'data/processed/time_series_dt5min.csv',
}

print(f"\n⚙️  Configuration:")
print(f"   Longueur séquence: {CONFIG['sequence_length']} timesteps")
print(f"   Durée séquence: {CONFIG['sequence_length'] * 5 / 60:.1f} heures (avec Δt=5min)")
print(f"   Stride: {CONFIG['stride']} timesteps ({CONFIG['stride'] * 5} minutes)")
print(f"   Split: {CONFIG['train_ratio']:.0%}/{CONFIG['val_ratio']:.0%}/{CONFIG['test_ratio']:.0%}")

# Estimation du nombre de séquences
print(f"\n💡 Avec ces paramètres:")
print(f"   - stride=12 : nouvelle séquence toutes les heures")
print(f"   - Si vous avez 30 jours de données → ~700 séquences")
print(f"   - Si vous avez 7 jours  → ~170 séquences")
print(f"   - Si vous avez 3 mois   → ~2100 séquences")

# ============================================================================
# ÉTAPE 1 : CHARGER LES DONNÉES
# ============================================================================

print(f"\n{'='*70}")
print("📂 ÉTAPE 1 : Chargement des données")
print("="*70)

INPUT_FILE = CONFIG['input_file']

try:
    # Charger la série temporelle
    df_ts = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    print(f"✓ Chargé: {INPUT_FILE}")
    print(f"  Shape: {df_ts.shape}")
    print(f"  Colonnes: {df_ts.columns.tolist()}")
    print(f"  Période: {df_ts.index.min()} → {df_ts.index.max()}")
    print(f"  Durée: {(df_ts.index.max() - df_ts.index.min()).days} jours")
    
    # Détecter la colonne de données
    if 'job_count' in df_ts.columns:
        data_col = 'job_count'
    elif 'arrival_rate' in df_ts.columns:
        data_col = 'arrival_rate'
    elif 'num_arrivals' in df_ts.columns:
        data_col = 'num_arrivals'
    else:
        # Prendre la première colonne numérique
        numeric_cols = df_ts.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            data_col = numeric_cols[0]
            print(f"\n⚠️  Colonne auto-détectée: '{data_col}'")
        else:
            raise ValueError(f"Aucune colonne numérique trouvée. Colonnes: {df_ts.columns.tolist()}")
    
    print(f"\n  Colonne utilisée: '{data_col}'")
    arrival_rates = df_ts[data_col].values
    
    print(f"\n  Statistiques {data_col}:")
    print(f"    N points:   {len(arrival_rates):,}")
    print(f"    Min:        {arrival_rates.min():.2f}")
    print(f"    Max:        {arrival_rates.max():.2f}")
    print(f"    Mean:       {arrival_rates.mean():.2f}")
    print(f"    Std:        {arrival_rates.std():.2f}")
    print(f"    Médiane:    {np.median(arrival_rates):.2f}")
    
    # Vérifier s'il y a assez de données
    min_length_needed = CONFIG['sequence_length']
    if len(arrival_rates) < min_length_needed:
        raise ValueError(
            f"Pas assez de données ! "
            f"Vous avez {len(arrival_rates)} points, "
            f"mais il faut au moins {min_length_needed} pour créer une séquence."
        )
    
    # Calculer le nombre de séquences qu'on va créer
    n_sequences_expected = (len(arrival_rates) - CONFIG['sequence_length']) // CONFIG['stride'] + 1
    print(f"\n  Séquences attendues: {n_sequences_expected:,}")
    
    if n_sequences_expected < 100:
        print(f"\n  ⚠️  ATTENTION : Seulement {n_sequences_expected} séquences !")
        print(f"     Recommandé : au moins 500-1000 séquences")
        print(f"     Solutions :")
        print(f"       - Réduire sequence_length (ex: 144 au lieu de 288)")
        print(f"       - Réduire stride (ex: 6 au lieu de 12)")
        print(f"       - Utiliser plus de données")
    
except FileNotFoundError:
    print(f"❌ Fichier non trouvé: {INPUT_FILE}")
    print(f"\n💡 Fichiers disponibles:")
    
    # Chercher des fichiers time_series
    time_series_dir = Path("data/processed/")
    if time_series_dir.exists():
        files = list(time_series_dir.glob("*.csv"))
        if files:
            print(f"   Trouvés dans {time_series_dir}:")
            for f in files:
                print(f"     - {f.name}")
            print(f"\n   Modifiez CONFIG['input_file'] avec le bon chemin.")
        else:
            print(f"   Aucun fichier .csv trouvé dans {time_series_dir}")
    else:
        print(f"   Le dossier {time_series_dir} n'existe pas.")
        print(f"\n   Vous devez d'abord créer la série temporelle avec:")
        print(f"     python src/data/builder.py")
    
    sys.exit(1)

except Exception as e:
    print(f"❌ Erreur : {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# ÉTAPE 2 : CRÉER LES SÉQUENCES
# ============================================================================

print(f"\n{'='*70}")
print("✂️  ÉTAPE 2 : Création des séquences")
print("="*70)

def create_sequences(data, seq_length, stride):
    """
    Crée des séquences avec fenêtre glissante.
    
    Args:
        data: Array 1D des valeurs
        seq_length: Longueur des séquences
        stride: Pas de glissement
    
    Returns:
        Array 3D (n_sequences, seq_length, 1)
    """
    if len(data) < seq_length:
        raise ValueError(f"Données trop courtes! {len(data)} < {seq_length}")
    
    sequences = []
    for i in range(0, len(data) - seq_length + 1, stride):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    
    # Convertir en array 3D
    sequences = np.array(sequences)
    sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)
    
    return sequences

# Créer les séquences
print(f"Création avec fenêtre glissante...")
print(f"  Longueur: {CONFIG['sequence_length']}")
print(f"  Stride: {CONFIG['stride']}")

sequences = create_sequences(
    arrival_rates, 
    CONFIG['sequence_length'], 
    CONFIG['stride']
)

print(f"\n✓ Séquences créées: {sequences.shape}")
print(f"  N séquences:  {sequences.shape[0]:,}")
print(f"  Longueur:     {sequences.shape[1]}")
print(f"  N features:   {sequences.shape[2]}")

# Vérifier la couverture
coverage = (sequences.shape[0] * CONFIG['stride']) / len(arrival_rates) * 100
print(f"  Couverture:   {coverage:.1f}% des données utilisées")

# Alertes
if sequences.shape[0] < 100:
    print(f"\n  ⚠️  ATTENTION : Seulement {sequences.shape[0]} séquences créées !")
    print(f"     C'est très peu pour entraîner un VAE.")
    print(f"     Recommandations :")
    print(f"       - Réduire stride à 6 ou 1")
    print(f"       - Ou réduire sequence_length à 144")
elif sequences.shape[0] < 500:
    print(f"\n  ℹ️  {sequences.shape[0]} séquences : c'est peu mais acceptable")
else:
    print(f"\n  ✅ {sequences.shape[0]} séquences : bon nombre !")

# Visualiser quelques séquences
print(f"\n📊 Visualisation de 5 séquences aléatoires...")

fig, axes = plt.subplots(5, 1, figsize=(14, 10))
np.random.seed(42)
n_samples = min(5, len(sequences))
sample_indices = np.random.choice(len(sequences), n_samples, replace=False)

for idx, seq_idx in enumerate(sample_indices):
    axes[idx].plot(sequences[seq_idx, :, 0], linewidth=1.5)
    axes[idx].set_ylabel('Valeur')
    axes[idx].set_title(f'Séquence #{seq_idx} (longueur={CONFIG["sequence_length"]})', 
                       fontsize=10, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axhline(y=sequences[seq_idx, :, 0].mean(), 
                     color='red', linestyle='--', alpha=0.5, label='Moyenne')
    if idx == 0:
        axes[idx].legend()

axes[-1].set_xlabel(f'Position dans la séquence (0-{CONFIG["sequence_length"]-1})')
plt.tight_layout()

output_dir_temp = Path("data/processed/sequences")
output_dir_temp.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir_temp / 'sample_sequences.png', dpi=150)
print(f"✓ Sauvegardé: {output_dir_temp / 'sample_sequences.png'}")
plt.close()

# ============================================================================
# ÉTAPE 3 : NORMALISATION
# ============================================================================

print(f"\n{'='*70}")
print("📏 ÉTAPE 3 : Normalisation")
print("="*70)

# Créer le scaler
scaler = StandardScaler()

# Reshape pour scaler: (n_sequences * seq_length, n_features)
n_sequences, seq_length, n_features = sequences.shape
sequences_flat = sequences.reshape(-1, n_features)

print(f"Avant normalisation:")
print(f"  Shape: {sequences_flat.shape}")
print(f"  Mean:  {sequences_flat.mean():.4f}")
print(f"  Std:   {sequences_flat.std():.4f}")
print(f"  Min:   {sequences_flat.min():.4f}")
print(f"  Max:   {sequences_flat.max():.4f}")

# Fit et transform
sequences_norm_flat = scaler.fit_transform(sequences_flat)

# Reshape back
sequences_norm = sequences_norm_flat.reshape(n_sequences, seq_length, n_features)

print(f"\nAprès normalisation:")
print(f"  Mean:  {sequences_norm_flat.mean():.6f} (devrait être ~0)")
print(f"  Std:   {sequences_norm_flat.std():.6f} (devrait être ~1)")
print(f"  Min:   {sequences_norm_flat.min():.4f}")
print(f"  Max:   {sequences_norm_flat.max():.4f}")

# ============================================================================
# ÉTAPE 4 : SPLIT TRAIN/VAL/TEST
# ============================================================================

print(f"\n{'='*70}")
print("🔀 ÉTAPE 4 : Split train/val/test")
print("="*70)

# Shuffle puis split
np.random.seed(CONFIG['random_seed'])
indices = np.random.permutation(len(sequences_norm))

# Calculer indices
n_total = len(sequences_norm)
n_train = int(n_total * CONFIG['train_ratio'])
n_val = int(n_total * CONFIG['val_ratio'])

# Split
train_indices = indices[:n_train]
val_indices = indices[n_train:n_train + n_val]
test_indices = indices[n_train + n_val:]

train_sequences = sequences_norm[train_indices]
val_sequences = sequences_norm[val_indices]
test_sequences = sequences_norm[test_indices]

print(f"✓ Split effectué:")
print(f"  Train: {train_sequences.shape} ({len(train_sequences)/n_total*100:.1f}%)")
print(f"  Val:   {val_sequences.shape} ({len(val_sequences)/n_total*100:.1f}%)")
print(f"  Test:  {test_sequences.shape} ({len(test_sequences)/n_total*100:.1f}%)")

# Vérification finale
if len(train_sequences) < 50:
    print(f"\n  ⚠️  WARNING : Seulement {len(train_sequences)} échantillons d'entraînement !")
    print(f"     Le VAE risque de ne pas bien apprendre.")

# ============================================================================
# ÉTAPE 5 : SAUVEGARDE
# ============================================================================

print(f"\n{'='*70}")
print("💾 ÉTAPE 5 : Sauvegarde")
print("="*70)

# Créer dossier de sortie
output_dir = Path("data/processed/sequences")
output_dir.mkdir(parents=True, exist_ok=True)

# Sauvegarder les séquences
np.save(output_dir / 'train.npy', train_sequences)
np.save(output_dir / 'val.npy', val_sequences)
np.save(output_dir / 'test.npy', test_sequences)

print(f"✓ Séquences sauvegardées:")
print(f"  {output_dir / 'train.npy'}")
print(f"  {output_dir / 'val.npy'}")
print(f"  {output_dir / 'test.npy'}")

# Sauvegarder le scaler
with open(output_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✓ Scaler sauvegardé: {output_dir / 'scaler.pkl'}")

# Sauvegarder la config
import json
config_to_save = CONFIG.copy()
config_to_save['n_sequences'] = int(n_total)
config_to_save['n_train'] = int(len(train_sequences))
config_to_save['n_val'] = int(len(val_sequences)s)
config_to_save['n_test'] = int(len(test_sequences))
config_to_save['data_source'] = str(INPUT_FILE)
config_to_save['n_features'] = int(n_features)

with open(output_dir / 'config.json', 'w') as f:
    json.dump(config_to_save, f, indent=2)

print(f"✓ Configuration sauvegardée: {output_dir / 'config.json'}")

# ============================================================================
# ÉTAPE 6 : VALIDATION
# ============================================================================

print(f"\n{'='*70}")
print("✅ ÉTAPE 6 : Validation")
print("="*70)

# Tester le chargement
train_loaded = np.load(output_dir / 'train.npy')
print(f"✓ Test de chargement: {train_loaded.shape}")

# Vérifier la dénormalisation
sample_seq_norm = train_sequences[0]  # (seq_length, 1)
sample_seq_denorm = scaler.inverse_transform(sample_seq_norm)

print(f"\n✓ Test de dénormalisation:")
print(f"  Normalisé:    min={sample_seq_norm.min():.2f}, max={sample_seq_norm.max():.2f}")
print(f"  Dénormalisé:  min={sample_seq_denorm.min():.2f}, max={sample_seq_denorm.max():.2f}")

# Visualiser distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Distribution avant normalisation
axes[0].hist(sequences.flatten(), bins=50, alpha=0.7, edgecolor='black')
axes[0].set_title('Distribution Avant Normalisation', fontweight='bold')
axes[0].set_xlabel(data_col)
axes[0].set_ylabel('Fréquence')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(sequences.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean={sequences.mean():.1f}')
axes[0].legend()

# Distribution après normalisation
axes[1].hist(sequences_norm.flatten(), bins=50, alpha=0.7, edgecolor='black')
axes[1].set_title('Distribution Après Normalisation (N(0,1))', fontweight='bold')
axes[1].set_xlabel('Valeur Normalisée')
axes[1].set_ylabel('Fréquence')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Mean=0')
axes[1].legend()

plt.tight_layout()
plt.savefig(output_dir / 'normalization_check.png', dpi=150)
print(f"\n✓ Visualisation sauvegardée: {output_dir / 'normalization_check.png'}")
plt.close()

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

print(f"\n{'='*70}")
print("🎉 PRÉPARATION TERMINÉE AVEC SUCCÈS !")
print("="*70)

print(f"\n📊 Résumé:")
print(f"  Données sources:     {len(arrival_rates):,} timesteps")
print(f"  Période:             {(df_ts.index.max() - df_ts.index.min()).days} jours")
print(f"  Séquences créées:    {n_total:,} séquences de longueur {CONFIG['sequence_length']}")
print(f"  Train:               {len(train_sequences):,} séquences")
print(f"  Validation:          {len(val_sequences):,} séquences")
print(f"  Test:                {len(test_sequences):,} séquences")

print(f"\n📁 Fichiers créés dans {output_dir}:")
print(f"  ✓ train.npy         {train_sequences.shape}")
print(f"  ✓ val.npy           {val_sequences.shape}")
print(f"  ✓ test.npy          {test_sequences.shape}")
print(f"  ✓ scaler.pkl")
print(f"  ✓ config.json")
print(f"  ✓ sample_sequences.png")
print(f"  ✓ normalization_check.png")

print(f"\n🚀 Prochaine étape:")
if len(train_sequences) >= 500:
    print(f"  ✅ Vous avez assez de données !")
    print(f"  Lancez: python scripts/train_lstm_vae.py")
elif len(train_sequences) >= 100:
    print(f"  ⚠️  Vous avez peu de données ({len(train_sequences)} séquences)")
    print(f"  Vous pouvez quand même essayer:")
    print(f"    python scripts/train_lstm_vae.py")
    print(f"  Mais pour de meilleurs résultats, réduisez stride ou augmentez les données.")
else:
    print(f"  ❌ Pas assez de données ({len(train_sequences)} séquences)")
    print(f"  Recommandations:")
    print(f"    - Réduire stride (ex: stride=6)")
    print(f"    - Réduire sequence_length (ex: sequence_length=144)")
    print(f"    - Utiliser plus de données sources")

print(f"\n{'='*70}")