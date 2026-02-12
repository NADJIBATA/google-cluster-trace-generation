"""
Test et Évaluation du LSTM-VAE
================================

Ce script teste le modèle entraîné et diagnostique les problèmes potentiels.

Tests effectués:
1. Reconstruction des données de test
2. Diversité de l'espace latent
3. Qualité des générations
4. Diagnostic du posterior collapse
5. Métriques de performance
"""

import numpy as np
import torch
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys
from scipy import stats

sys.path.append('.')

from src.models.vae_lstm import LSTMVAE

print("="*70)
print("🧪 TEST ET ÉVALUATION DU LSTM-VAE")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'checkpoint_path': 'checkpoints/lstm_vae_best.pth',
    'test_data_path': 'data/processed/sequences/test.npy',
    'scaler_path': 'data/processed/sequences/scaler.pkl',
    'output_dir': 'results/evaluation',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"\n⚙️  Configuration:")
print(f"   Device: {CONFIG['device']}")
print(f"   Checkpoint: {CONFIG['checkpoint_path']}")

# Créer dossier de sortie
output_dir = Path(CONFIG['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TEST 1: CHARGEMENT DU MODÈLE
# ============================================================================

print(f"\n{'='*70}")
print("📂 TEST 1: Chargement du modèle")
print("="*70)

checkpoint_path = Path(CONFIG['checkpoint_path'])
if not checkpoint_path.exists():
    print(f"❌ Checkpoint non trouvé: {checkpoint_path}")
    print(f"\nFichiers disponibles dans checkpoints/:")
    if Path('checkpoints').exists():
        for f in Path('checkpoints').glob('*.pth'):
            print(f"   - {f.name}")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])

print(f"✓ Checkpoint chargé:")
print(f"   Epoch: {checkpoint['epoch']}")
print(f"   Val loss: {checkpoint['val_loss']:.4f}")

# Reconstruire le modèle
model_config = checkpoint['vae_config']
model = LSTMVAE(
    input_size=1,
    sequence_length=checkpoint['config']['sequence_length'],
    **model_config
)

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(CONFIG['device'])
model.eval()

print(f"\n✓ Modèle reconstruit:")
print(f"   Paramètres: {model.count_parameters():,}")
print(f"   Latent dim: {model_config['latent_dim']}")

# ============================================================================
# TEST 2: DIAGNOSTIC DU POSTERIOR COLLAPSE
# ============================================================================

print(f"\n{'='*70}")
print("🔍 TEST 2: Diagnostic du Posterior Collapse")
print("="*70)

# Analyser l'historique
if 'history' in checkpoint:
    history = checkpoint['history']
    
    final_kl = history['val_kl'][-1] if isinstance(history['val_kl'], list) else history['val_kl']
    final_recon = history['val_recon'][-1] if isinstance(history['val_recon'], list) else history['val_recon']
    
    print(f"\n📊 Métriques finales:")
    print(f"   KL divergence: {final_kl:.4f}")
    print(f"   Reconstruction: {final_recon:.2f}")
    
    if final_kl > 0:
        ratio = final_recon / final_kl
        print(f"   Ratio R/KL: {ratio:.2f}")
    
    # Diagnostic
    print(f"\n🔍 Diagnostic:")
    if final_kl < 0.01:
        print(f"   ❌ SEVERE COLLAPSE: KL ≈ 0")
        print(f"      Le modèle ignore complètement l'espace latent !")
        collapse_severity = "SEVERE"
    elif final_kl < 0.1:
        print(f"   ❌ POSTERIOR COLLAPSE: KL très faible ({final_kl:.4f})")
        print(f"      Le modèle utilise très peu l'espace latent")
        collapse_severity = "HIGH"
    elif final_kl < 1.0:
        print(f"   ⚠️  RISQUE DE COLLAPSE: KL faible ({final_kl:.2f})")
        print(f"      Surveillez la diversité des générations")
        collapse_severity = "MEDIUM"
    elif final_kl > 20:
        print(f"   ⚠️  KL TRÈS ÉLEVÉE: {final_kl:.2f}")
        print(f"      Sur-régularisation possible")
        collapse_severity = "OVER_REGULARIZED"
    else:
        print(f"   ✅ KL SAINE: {final_kl:.2f}")
        print(f"      L'espace latent est bien utilisé")
        collapse_severity = "NONE"
else:
    print("⚠️  Historique non disponible dans le checkpoint")
    collapse_severity = "UNKNOWN"

# ============================================================================
# TEST 3: RECONSTRUCTION SUR DONNÉES DE TEST
# ============================================================================

print(f"\n{'='*70}")
print("🔄 TEST 3: Reconstruction sur données de test")
print("="*70)

test_path = Path(CONFIG['test_data_path'])
if test_path.exists():
    test_data = np.load(test_path)
    print(f"✓ Données de test chargées: {test_data.shape}")
    
    # Prendre quelques échantillons
    n_samples = min(10, len(test_data))
    test_samples = test_data[:n_samples]
    
    # Reconstruction
    with torch.no_grad():
        x_test = torch.FloatTensor(test_samples).to(CONFIG['device'])
        x_recon, mu, logvar = model(x_test)
        
        x_recon = x_recon.cpu().numpy()
        mu = mu.cpu().numpy()
        logvar = logvar.cpu().numpy()
    
    # Calculer MSE
    mse = np.mean((test_samples - x_recon) ** 2)
    print(f"\n📊 Erreur de reconstruction:")
    print(f"   MSE: {mse:.4f}")
    print(f"   RMSE: {np.sqrt(mse):.4f}")
    
    # Visualiser quelques reconstructions
    fig, axes = plt.subplots(min(3, n_samples), 1, figsize=(14, 3*min(3, n_samples)))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(min(3, n_samples)):
        ax = axes[i] if n_samples > 1 else axes[0]
        
        ax.plot(test_samples[i].flatten(), label='Original', linewidth=2, alpha=0.7)
        ax.plot(x_recon[i].flatten(), label='Reconstruction', linewidth=2, alpha=0.7, linestyle='--')
        ax.set_title(f'Échantillon {i+1}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reconstruction_test.png', dpi=150)
    print(f"\n✓ Visualisation sauvegardée: {output_dir / 'reconstruction_test.png'}")
    plt.close()
    
else:
    print(f"⚠️  Données de test non trouvées: {test_path}")
    test_data = None

# ============================================================================
# TEST 4: DIVERSITÉ DE L'ESPACE LATENT
# ============================================================================

print(f"\n{'='*70}")
print("🎨 TEST 4: Diversité de l'espace latent")
print("="*70)

# Charger données d'entraînement pour analyser l'espace latent
train_path = Path('data/processed/sequences/train.npy')
if train_path.exists():
    train_data = np.load(train_path)
    print(f"✓ Données d'entraînement chargées: {train_data.shape}")
    
    # Encoder toutes les séquences
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
    print(f"✓ Espace latent: {latent_codes.shape}")
    
    # Statistiques
    latent_mean = latent_codes.mean(axis=0)
    latent_std = latent_codes.std(axis=0)
    latent_var = latent_codes.var(axis=0)
    
    print(f"\n📊 Statistiques de l'espace latent:")
    print(f"   Mean (global): {latent_mean.mean():.4f}")
    print(f"   Std (global): {latent_std.mean():.4f}")
    print(f"   Variance (avg): {latent_var.mean():.4f}")
    print(f"   Variance (min): {latent_var.min():.4f}")
    print(f"   Variance (max): {latent_var.max():.4f}")
    
    # Diagnostic de collapse basé sur variance
    inactive_dims = np.sum(latent_var < 0.01)
    print(f"\n🔍 Dimensions inactives (var < 0.01): {inactive_dims}/{len(latent_var)}")
    
    if inactive_dims > len(latent_var) * 0.5:
        print(f"   ❌ Plus de 50% des dimensions sont inactives !")
        print(f"      → Posterior collapse confirmé")
    elif inactive_dims > len(latent_var) * 0.2:
        print(f"   ⚠️  {inactive_dims} dimensions inactives")
        print(f"      → Sous-utilisation de l'espace latent")
    else:
        print(f"   ✅ Espace latent bien utilisé")
    
    # Visualiser variance par dimension
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Variance
    ax = axes[0]
    ax.bar(range(len(latent_var)), latent_var)
    ax.set_title('Variance par Dimension Latente', fontweight='bold')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Variance')
    ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Seuil inactif')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distribution de la première dimension
    ax = axes[1]
    ax.hist(latent_codes[:, 0], bins=50, edgecolor='black', alpha=0.7)
    ax.set_title('Distribution - Dimension 0', fontweight='bold')
    ax.set_xlabel('Valeur')
    ax.set_ylabel('Fréquence')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_space_analysis.png', dpi=150)
    print(f"\n✓ Analyse sauvegardée: {output_dir / 'latent_space_analysis.png'}")
    plt.close()
    
else:
    print(f"⚠️  Données d'entraînement non trouvées")
    latent_codes = None

# ============================================================================
# TEST 5: QUALITÉ DES GÉNÉRATIONS
# ============================================================================

print(f"\n{'='*70}")
print("🎲 TEST 5: Qualité des générations")
print("="*70)

# Générer plusieurs échantillons
n_gen = 20
print(f"Génération de {n_gen} échantillons...")

generated_samples = []
with torch.no_grad():
    for i in range(n_gen):
        z = torch.randn(1, model.latent_dim).to(CONFIG['device'])
        x_gen = model.decode(z)
        generated_samples.append(x_gen.cpu().numpy()[0])

generated_samples = np.array(generated_samples)
print(f"✓ Généré: {generated_samples.shape}")

# Statistiques
gen_mean = generated_samples.mean()
gen_std = generated_samples.std()
gen_min = generated_samples.min()
gen_max = generated_samples.max()

print(f"\n📊 Statistiques des générations (normalisées):")
print(f"   Mean: {gen_mean:.4f}")
print(f"   Std:  {gen_std:.4f}")
print(f"   Min:  {gen_min:.4f}")
print(f"   Max:  {gen_max:.4f}")

# Vérifier diversité
unique_values = np.unique(generated_samples.round(2))
print(f"\n🎨 Diversité:")
print(f"   Valeurs uniques: {len(unique_values)}")

if len(unique_values) < 10:
    print(f"   ❌ TRÈS PEU DE DIVERSITÉ !")
    print(f"      Les générations sont presque identiques")
    diversity = "LOW"
elif len(unique_values) < 50:
    print(f"   ⚠️  Diversité limitée")
    diversity = "MEDIUM"
else:
    print(f"   ✅ Bonne diversité")
    diversity = "HIGH"

# Calculer la variance inter-échantillons
inter_sample_var = np.var([s.mean() for s in generated_samples])
print(f"   Variance inter-échantillons: {inter_sample_var:.4f}")

if inter_sample_var < 0.01:
    print(f"   ❌ Tous les échantillons sont quasi-identiques !")
elif inter_sample_var < 0.1:
    print(f"   ⚠️  Faible variation entre échantillons")
else:
    print(f"   ✅ Variation satisfaisante")

# Visualiser
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Quelques générations
ax = axes[0]
for i in range(min(5, n_gen)):
    ax.plot(generated_samples[i].flatten(), alpha=0.7, label=f'Gen {i+1}')
ax.set_title('Échantillons Générés (normalisés)', fontweight='bold')
ax.set_xlabel('Timestep')
ax.set_ylabel('Valeur')
ax.legend()
ax.grid(True, alpha=0.3)

# Distribution
ax = axes[1]
ax.hist(generated_samples.flatten(), bins=50, edgecolor='black', alpha=0.7)
ax.set_title('Distribution des Valeurs Générées', fontweight='bold')
ax.set_xlabel('Valeur')
ax.set_ylabel('Fréquence')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'generated_samples.png', dpi=150)
print(f"\n✓ Générations sauvegardées: {output_dir / 'generated_samples.png'}")
plt.close()

# ============================================================================
# TEST 6: COMPARAISON AVEC DONNÉES RÉELLES
# ============================================================================

print(f"\n{'='*70}")
print("📊 TEST 6: Comparaison avec données réelles")
print("="*70)

if train_data is not None:
    # Statistiques des données réelles (normalisées)
    real_mean = train_data.mean()
    real_std = train_data.std()
    
    print(f"\n📊 Données réelles (normalisées):")
    print(f"   Mean: {real_mean:.4f}")
    print(f"   Std:  {real_std:.4f}")
    
    print(f"\n📊 Données générées:")
    print(f"   Mean: {gen_mean:.4f}")
    print(f"   Std:  {gen_std:.4f}")
    
    # Test de Kolmogorov-Smirnov
    ks_stat, ks_pvalue = stats.ks_2samp(
        train_data.flatten()[:10000],  # Échantillon
        generated_samples.flatten()
    )
    
    print(f"\n📈 Test de Kolmogorov-Smirnov:")
    print(f"   Statistique: {ks_stat:.4f}")
    print(f"   P-value: {ks_pvalue:.4f}")
    
    if ks_pvalue > 0.05:
        print(f"   ✅ Les distributions sont statistiquement similaires")
    else:
        print(f"   ⚠️  Les distributions diffèrent significativement")

# ============================================================================
# RAPPORT FINAL
# ============================================================================

print(f"\n{'='*70}")
print("📋 RAPPORT FINAL")
print("="*70)

report = {
    'collapse_severity': collapse_severity,
    'kl_divergence': float(final_kl) if 'final_kl' in locals() else None,
    'reconstruction_mse': float(mse) if 'mse' in locals() else None,
    'diversity': diversity if 'diversity' in locals() else None,
    'inactive_dimensions': int(inactive_dims) if 'inactive_dims' in locals() else None,
    'total_dimensions': int(model.latent_dim),
    'inter_sample_variance': float(inter_sample_var) if 'inter_sample_var' in locals() else None,
}

print(f"\n🎯 Résumé:")
print(f"   Posterior Collapse: {collapse_severity}")
print(f"   KL Divergence: {report['kl_divergence']:.4f}" if report['kl_divergence'] else "   KL Divergence: N/A")
print(f"   MSE Reconstruction: {report['reconstruction_mse']:.4f}" if report['reconstruction_mse'] else "   MSE: N/A")
print(f"   Diversité: {report['diversity']}" if report['diversity'] else "   Diversité: N/A")
print(f"   Dimensions actives: {report['total_dimensions'] - report['inactive_dimensions']}/{report['total_dimensions']}" if report['inactive_dimensions'] is not None else "")

# Verdict global
print(f"\n🏆 VERDICT GLOBAL:")

score = 0
issues = []

if collapse_severity == "NONE":
    score += 3
    print(f"   ✅ Pas de posterior collapse")
elif collapse_severity in ["MEDIUM", "UNKNOWN"]:
    score += 1
    issues.append("Risque de collapse")
    print(f"   ⚠️  Risque de posterior collapse")
else:
    issues.append("Posterior collapse détecté")
    print(f"   ❌ Posterior collapse détecté")

if diversity == "HIGH":
    score += 2
    print(f"   ✅ Bonne diversité des générations")
elif diversity == "MEDIUM":
    score += 1
    issues.append("Diversité limitée")
    print(f"   ⚠️  Diversité limitée")
else:
    issues.append("Très peu de diversité")
    print(f"   ❌ Très peu de diversité")

if report['inactive_dimensions'] is not None:
    active_ratio = 1 - (report['inactive_dimensions'] / report['total_dimensions'])
    if active_ratio > 0.8:
        score += 1
        print(f"   ✅ Espace latent bien utilisé")
    elif active_ratio > 0.5:
        issues.append("Sous-utilisation de l'espace latent")
        print(f"   ⚠️  Sous-utilisation de l'espace latent")
    else:
        issues.append("Espace latent très peu utilisé")
        print(f"   ❌ Espace latent très peu utilisé")

print(f"\n⭐ Score: {score}/6")

if score >= 5:
    print(f"   ✅ EXCELLENT - Le modèle fonctionne très bien !")
    recommendation = "Le modèle est prêt pour la génération de scénarios"
elif score >= 3:
    print(f"   ⚠️  ACCEPTABLE - Quelques améliorations possibles")
    recommendation = "Le modèle fonctionne mais pourrait être amélioré"
else:
    print(f"   ❌ PROBLÉMATIQUE - Ré-entraînement recommandé")
    recommendation = "Utilisez train_lstm_vae_anticollapse.py"

if issues:
    print(f"\n⚠️  Problèmes détectés:")
    for issue in issues:
        print(f"   - {issue}")

print(f"\n💡 Recommandation:")
print(f"   {recommendation}")

# Sauvegarder le rapport
report['score'] = score
report['recommendation'] = recommendation
report['issues'] = issues

with open(output_dir / 'evaluation_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Rapport sauvegardé: {output_dir / 'evaluation_report.json'}")

print(f"\n{'='*70}")
print("📁 Fichiers créés:")
print(f"   {output_dir}/reconstruction_test.png")
print(f"   {output_dir}/latent_space_analysis.png")
print(f"   {output_dir}/generated_samples.png")
print(f"   {output_dir}/evaluation_report.json")
print("="*70)