"""
Comparaison Traces Générées vs Traces Réelles — VERSION CORRIGÉE
=================================================================

Correction principale : comparaison en espace NORMALISÉ cohérent.
Le TimeStepScaler normalise par timestep → on ne peut pas comparer
directement avec la série brute après dénormalisation.

Stratégie :
  - Métriques statistiques → espace normalisé (les deux côtés)
  - Visualisations → dénormalisé (pour l'interprétation)
"""

import os, sys
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')
import json

sys.path.append('.')

# ============================================================================
# TIMESTEP SCALER (copie fidèle de votre classe)
# ============================================================================

class TimeStepScaler:
    def __init__(self):
        self.mean_ = None
        self.var_  = None
        self.seq_len  = None
        self.n_features  = None

    def fit(self, sequences):
        seqs = np.asarray(sequences)
        if seqs.ndim != 3:
            raise ValueError("fit: sequences must be 3D")
        self.seq_len = seqs.shape[1]
        self.n_features = seqs.shape[2]
        self.mean_ = seqs.mean(axis=0)
        self.var_  = seqs.var(axis=0)
        self.var_[self.var_ == 0] = 1e-8
        return self

    def transform(self, sequences):
        seqs = np.asarray(sequences)
        if seqs.ndim == 3:
            return (seqs - self.mean_) / np.sqrt(self.var_)
        if seqs.ndim == 2 and seqs.shape == (self.seq_len, self.n_features):
            return (seqs - self.mean_) / np.sqrt(self.var_)
        raise ValueError(f"transform: unexpected shape {seqs.shape}")

    def inverse_transform(self, X):
        arr = np.asarray(X)
        if arr.ndim == 2 and arr.shape == (self.seq_len, self.n_features):
            return arr * np.sqrt(self.var_) + self.mean_
        if arr.ndim == 3:
            return arr * np.sqrt(self.var_) + self.mean_
        raise ValueError(f"inverse_transform: unexpected shape {arr.shape}")

    def fit_transform(self, sequences):
        self.fit(sequences)
        return self.transform(sequences)

# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = Path(__file__).parent
repo_root  = script_dir.parent.parent

CONFIG = {
    'real_data_path'     : repo_root / 'data' / 'processed' / 'time_series_dt5min.csv',
    'train_seq_path'     : repo_root / 'data' / 'processed' / 'sequences' / 'train.npy',
    'test_seq_path'      : repo_root / 'data' / 'processed' / 'sequences' / 'test.npy',
    'scaler_path'        : repo_root / 'data' / 'processed' / 'sequences' / 'scaler.pkl',
    'generated_dir'      : repo_root / 'data' / 'generated' / 'gen2',
    'output_dir'         : repo_root / 'results' / 'comparison',
    'delta_t_minutes'    : 5,
    'n_bins'             : 50,
}

output_dir = Path(CONFIG['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("📊 COMPARAISON TRACES GÉNÉRÉES vs TRACES RÉELLES")
print("="*70)

# ============================================================================
# CHARGEMENT DU SCALER
# ============================================================================

print(f"\n{'='*70}")
print("📂 Chargement")
print("="*70)

with open(CONFIG['scaler_path'], 'rb') as f:
    scaler = pickle.load(f)

scaler_type = type(scaler).__name__
print(f"✓ Scaler chargé: type={scaler_type}")

# Adapter selon le type de scaler
if hasattr(scaler, 'mean_') and np.ndim(scaler.mean_) == 0:
    # StandardScaler classique (scalaire)
    sc_mean = float(scaler.mean_)
    sc_std  = float(np.sqrt(scaler.var_))
elif hasattr(scaler, 'mean_') and np.ndim(scaler.mean_) >= 1:
    sc_mean = float(np.asarray(scaler.mean_).mean())
    sc_std  = float(np.sqrt(np.asarray(scaler.var_).mean()))
else:
    sc_mean, sc_std = 0.0, 1.0

print(f"  mean={sc_mean:.4f}, std={sc_std:.4f}")

# ============================================================================
# CHARGEMENT DES SÉQUENCES RÉELLES (normalisées)
# ============================================================================

train_norm = np.load(CONFIG['train_seq_path'])   # (N, seq_len, 1)
test_norm  = np.load(CONFIG['test_seq_path'])    # (N, seq_len, 1)
seq_len    = train_norm.shape[1]

print(f"\n✓ Séquences train (normalisées): {train_norm.shape}")
print(f"✓ Séquences test  (normalisées): {test_norm.shape}")
print(f"  seq_len={seq_len}")

real_norm_flat = test_norm.flatten()
print(f"\n✓ Vecteur réel normalisé: {real_norm_flat.shape}")
print(f"   mean={real_norm_flat.mean():.4f}, std={real_norm_flat.std():.4f}")

# ============================================================================
# FONCTION DE DÉNORMALISATION UNIVERSELLE
# ============================================================================

def denorm_sequence(data_norm):
    """Dénormalise avec le scaler chargé, quel que soit son type."""
    flat = np.asarray(data_norm).flatten().reshape(-1, 1)
    try:
        return np.maximum(scaler.inverse_transform(flat).flatten(), 0)
    except Exception:
        return np.maximum(flat.flatten() * sc_std + sc_mean, 0)

# ============================================================================
# CHARGEMENT DES SCÉNARIOS GÉNÉRÉS
# ============================================================================

gen_dir = Path(CONFIG['generated_dir'])

scenarios_norm   = {}   # en espace normalisé  → pour les métriques
scenarios_denorm = {}   # en espace original   → pour les visuels

def load_scenario(path, name):
    """
    Charge un scénario .npy.
    Retourne (data_norm, data_denorm).
    
    - Si les valeurs sont proches de 0 (mean < 2) → déjà normalisé
    - Sinon → déjà dénormalisé, on re-normalise via StandardScaler
    """
    data = np.load(path)
    flat = data.flatten()
    print(f"  {name}: shape={data.shape}  mean={flat.mean():.4f}  std={flat.std():.4f}")

    if abs(flat.mean()) < 3.0 and 0.1 < flat.std() < 6.0:
        # Déjà normalisé
        print(f"    → déjà normalisé ✓")
        data_norm  = flat
        data_denorm = denorm_sequence(flat)
    else:
        # Déjà dénormalisé (grandes valeurs)
        print(f"    → déjà dénormalisé, re-normalisation ✓")
        data_denorm = np.maximum(flat, 0)
        # Re-normaliser via le StandardScaler
        data_norm   = scaler.transform(flat.reshape(-1, 1)).flatten()

    return data_norm, data_denorm

for name in ['nominal', 'optimistic', 'pessimistic']:
    for candidate in [
        gen_dir  / f'scenario_{name}.npy',
        gen_dir / f'scenario_{name}.npy',
    ]:
        if candidate.exists():
            n, d = load_scenario(candidate, name)
            scenarios_norm[name]   = n
            scenarios_denorm[name] = d
            print(f"✓ '{name}' : norm_mean={n.mean():.4f}  denorm_mean={d.mean():.2f}")
            break

# Scénarios aléatoires
for candidate in [gen_dir / 'scenarios_random.npy',
                  gen_dir / 'random' / 'scenarios_random.npy']:
    if candidate.exists():
        rd = np.load(candidate)
        print(f"  random: shape={rd.shape}  mean={rd.mean():.4f}")
        # Prendre la moyenne sur les scénarios
        rd_mean = rd.mean(axis=0) if rd.ndim == 3 else rd.mean(axis=0)
        n, d = load_scenario(candidate.parent / candidate.name, 'random')
        # recalculer sur la moyenne
        rd_flat = rd.mean(axis=0).flatten()
        if abs(rd_flat.mean()) < 3.0:
            scenarios_norm['random_mean']   = rd_flat
            scenarios_denorm['random_mean'] = denorm_sequence(rd_flat)
        else:
            scenarios_denorm['random_mean'] = np.maximum(rd_flat, 0)
            scenarios_norm['random_mean']   = scaler.transform(
                rd_flat.reshape(-1,1)).flatten()
        print(f"✓ 'random_mean' : norm_mean={scenarios_norm['random_mean'].mean():.4f}  denorm_mean={scenarios_denorm['random_mean'].mean():.2f}")
        break

if not scenarios_norm:
    print("❌ Aucun scénario trouvé."); sys.exit(1)

print(f"\n✓ {len(scenarios_norm)} scénarios prêts")

# ============================================================================
# SÉRIE RÉELLE BRUTE (pour profil journalier uniquement)
# ============================================================================

df_real = pd.read_csv(CONFIG['real_data_path'], index_col=0, parse_dates=True)
real_col = next((c for c in ['job_count','arrival_rate','num_arrivals']
                 if c in df_real.columns), df_real.select_dtypes(include=[np.number]).columns[0])
real_series_raw = df_real[real_col].values

print(f"\n✓ Série brute: mean={real_series_raw.mean():.2f}, std={real_series_raw.std():.2f}")

# Normaliser la série brute avec le même StandardScaler
real_series_norm = scaler.transform(real_series_raw.reshape(-1, 1)).flatten()

# ============================================================================
# MÉTRIQUES (tout en espace normalisé cohérent)
# ============================================================================

print(f"\n{'='*70}")
print("📈 MÉTRIQUES — espace normalisé")
print("="*70)

def compute_metrics(real_flat, gen_flat, name, n_bins=50):
    r = real_flat[~np.isnan(real_flat)]
    g = gen_flat[~np.isnan(gen_flat)]

    m = {'name': name}
    for prefix, arr in [('real', r), ('gen', g)]:
        m[f'{prefix}_mean'] = float(arr.mean())
        m[f'{prefix}_std']  = float(arr.std())
        m[f'{prefix}_min']  = float(arr.min())
        m[f'{prefix}_max']  = float(arr.max())
        for p in [25, 50, 75, 95]:
            m[f'{prefix}_p{p}'] = float(np.percentile(arr, p))

    m['mean_error_pct'] = abs(m['gen_mean'] - m['real_mean']) / max(abs(m['real_mean']), 1e-6) * 100
    m['std_error_pct']  = abs(m['gen_std']  - m['real_std'])  / max(abs(m['real_std']),  1e-6) * 100

    ks, pv = stats.ks_2samp(r, g)
    m['ks_statistic'] = float(ks)
    m['ks_pvalue']    = float(pv)
    m['ks_similar']   = bool(pv > 0.05)

    wd = stats.wasserstein_distance(r, g)
    m['wasserstein_distance'] = float(wd)
    rng = m['real_max'] - m['real_min']
    m['wasserstein_normalized'] = float(wd / max(rng, 1e-6))

    all_v = np.concatenate([r, g])
    bins  = np.linspace(all_v.min(), all_v.max(), n_bins + 1)
    hr, _ = np.histogram(r, bins=bins, density=True)
    hg, _ = np.histogram(g, bins=bins, density=True)
    hr = (hr + 1e-10) / (hr + 1e-10).sum()
    hg = (hg + 1e-10) / (hg + 1e-10).sum()
    js  = float(jensenshannon(hr, hg))
    m['js_divergence']     = js
    m['js_similarity_pct'] = (1 - js) * 100

    def acf(x, lag):
        return float(np.corrcoef(x[:-lag], x[lag:])[0,1]) if len(x) > lag else 0.0

    for lag, label in [(1,'lag1'),(12,'lag12'),(288,'lag288')]:
        m[f'real_autocorr_{label}'] = acf(r, lag)
        m[f'gen_autocorr_{label}']  = acf(g, lag)

    return m

all_metrics = {}
for name, gen_flat in scenarios_norm.items():
    print(f"\n  Scénario: {name.upper()}")
    m = compute_metrics(real_norm_flat, gen_flat, name, CONFIG['n_bins'])
    all_metrics[name] = m
    ok = '✅' if m['ks_similar'] else '⚠️'
    print(f"    Mean  réel={m['real_mean']:.4f} | généré={m['gen_mean']:.4f} | erreur={m['mean_error_pct']:.1f}%")
    print(f"    Std   réel={m['real_std']:.4f}  | généré={m['gen_std']:.4f}  | erreur={m['std_error_pct']:.1f}%")
    print(f"    KS    stat={m['ks_statistic']:.4f} p={m['ks_pvalue']:.4f} {ok}")
    print(f"    Wass  norm={m['wasserstein_normalized']:.4f}")
    print(f"    JS    sim ={m['js_similarity_pct']:.1f}%")

# ============================================================================
# VISUALISATIONS
# ============================================================================

print(f"\n{'='*70}")
print("📊 Visualisations")
print("="*70)

COLORS = {'nominal':'blue','optimistic':'green','pessimistic':'red',
          'random_mean':'purple','real':'black'}

# ── Figure 1 : Distributions (espace normalisé) ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distributions — espace normalisé (Réel vs Généré)', fontsize=13, fontweight='bold')

ax = axes[0,0]
ax.hist(real_norm_flat, bins=80, alpha=0.6, density=True, color='black', label='Réel')
for n, g in scenarios_norm.items():
    ax.hist(g, bins=80, alpha=0.45, density=True,
            color=COLORS.get(n,'gray'), label=n.capitalize())
ax.set_title('Histogrammes'); ax.set_xlabel('Valeur normalisée'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[0,1]
ax.plot(np.sort(real_norm_flat),
        np.linspace(0,1,len(real_norm_flat)), color='black', lw=2, label='Réel')
for n, g in scenarios_norm.items():
    ax.plot(np.sort(g), np.linspace(0,1,len(g)),
            color=COLORS.get(n,'gray'), lw=2, ls='--', label=n.capitalize())
ax.set_title('CDF'); ax.set_xlabel('Valeur normalisée'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1,0]
bp_data   = [real_norm_flat] + list(scenarios_norm.values())
bp_labels = ['Réel'] + [k.capitalize() for k in scenarios_norm]
bp_colors = ['black'] + [COLORS.get(k,'gray') for k in scenarios_norm]
bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True)
for patch, c in zip(bp['boxes'], bp_colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.set_title('Box Plots'); ax.set_ylabel('Valeur normalisée'); ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=20)

ax = axes[1,1]
pcts = np.arange(5,100,5)
rp   = np.percentile(real_norm_flat, pcts)
for n, g in scenarios_norm.items():
    gp = np.percentile(g, pcts)
    ax.plot(rp, gp, 'o-', color=COLORS.get(n,'gray'), lw=2, label=n.capitalize())
ax.plot([rp.min(),rp.max()],[rp.min(),rp.max()], 'k--', lw=2, label='Parfait', alpha=0.5)
ax.set_title('Percentile-Percentile'); ax.set_xlabel('Réel'); ax.set_ylabel('Généré')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '01_distributions_comparison.png', dpi=200, bbox_inches='tight')
print("✓ 01_distributions_comparison.png")
plt.close()

# ── Figure 2 : Profils temporels (espace original) ───────────────────────────
n_plots = len(scenarios_denorm) + 1
fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4*n_plots))
if n_plots == 1: axes = [axes]

dt = CONFIG['delta_t_minutes'] / 60
ts = np.arange(288) * dt

ax = axes[0]
ax.plot(ts, real_series_raw[:288], color='black', lw=2)
ax.fill_between(ts, 0, real_series_raw[:288], alpha=0.25, color='black')
ax.set_title('Données RÉELLES (24h)', fontweight='bold')
ax.set_ylabel('Arrivées'); ax.grid(alpha=0.3); ax.set_xlim(0,24)

for idx, (name, data) in enumerate(scenarios_denorm.items()):
    ax = axes[idx+1]
    sample = data[:288]
    ax.plot(ts[:len(sample)], sample, color=COLORS.get(name,'gray'), lw=2)
    ax.fill_between(ts[:len(sample)], 0, sample, alpha=0.25, color=COLORS.get(name,'gray'))
    ax.set_title(f'Scénario GÉNÉRÉ — {name.upper()}', fontweight='bold')
    ax.set_ylabel('Arrivées'); ax.grid(alpha=0.3); ax.set_xlim(0,24)

axes[-1].set_xlabel('Heure de la journée')
plt.tight_layout()
plt.savefig(output_dir / '02_temporal_profiles.png', dpi=200, bbox_inches='tight')
print("✓ 02_temporal_profiles.png")
plt.close()

# ── Figure 3 : Métriques ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Métriques de Similarité (espace normalisé)', fontsize=13, fontweight='bold')

names = list(all_metrics.keys())
bc    = [COLORS.get(n,'gray') for n in names]

for ax, key, title in [
    (axes[0], 'wasserstein_normalized', 'Wasserstein normalisé (↓)'),
    (axes[1], 'js_divergence',          'JS Divergence (↓)'),
    (axes[2], 'js_similarity_pct',      'Similarité JS % (↑)'),
]:
    vals = [all_metrics[n][key] for n in names]
    bars = ax.bar(names, vals, color=bc, alpha=0.75, edgecolor='black')
    ax.set_title(title, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15)
    if key == 'js_similarity_pct': ax.set_ylim(0,105)

plt.tight_layout()
plt.savefig(output_dir / '03_similarity_metrics.png', dpi=200, bbox_inches='tight')
print("✓ 03_similarity_metrics.png")
plt.close()

# ── Figure 4 : Autocorrélation ────────────────────────────────────────────────
max_lag = min(144, len(real_norm_flat)-1)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Autocorrélation (ACF)", fontsize=13, fontweight='bold')

def acf_series(x, max_lag):
    out = [1.0]
    for lag in range(1, max_lag+1):
        out.append(float(np.corrcoef(x[:-lag], x[lag:])[0,1]) if len(x)>lag else 0.0)
    return np.array(out)

lags_h = np.arange(max_lag+1) * CONFIG['delta_t_minutes'] / 60
racf   = acf_series(real_norm_flat[:5000], max_lag)

axes[0].plot(lags_h, racf, color='black', lw=2, label='Réel')
for n, g in scenarios_norm.items():
    ga = acf_series(g[:5000], min(max_lag, len(g)-1))
    axes[0].plot(lags_h[:len(ga)], ga, lw=2, ls='--',
                 color=COLORS.get(n,'gray'), label=n.capitalize())
axes[0].set_title('ACF'); axes[0].set_xlabel('Lag (h)'); axes[0].legend(); axes[0].grid(alpha=0.3)

for n, g in scenarios_norm.items():
    ga  = acf_series(g[:5000], min(max_lag, len(g)-1))
    ml  = min(len(racf), len(ga))
    err = np.abs(racf[:ml] - ga[:ml])
    axes[1].plot(lags_h[:ml], err, lw=2, color=COLORS.get(n,'gray'), label=n.capitalize())
axes[1].set_title('Erreur ACF'); axes[1].set_xlabel('Lag (h)'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '04_autocorrelation.png', dpi=200, bbox_inches='tight')
print("✓ 04_autocorrelation.png")
plt.close()

# ── Figure 5 : Profil journalier moyen ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))

df_temp = pd.read_csv(CONFIG['real_data_path'], index_col=0, parse_dates=True)
df_temp['v'] = df_temp[real_col]
df_temp['h'] = df_temp.index.hour + df_temp.index.minute / 60
hr = df_temp.groupby('h')['v'].agg(['mean','std'])
ax.plot(hr.index, hr['mean'], color='black', lw=3, label='Réel (moy)', zorder=5)
ax.fill_between(hr.index, hr['mean']-hr['std'], hr['mean']+hr['std'],
                alpha=0.18, color='black', label='Réel (±1σ)')

for name, data in scenarios_denorm.items():
    s = data[:288]
    ax.plot(np.arange(len(s))*dt, s, lw=2, ls='--',
            color=COLORS.get(name,'gray'), label=f'Généré — {name.capitalize()}', alpha=0.85)

ax.set_title('Profil Journalier Moyen', fontsize=14, fontweight='bold')
ax.set_xlabel('Heure'); ax.set_ylabel("Arrivées"); ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0,24)
plt.tight_layout()
plt.savefig(output_dir / '05_daily_profile_comparison.png', dpi=200, bbox_inches='tight')
print("✓ 05_daily_profile_comparison.png")
plt.close()

# ============================================================================
# TABLEAU DE SYNTHÈSE
# ============================================================================

print(f"\n{'='*70}")
print("📋 TABLEAU DE SYNTHÈSE (espace normalisé)")
print("="*70)
print(f"\n{'Scénario':<16} {'Mean err%':<11} {'Std err%':<10} {'KS':^12} {'Wass_norm':^11} {'JS Sim%':^9}")
print("-"*72)
for n, m in all_metrics.items():
    ks = f"{m['ks_statistic']:.3f} {'✅' if m['ks_similar'] else '⚠️'}"
    print(f"{n:<16} {m['mean_error_pct']:<11.1f} {m['std_error_pct']:<10.1f} "
          f"{ks:<12} {m['wasserstein_normalized']:<11.4f} {m['js_similarity_pct']:<9.1f}")
print("-"*72)

# ============================================================================
# SAUVEGARDE
# ============================================================================

with open(output_dir / 'comparison_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

rows = [{
    'scenario': n,
    'mean_error_pct': round(m['mean_error_pct'],2),
    'std_error_pct': round(m['std_error_pct'],2),
    'ks_statistic': round(m['ks_statistic'],4),
    'ks_pvalue': round(m['ks_pvalue'],4),
    'ks_similar': m['ks_similar'],
    'wasserstein_normalized': round(m['wasserstein_normalized'],4),
    'js_divergence': round(m['js_divergence'],4),
    'js_similarity_pct': round(m['js_similarity_pct'],2),
} for n, m in all_metrics.items()]

pd.DataFrame(rows).to_csv(output_dir / 'comparison_summary.csv', index=False)

print(f"\n✓ comparison_metrics.json")
print(f"✓ comparison_summary.csv")

print(f"\n{'='*70}")
print("🎉 COMPARAISON TERMINÉE !")
print("="*70)
print(f"\n📁 Résultats dans: {output_dir}")