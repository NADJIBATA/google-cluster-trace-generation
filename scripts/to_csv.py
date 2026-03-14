"""
Conversion des Scénarios Générés en CSV
========================================

Convertit les scénarios .npy en fichiers CSV formatés comme les données originales,
avec timestamps, dates, et toutes les features.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import sys

# on Windows, for emojis and accents, s'assurer de l'UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

print("="*70)
print("📄 CONVERSION DES SCÉNARIOS EN CSV")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'scenarios_dir': 'data/generated',                        # où chercher les .npy
    'output_dir': 'data/generated/generative',                # corrige : "generated" pas "generates"
    'scaler_path': 'data/processed/sequences/scaler.pkl',
    
    # Configuration temporelle
    'start_date': '2011-05-01 00:00:00',  # Date de début
    'timestep_minutes': 5,                 # Intervalle entre points (5min)
    
    # Colonnes à inclure
    'include_features': True,  # Ajouter features (hour, day_of_week, etc.)
}

print(f"\n⚙️  Configuration:")
print(f"   Start date: {CONFIG['start_date']}")
print(f"   Timestep: {CONFIG['timestep_minutes']} minutes")
print(f"   Features: {CONFIG['include_features']}")

# ============================================================================
# CHARGEMENT
# ============================================================================

print(f"\n{'='*70}")
print("📂 Chargement")
print("="*70)

# Charger scaler
scaler = pickle.load(open(CONFIG['scaler_path'], 'rb'))
print(f"✓ Scaler chargé")

# Charger scenarios (chercher dans les sous‑dossiers au besoin)
scenarios_dir = Path(CONFIG['scenarios_dir'])
output_dir = Path(CONFIG['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

scenarios = {}
for file in sorted(scenarios_dir.rglob("*.npy")):
    name = file.stem
    data = np.load(file)
    scenarios[name] = data
    rel = file.relative_to(scenarios_dir)
    print(f"✓ {rel} → {name}: {data.shape}")

if not scenarios:
    print(f"\n❌ Aucun scénario trouvé dans {scenarios_dir}")
    print(f"   Vérifiez que les scénarios ont été générés")
    exit(1)

# ============================================================================
# FONCTIONS DE CONVERSION
# ============================================================================

def add_temporal_features(df, start_date):
    """Ajoute les features temporelles comme dans les données originales."""
    
    # Créer index datetime
    start = pd.to_datetime(start_date)
    timestamps = [start + timedelta(minutes=CONFIG['timestep_minutes']*i) 
                  for i in range(len(df))]
    df['datetime'] = timestamps
    
    # Features temporelles
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    
    return df

def add_rolling_features(df, value_col='arrival_rate', windows=[12, 24]):
    """Ajoute les moyennes glissantes."""
    
    for w in windows:
        df[f'{value_col}_mean_{w}'] = df[value_col].rolling(
            window=w, min_periods=1
        ).mean()
        
        df[f'{value_col}_std_{w}'] = df[value_col].rolling(
            window=w, min_periods=1
        ).std()
        
        df[f'{value_col}_max_{w}'] = df[value_col].rolling(
            window=w, min_periods=1
        ).max()
        
        df[f'{value_col}_min_{w}'] = df[value_col].rolling(
            window=w, min_periods=1
        ).min()
    
    return df

def scenario_to_dataframe(data, scaler, scenario_name, start_date):
    """
    Convertit un scénario en DataFrame formaté.
    
    Args:
        data: Array normalisé (N, seq_len, 1) ou (seq_len, 1)
        scaler: Scaler pour dénormalisation
        scenario_name: Nom du scénario
        start_date: Date de début
    
    Returns:
        DataFrame avec toutes les colonnes
    """
    
    # Gérer les dimensions
    if data.ndim == 2:
        # (seq_len, 1) → (1, seq_len, 1)
        data = data.reshape(1, -1, 1)
    
    # Si multiple séquences, prendre la première ou la moyenne
    if data.shape[0] > 1:
        print(f"  ℹ️  {scenario_name}: {data.shape[0]} séquences → utilise la moyenne")
        # Moyenne sur toutes les séquences
        data = data.mean(axis=0, keepdims=True)
    
    # Flatten et dénormaliser
    data_flat = data.reshape(-1, 1)
    data_denorm = scaler.inverse_transform(data_flat)
    
    # Créer DataFrame
    df = pd.DataFrame({
        'arrival_rate': data_denorm.flatten(),
        'job_count': data_denorm.flatten()  # Alias
    })
    
    # Arrondir à des valeurs entières pour job_count
    df['job_count'] = df['job_count'].round().astype(int)
    
    # Clipper valeurs négatives
    df['arrival_rate'] = df['arrival_rate'].clip(lower=0)
    df['job_count'] = df['job_count'].clip(lower=0)
    
    # Ajouter features temporelles
    df = add_temporal_features(df, start_date)
    
    # Ajouter rolling features si demandé
    if CONFIG['include_features']:
        df = add_rolling_features(df, 'arrival_rate', windows=[12, 24])
    
    # Réorganiser colonnes
    cols_order = ['datetime', 'arrival_rate', 'job_count', 'hour', 'day_of_week']
    if CONFIG['include_features']:
        cols_order.extend([
            'is_weekend', 'is_business_hours',
            'arrival_rate_mean_12', 'arrival_rate_std_12',
            'arrival_rate_max_12', 'arrival_rate_min_12',
            'arrival_rate_mean_24', 'arrival_rate_std_24',
            'arrival_rate_max_24', 'arrival_rate_min_24'
        ])
    
    # Garder seulement les colonnes qui existent
    cols_order = [c for c in cols_order if c in df.columns]
    df = df[cols_order]
    
    return df

# ============================================================================
# CONVERSION DES SCÉNARIOS
# ============================================================================

print(f"\n{'='*70}")
print("🔄 Conversion en CSV")
print("="*70)

for scenario_name, data in scenarios.items():
    print(f"\n📝 Traitement: {scenario_name}")
    print(f"   Shape: {data.shape}")
    
    # Convertir
    df = scenario_to_dataframe(
        data, 
        scaler, 
        scenario_name,
        CONFIG['start_date']
    )
    
    print(f"   → DataFrame: {df.shape}")
    print(f"   → Période: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   → Durée: {len(df) * CONFIG['timestep_minutes'] / 60:.1f} heures")
    
    # Sauvegarder
    output_file = output_dir / f"{scenario_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"   ✓ Sauvegardé: {output_file}")
    
    # Afficher aperçu
    print(f"\n   Aperçu:")
    print(df.head(3).to_string(index=False))

# ============================================================================
# CRÉER UN FICHIER COMBINÉ (OPTIONNEL)
# ============================================================================

print(f"\n{'='*70}")
print("📦 Création du fichier combiné (tous scénarios)")
print("="*70)

# Charger tous les CSV
dfs_combined = []

for scenario_name in scenarios.keys():
    csv_file = output_dir / f"{scenario_name}.csv"
    df = pd.read_csv(csv_file, parse_dates=['datetime'])
    df['scenario'] = scenario_name  # Ajouter colonne scenario
    dfs_combined.append(df)

# Combiner
df_all = pd.concat(dfs_combined, ignore_index=True)

# Sauvegarder
combined_file = output_dir / "all_scenarios_combined.csv"
df_all.to_csv(combined_file, index=False)

print(f"✓ Fichier combiné créé: {combined_file}")
print(f"  Shape: {df_all.shape}")
print(f"  Scénarios: {df_all['scenario'].unique()}")

# ============================================================================
# STATISTIQUES
# ============================================================================

print(f"\n{'='*70}")
print("📊 Statistiques des scénarios")
print("="*70)

print(f"\n{'Scénario':<20} {'Points':<8} {'Mean':<12} {'Std':<12} {'Min':<8} {'Max':<8}")
print("-"*80)

for scenario_name in sorted(scenarios.keys()):
    csv_file = output_dir / f"{scenario_name}.csv"
    df = pd.read_csv(csv_file)
    
    stats = df['arrival_rate'].describe()
    print(f"{scenario_name:<20} {len(df):<8} "
          f"{stats['mean']:<12.2f} {stats['std']:<12.2f} "
          f"{stats['min']:<8.2f} {stats['max']:<8.2f}")

# ============================================================================
# DOCUMENTATION
# ============================================================================

print(f"\n{'='*70}")
print("📝 Création de la documentation")
print("="*70)

readme_content = f"""# Scénarios Générés - Format CSV

## 📁 Fichiers

{chr(10).join([f"- `{name}.csv` : Scénario {name}" for name in scenarios.keys()])}
- `all_scenarios_combined.csv` : Tous les scénarios combinés

## 📊 Format des Fichiers

### Colonnes Principales
- `datetime` : Timestamp (format: YYYY-MM-DD HH:MM:SS)
- `arrival_rate` : Taux d'arrivée des jobs (float)
- `job_count` : Nombre de jobs (int)

### Features Temporelles
- `hour` : Heure de la journée (0-23)
- `day_of_week` : Jour de la semaine (0=Lundi, 6=Dimanche)
- `is_weekend` : 1 si weekend, 0 sinon
- `is_business_hours` : 1 si 9h-17h, 0 sinon

### Features Calculées (Fenêtres Glissantes)
- `arrival_rate_mean_12` : Moyenne sur 12 intervalles (1h)
- `arrival_rate_std_12` : Écart-type sur 12 intervalles
- `arrival_rate_max_12` : Maximum sur 12 intervalles
- `arrival_rate_min_12` : Minimum sur 12 intervalles
- `arrival_rate_mean_24` : Moyenne sur 24 intervalles (2h)
- `arrival_rate_std_24` : Écart-type sur 24 intervalles
- `arrival_rate_max_24` : Maximum sur 24 intervalles
- `arrival_rate_min_24` : Minimum sur 24 intervalles

## 🕐 Configuration Temporelle
- Date de début : {CONFIG['start_date']}
- Intervalle : {CONFIG['timestep_minutes']} minutes
- Durée typique : {list(scenarios.values())[0].shape[1] * CONFIG['timestep_minutes'] / 60:.1f} heures

## 📈 Utilisation

### Charger un scénario
```python
import pandas as pd

# Charger
df = pd.read_csv('nominal.csv', parse_dates=['datetime'])

# Afficher
print(df.head())

# Statistiques
print(df['arrival_rate'].describe())
```

### Visualiser
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['arrival_rate'])
plt.xlabel('Time')
plt.ylabel('Arrival Rate')
plt.title('Scenario: Nominal')
plt.grid(True)
plt.show()
```

### Comparer plusieurs scénarios
```python
# Charger le fichier combiné
df_all = pd.read_csv('all_scenarios_combined.csv', parse_dates=['datetime'])

# Grouper par scénario
for scenario in df_all['scenario'].unique():
    df_scenario = df_all[df_all['scenario'] == scenario]
    plt.plot(df_scenario['datetime'], df_scenario['arrival_rate'], 
             label=scenario)

plt.legend()
plt.show()
```

## 📊 Statistiques des Scénarios

{chr(10).join([f"### {name}" + chr(10) + 
               f"- Points: {len(scenarios[name].flatten())}" + chr(10) +
               f"- Moyenne: {scenarios[name].mean():.2f}" + chr(10) +
               f"- Écart-type: {scenarios[name].std():.2f}"
               for name in sorted(scenarios.keys())])}

## 🔄 Génération

Ces fichiers ont été générés à partir des modèles VAE entraînés sur les données
Google Cluster 2011. Les scénarios représentent différentes conditions de charge :

- **nominal** : Charge typique/moyenne
- **optimistic** : Charge basse
- **pessimistic** : Charge haute
- **stressed** : Charge variable avec pics
- **random** : Échantillon aléatoire

---

Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

readme_file = output_dir / "README.md"
with open(readme_file, 'w') as f:
    f.write(readme_content)

print(f"✓ Documentation créée: {readme_file}")

# ============================================================================
# RÉSUMÉ
# ============================================================================

print(f"\n{'='*70}")
print("🎉 CONVERSION TERMINÉE !")
print("="*70)

print(f"\n📁 Fichiers créés dans: {output_dir}/")
print(f"\n✓ {len(scenarios)} scénarios convertis")
print(f"✓ 1 fichier combiné")
print(f"✓ 1 documentation (README.md)")

print(f"\n📊 Utilisation:")
print(f"  import pandas as pd")
print(f"  df = pd.read_csv('{output_dir}/nominal.csv')")
print(f"  print(df.head())")

print(f"\n{'='*70}")