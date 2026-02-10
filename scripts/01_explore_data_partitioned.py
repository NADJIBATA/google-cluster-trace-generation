#!/usr/bin/env python3
"""
Script d'exploration des données Google Cluster 2011 - VERSION PARTITIONNÉE
Adapté pour gérer les fichiers part-*-of-00500.csv.gz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import json
import glob
from tqdm import tqdm

# Configuration - MODIFIEZ LE CHEMIN ICI
DATA_PATH = Path("data/processed")  # ⚠️ CHANGEZ CECI selon où sont vos données
OUTPUT_PATH = Path("/data/processed")
FIGURE_PATH = Path("/mnt/user-data/outputs/results/figures")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
FIGURE_PATH.mkdir(parents=True, exist_ok=True)

def find_partition_files(data_path):
    """Trouve tous les fichiers part-*-of-*.csv.gz"""
    print(f"🔍 Recherche des fichiers partitionnés dans {data_path}")
    
    patterns = [
        "part-*-of-*.csv.gz",
        "part-*.csv.gz",
        "*.csv.gz"
    ]
    
    files = []
    for pattern in patterns:
        found = list(data_path.glob(pattern))
        files.extend(found)
    
    # Trier par nom pour traiter dans l'ordre
    files = sorted(set(files))
    
    print(f"✅ Trouvé {len(files)} fichiers")
    
    if files:
        total_size = sum(f.stat().st_size for f in files) / (1024**3)  # GB
        print(f"📊 Taille totale : {total_size:.2f} GB")
    
    return files

def load_single_partition(file_path, nrows=None):
    """Charge un fichier partition."""
    try:
        df = pd.read_csv(file_path, compression='gzip', nrows=nrows)
        return df
    except Exception as e:
        print(f"   ⚠️ Erreur sur {file_path.name}: {e}")
        return None

def identify_columns(df):
    """Identifie automatiquement les colonnes importantes."""
    
    print("\n🔍 Identification des colonnes...")
    print(f"   Colonnes disponibles : {df.columns.tolist()}")
    
    # Chercher la colonne event_type
    event_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'event' in col_lower and 'type' in col_lower:
            event_col = col
            break
    
    if event_col is None:
        # Essayer des noms standards Google Cluster
        possible_names = ['type', 'event_type', 'event', 'event_name']
        for name in possible_names:
            if name in df.columns:
                event_col = name
                break
    
    # Chercher la colonne timestamp
    timestamp_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'time' in col_lower:
            timestamp_col = col
            break
    
    # Chercher la colonne job_id
    job_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'job' in col_lower and 'id' in col_lower:
            job_col = col
            break
    
    print(f"\n✅ Colonnes identifiées :")
    print(f"   Event Type : {event_col}")
    print(f"   Timestamp  : {timestamp_col}")
    print(f"   Job ID     : {job_col}")
    
    return event_col, timestamp_col, job_col

def load_and_filter_partitions(files, max_files=None, sample_size=None):
    """
    Charge les partitions et filtre les événements SUBMIT.
    
    Args:
        files: liste des fichiers à charger
        max_files: nombre maximum de fichiers à traiter (None = tous)
        sample_size: nombre de lignes par fichier (None = tout)
    """
    
    print(f"\n📂 Chargement des partitions...")
    
    if max_files is not None:
        files = files[:max_files]
        print(f"   Traitement de {len(files)} fichiers (max_files={max_files})")
    else:
        print(f"   Traitement de TOUS les {len(files)} fichiers")
    
    if sample_size is not None:
        print(f"   ⚠️  Mode échantillonnage : {sample_size} lignes par fichier")
    
    all_submit_events = []
    
    # Charger le premier fichier pour identifier les colonnes
    print(f"\n🔍 Analyse du premier fichier...")
    first_df = load_single_partition(files[0], nrows=1000)
    if first_df is None:
        print("❌ Impossible de charger le premier fichier")
        return None, None, None, None
    
    event_col, timestamp_col, job_col = identify_columns(first_df)
    
    if event_col is None or timestamp_col is None:
        print("❌ Colonnes essentielles non trouvées !")
        print(f"   Colonnes disponibles : {first_df.columns.tolist()}")
        return None, None, None, None
    
    # Identifier le code pour SUBMIT
    print(f"\n🔍 Détection du code SUBMIT...")
    event_types = first_df[event_col].unique()
    print(f"   Types d'événements trouvés : {event_types}")
    
    # SUBMIT peut être :
    # - 0 dans Google Cluster 2011
    # - "SUBMIT" en texte
    # - autre selon la version
    submit_code = None
    if 0 in event_types:
        submit_code = 0
        print(f"   ✅ Code SUBMIT identifié : 0")
    elif 'SUBMIT' in [str(e).upper() for e in event_types]:
        submit_code = 'SUBMIT'
        print(f"   ✅ Code SUBMIT identifié : 'SUBMIT'")
    else:
        print(f"   ⚠️  Code SUBMIT non identifié automatiquement")
        print(f"   Types trouvés : {event_types}")
        submit_code = event_types[0]  # Par défaut, prendre le premier
        print(f"   ⚠️  Utilisation de {submit_code} par défaut")
    
    # Maintenant charger tous les fichiers
    print(f"\n📊 Chargement et filtrage de {len(files)} fichiers...")
    
    for file in tqdm(files, desc="Traitement"):
        df = load_single_partition(file, nrows=sample_size)
        
        if df is None:
            continue
        
        # Filtrer les SUBMIT
        if submit_code == 0:
            submit_mask = df[event_col] == 0
        elif submit_code == 'SUBMIT':
            submit_mask = df[event_col].astype(str).str.upper() == 'SUBMIT'
        else:
            submit_mask = df[event_col] == submit_code
        
        df_submit = df[submit_mask].copy()
        
        if len(df_submit) > 0:
            # Extraire timestamp et job_id
            df_submit['timestamp_us'] = pd.to_numeric(df_submit[timestamp_col], errors='coerce')
            
            if job_col is not None:
                df_submit['job_id'] = df_submit[job_col]
            
            all_submit_events.append(df_submit[['timestamp_us', 'job_id'] if job_col else ['timestamp_us']])
    
    if not all_submit_events:
        print("❌ Aucun événement SUBMIT trouvé !")
        return None, None, None, None
    
    # Concaténer tous les événements
    print(f"\n🔗 Fusion des données...")
    df_all_submit = pd.concat(all_submit_events, ignore_index=True)
    
    # Supprimer les doublons potentiels
    df_all_submit = df_all_submit.drop_duplicates()
    
    # Trier par timestamp
    df_all_submit = df_all_submit.sort_values('timestamp_us').reset_index(drop=True)
    
    print(f"✅ {len(df_all_submit):,} événements SUBMIT extraits")
    
    return df_all_submit, event_col, timestamp_col, job_col

def create_time_series(df_submit, delta_t_minutes=10):
    """Crée une série temporelle du nombre d'arrivées par intervalle."""
    
    print(f"\n⏱️  Création de la série temporelle (Δt = {delta_t_minutes} minutes)...")
    
    # Convertir en datetime
    df_submit['datetime'] = pd.to_datetime(df_submit['timestamp_us'], unit='us')
    
    # Info sur la période couverte
    start_time = df_submit['datetime'].min()
    end_time = df_submit['datetime'].max()
    duration = end_time - start_time
    
    print(f"   📅 Période : {start_time} → {end_time}")
    print(f"   ⏳ Durée : {duration.days} jours, {duration.seconds//3600} heures")
    
    # Créer des bins temporels
    df_submit['time_bin'] = df_submit['datetime'].dt.floor(f'{delta_t_minutes}min')
    
    # Compter les arrivées par bin
    time_series = df_submit.groupby('time_bin').size().reset_index(name='num_arrivals')
    
    print(f"✅ Série temporelle créée : {len(time_series):,} intervalles")
    
    return time_series

def analyze_daily_patterns(time_series):
    """Analyse les patterns journaliers."""
    
    time_series['hour'] = time_series['time_bin'].dt.hour
    time_series['day'] = time_series['time_bin'].dt.date
    time_series['day_of_week'] = time_series['time_bin'].dt.dayofweek
    
    # Statistiques globales
    stats = {
        'mean': float(time_series['num_arrivals'].mean()),
        'std': float(time_series['num_arrivals'].std()),
        'min': int(time_series['num_arrivals'].min()),
        'max': int(time_series['num_arrivals'].max()),
        'median': float(time_series['num_arrivals'].median()),
        'total_jobs': int(time_series['num_arrivals'].sum()),
        'num_intervals': int(len(time_series)),
        'start_time': str(time_series['time_bin'].min()),
        'end_time': str(time_series['time_bin'].max()),
    }
    
    print("\n" + "=" * 70)
    print("📊 STATISTIQUES GLOBALES")
    print("=" * 70)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key:20s} : {value:,.2f}")
        else:
            print(f"   {key:20s} : {value:,}")
    
    # Patterns horaires
    hourly_pattern = time_series.groupby('hour')['num_arrivals'].agg(['mean', 'std', 'min', 'max'])
    
    print("\n📈 PATTERN HORAIRE MOYEN")
    print(hourly_pattern)
    
    return stats, hourly_pattern

def create_visualizations(time_series, hourly_pattern):
    """Crée des visualisations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Série temporelle complète (sous-échantillonnée si trop long)
    ax = axes[0, 0]
    if len(time_series) > 10000:
        # Sous-échantillonner pour la visualisation
        step = len(time_series) // 10000
        ts_plot = time_series.iloc[::step]
        title = f'Série Temporelle - Arrivées de Jobs (1/{step} points)'
    else:
        ts_plot = time_series
        title = 'Série Temporelle Complète - Arrivées de Jobs'
    
    ax.plot(ts_plot['time_bin'], ts_plot['num_arrivals'], linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Temps')
    ax.set_ylabel('Nombre d\'arrivées')
    ax.grid(True, alpha=0.3)
    
    # 2. Distribution des arrivées
    ax = axes[0, 1]
    ax.hist(time_series['num_arrivals'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_title('Distribution du Nombre d\'Arrivées', fontsize=12, fontweight='bold')
    ax.set_xlabel('Nombre d\'arrivées par intervalle')
    ax.set_ylabel('Fréquence')
    ax.grid(True, alpha=0.3)
    
    # 3. Pattern horaire
    ax = axes[1, 0]
    ax.plot(hourly_pattern.index, hourly_pattern['mean'], marker='o', linewidth=2)
    ax.fill_between(hourly_pattern.index, 
                     hourly_pattern['mean'] - hourly_pattern['std'],
                     hourly_pattern['mean'] + hourly_pattern['std'],
                     alpha=0.3)
    ax.set_title('Pattern Horaire Moyen (± 1 σ)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Heure de la journée')
    ax.set_ylabel('Nombre moyen d\'arrivées')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    
    # 4. Boxplot par jour de la semaine
    ax = axes[1, 1]
    day_names = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    time_series['day_name'] = time_series['day_of_week'].map(lambda x: day_names[x])
    time_series.boxplot(column='num_arrivals', by='day_name', ax=ax)
    ax.set_title('Distribution par Jour de la Semaine', fontsize=12, fontweight='bold')
    ax.set_xlabel('Jour')
    ax.set_ylabel('Nombre d\'arrivées')
    plt.suptitle('')  # Enlever le titre automatique
    
    plt.tight_layout()
    
    output_file = FIGURE_PATH / '01_exploration_initiale.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 Figure sauvegardée : {output_file}")
    
    plt.close()

def main():
    """Fonction principale."""
    
    print("=" * 70)
    print("📊 EXPLORATION DES DONNÉES GOOGLE CLUSTER 2011 (PARTITIONNÉES)")
    print("=" * 70)
    
    # Configuration
    MAX_FILES = None  # Mettre un nombre pour tester (ex: 10), None pour tout
    SAMPLE_SIZE = None  # Mettre un nombre pour tester (ex: 100000), None pour tout
    DELTA_T = 10  # minutes
    
    if MAX_FILES is not None:
        print(f"\n⚠️  MODE TEST : Traitement de {MAX_FILES} fichiers seulement")
    if SAMPLE_SIZE is not None:
        print(f"⚠️  MODE ÉCHANTILLONNAGE : {SAMPLE_SIZE} lignes par fichier")
    
    # 1. Trouver les fichiers
    print(f"\n📁 Recherche dans : {DATA_PATH}")
    files = find_partition_files(DATA_PATH)
    
    if not files:
        print("\n❌ Aucun fichier trouvé !")
        print("\n💡 Solutions :")
        print("   1. Vérifiez que DATA_PATH est correct (ligne 18 du script)")
        print("   2. Vérifiez que les fichiers sont bien au format part-*-of-*.csv.gz")
        print("   3. Listez le contenu : ls /data/raw/")
        return
    
    # 2. Charger et filtrer
    df_submit, event_col, timestamp_col, job_col = load_and_filter_partitions(
        files, 
        max_files=MAX_FILES,
        sample_size=SAMPLE_SIZE
    )
    
    if df_submit is None:
        print("❌ Échec du chargement")
        return
    
    # 3. Créer la série temporelle
    time_series = create_time_series(df_submit, delta_t_minutes=DELTA_T)
    
    # 4. Analyser les patterns
    stats, hourly_pattern = analyze_daily_patterns(time_series)
    
    # 5. Visualiser
    print("\n📊 Création des visualisations...")
    create_visualizations(time_series, hourly_pattern)
    
    # 6. Sauvegarder
    output_file = OUTPUT_PATH / f'time_series_dt{DELTA_T}min.csv'
    time_series.to_csv(output_file, index=False)
    print(f"💾 Série temporelle sauvegardée : {output_file}")
    
    stats_file = OUTPUT_PATH / 'stats_exploration.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"💾 Statistiques sauvegardées : {stats_file}")
    
    print("\n" + "=" * 70)
    print("✅ EXPLORATION TERMINÉE")
    print("=" * 70)
    print("\n📋 Prochaines étapes :")
    print("   1. Examiner les résultats dans results/figures/")
    print("   2. Si satisfait, relancer avec MAX_FILES=None pour tout traiter")
    print("   3. Ajuster Δt si nécessaire")
    print("   4. Lancer : python scripts/02_preprocess_vae.py")
    print("=" * 70)

if __name__ == "__main__":
    main()