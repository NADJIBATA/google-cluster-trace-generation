# src/data/loader_2019_bigquery.py

from google.cloud import bigquery
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleCluster2019BigQueryLoader:
    """
    Loader optimisé pour Google Cluster 2019 via BigQuery.
    """
    
    # Table publique Google Cluster 2019
    TABLE = "google.com:google-cluster-data.clusterdata_2019_a.job_events"
    
    def __init__(self, project_id: str = "inspired-studio-482317-j4"):
        """Initialise le client BigQuery."""
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        logger.info(f"✅ Client BigQuery initialisé (projet: {project_id})")
    
    def estimate_cost(self, query: str) -> dict:
        """Estime le coût d'une requête avant exécution."""
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = self.client.query(query, job_config=job_config)
        
        bytes_scanned = query_job.total_bytes_processed
        gb_scanned = bytes_scanned / (1024**3)
        
        return {
            'bytes': bytes_scanned,
            'gb': gb_scanned,
            'cost_usd': max(0, gb_scanned / 1024 - 1) * 5,  # $5/TB après 1TB gratuit
            'is_free': gb_scanned < 1024  # < 1 TB
        }
    
    def load_arrivals(self, 
                     limit: int = 100000,
                     save_path: str = None) -> pd.DataFrame:
        """
        Charge les arrivées de collections (jobs) depuis BigQuery.
        
        Args:
            limit: Nombre maximum de collections à charger
            save_path: Chemin pour sauvegarder les données
        
        Returns:
            DataFrame avec job_id, timestamp, datetime
        """
        # Requête optimisée
        query = f"""
        SELECT 
            job_id,
            MIN(time) as start_time
        FROM 
            `{self.TABLE}`
        WHERE
            time IS NOT NULL
            AND time > 0
            AND event_type = 0
        GROUP BY
            job_id
        ORDER BY
            start_time
        LIMIT {limit}
        """
        
        # Estimer le coût
        logger.info("Estimation du coût...")
        estimate = self.estimate_cost(query)
        logger.info(f"📊 Données à scanner: {estimate['gb']:.2f} GB")
        logger.info(f"💰 Coût: ${estimate['cost_usd']:.4f} {'(GRATUIT)' if estimate['is_free'] else ''}")
        
        # Exécuter
        logger.info(f"Chargement de {limit:,} collections...")
        df = self.client.query(query).to_dataframe()
        
        # Traitement
        df['datetime'] = pd.to_datetime(df['start_time'], unit='us')
        df = df.rename(columns={'start_time': 'timestamp'})
        df = df[['job_id', 'timestamp', 'datetime']].sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✅ {len(df):,} arrivées chargées")
        logger.info(f"   Période: {df['datetime'].min()} → {df['datetime'].max()}")
        
        # Sauvegarder
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"💾 Sauvegardé: {save_path}")
        
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Calcule les statistiques."""
        duration = df['datetime'].max() - df['datetime'].min()
        hourly = df.set_index('datetime').resample('1H').size()
        
        stats = {
            'total_jobs': int(len(df)),
            'unique_jobs': int(df['job_id'].nunique()),
            'start_date': str(df['datetime'].min()),
            'end_date': str(df['datetime'].max()),
            'duration_days': int(duration.days),
            'duration_hours': float(duration.total_seconds() / 3600),
            'avg_arrivals_per_hour': float(len(df) / (duration.total_seconds() / 3600)),
            'hourly_stats': {
                'mean': float(hourly.mean()),
                'median': float(hourly.median()),
                'std': float(hourly.std()),
                'min': int(hourly.min()),
                'max': int(hourly.max()),
                'q25': float(hourly.quantile(0.25)),
                'q75': float(hourly.quantile(0.75))
            }
        }
        
        return stats


if __name__ == '__main__':
    import os
    
    os.makedirs('data/processed', exist_ok=True)
    
    print("="*70)
    print("CHARGEMENT GOOGLE CLUSTER 2019")
    print("="*70)
    
    # Créer le loader
    loader = GoogleCluster2019BigQueryLoader()
    
    # Charger 100k collections (gratuit, rapide)
    arrivals = loader.load_arrivals(
        limit=100000,
        save_path='data/processed/2019_arrivals_raw.csv'
    )
    
    # Statistiques
    stats = loader.get_statistics(arrivals)
    
    print("\n" + "="*70)
    print("STATISTIQUES")
    print("="*70)
    print(f"Jobs:        {stats['total_jobs']:>15,}")
    print(f"Durée:       {stats['duration_days']:>15,} jours")
    print(f"Taux moyen:  {stats['avg_arrivals_per_hour']:>15.2f} jobs/h")
    print("="*70)
    
    # Sauvegarder stats
    with open('data/processed/2019_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n✅ Données prêtes pour le prétraitement")