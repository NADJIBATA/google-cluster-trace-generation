import pandas as pd
import gzip
import glob
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleCluster2011Loader:
    """
    Charge et extrait les arrivées de jobs depuis Google Cluster 2011.
    """
    
    COLUMNS = [
        'timestamp', 'missing_info', 'job_id', 'event_type',
        'user', 'scheduling_class', 'job_name', 'logical_job_name'
    ]
    
    SUBMIT_EVENT = 0  # Code pour SUBMIT
    
    def __init__(self, data_dir: str = 'data/raw/2011/job_events'):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob('part-*.csv.gz'))
        logger.info(f"Found {len(self.files)} files in {data_dir}")
    
    def load_file(self, filepath: Path, 
                  usecols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Charge un seul fichier .csv.gz
        
        Args:
            filepath: Chemin vers le fichier
            usecols: Colonnes à charger (None = toutes)
        
        Returns:
            DataFrame avec les données
        """
        if usecols is None:
            usecols = ['timestamp', 'job_id', 'event_type', 'user', 'scheduling_class']
        
        try:
            with gzip.open(filepath, 'rt') as f:
                df = pd.read_csv(
                    f,
                    header=None,
                    names=self.COLUMNS,
                    usecols=usecols,
                    dtype={
                        'timestamp': 'int64',
                        'job_id': 'int64',
                        'event_type': 'int32'
                    },
                    sep='\t'
                )
            return df
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def extract_job_arrivals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrait les événements SUBMIT (arrivées de jobs).
        
        Args:
            df: DataFrame brut
        
        Returns:
            DataFrame avec seulement les arrivées
        """
        # Filtrer SUBMIT uniquement
        arrivals = df[df['event_type'] == self.SUBMIT_EVENT].copy()
        
        # Convertir timestamp en datetime
        arrivals['datetime'] = pd.to_datetime(
            arrivals['timestamp'], 
            unit='us'
        )
        
        # Trier par timestamp
        arrivals = arrivals.sort_values('timestamp')
        
        return arrivals[['job_id', 'timestamp', 'datetime', 'user', 'scheduling_class']]
    
    def load_all_arrivals(self, 
                          max_files: Optional[int] = None,
                          save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Charge tous les fichiers et extrait toutes les arrivées.
        
        Args:
            max_files: Limiter le nombre de fichiers (None = tous)
            save_path: Sauvegarder le résultat (None = ne pas sauvegarder)
        
        Returns:
            DataFrame consolidé de toutes les arrivées
        """
        files_to_process = self.files[:max_files] if max_files else self.files
        logger.info(f"Processing {len(files_to_process)} files...")
        
        all_arrivals = []
        
        for i, filepath in enumerate(files_to_process, 1):
            logger.info(f"[{i}/{len(files_to_process)}] Processing {filepath.name}")
            
            # Charger le fichier
            df = self.load_file(filepath)
            
            if df.empty:
                logger.warning(f"  Skipping empty file")
                continue
            
            # Extraire les arrivées
            arrivals = self.extract_job_arrivals(df)
            all_arrivals.append(arrivals)
            
            logger.info(f"  Found {len(arrivals)} job arrivals")
        
        # Consolider
        logger.info("Consolidating all arrivals...")
        result = pd.concat(all_arrivals, ignore_index=True)
        
        # Dédupliquer (au cas où)
        result = result.drop_duplicates(subset=['job_id', 'timestamp'])
        result = result.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Total arrivals: {len(result):,}")
        logger.info(f"Date range: {result['datetime'].min()} to {result['datetime'].max()}")
        
        # Sauvegarder si demandé
        if save_path:
            logger.info(f"Saving to {save_path}")
            result.to_csv(save_path, index=False)
        
        return result
    
    def get_statistics(self, arrivals: pd.DataFrame) -> dict:
        """
        Calcule des statistiques descriptives sur les arrivées.
        """
        duration = arrivals['datetime'].max() - arrivals['datetime'].min()
        
        # Compter par heure
        hourly = arrivals.set_index('datetime').resample('1H').size()
        
        stats = {
            'total_jobs': len(arrivals),
            'start_date': str(arrivals['datetime'].min()),
            'end_date': str(arrivals['datetime'].max()),
            'duration_days': duration.days,
            'duration_hours': duration.total_seconds() / 3600,
            'avg_arrivals_per_hour': len(arrivals) / (duration.total_seconds() / 3600),
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


class GoogleCluster2019Loader:
    """
    Charge les données Google Cluster 2019 depuis BigQuery.
    Nécessite des credentials Google Cloud configurés.
    """
    
    DATASET = "google.com:google-cluster-data.clusterdata_2019_a"
    
    def __init__(self, project_id: str):
        from google.cloud import bigquery
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        logger.info(f"Connected to BigQuery project: {project_id}")
    
    def load_task_events(self, limit: Optional[int] = None, 
                        event_types: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Charge les task_events depuis BigQuery.
        
        Args:
            limit: Nombre maximum de lignes à charger
            event_types: Types d'événements à filtrer (None = tous)
        
        Returns:
            DataFrame avec les task events
        """
        query = f"""
        SELECT 
            time,
            job_id,
            task_index,
            machine_id,
            event_type,
            user,
            scheduling_class,
            priority,
            cpu_request,
            memory_request,
            disk_space_request,
            different_machine_restriction
        FROM `{self.DATASET}.task_events`
        """
        
        if event_types:
            event_list = ",".join(map(str, event_types))
            query += f" WHERE event_type IN ({event_list})"
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info("Executing BigQuery query for task_events...")
        df = self.client.query(query).to_dataframe()
        
        # Convertir timestamp
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        
        logger.info(f"Loaded {len(df):,} task events")
        return df
    
    def load_job_events(self, limit: Optional[int] = None,
                       event_types: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Charge les job_events depuis BigQuery.
        """
        query = f"""
        SELECT 
            time,
            job_id,
            event_type,
            user,
            scheduling_class,
            job_name,
            logical_job_name
        FROM `{self.DATASET}.job_events`
        """
        
        if event_types:
            event_list = ",".join(map(str, event_types))
            query += f" WHERE event_type IN ({event_list})"
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info("Executing BigQuery query for job_events...")
        df = self.client.query(query).to_dataframe()
        
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        
        logger.info(f"Loaded {len(df):,} job events")
        return df
    
    def extract_task_arrivals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrait les arrivées de tâches (event_type = 0).
        """
        arrivals = df[df['event_type'] == 0].copy()
        arrivals = arrivals[['time', 'job_id', 'task_index', 'datetime', 'user', 'scheduling_class']]
        arrivals = arrivals.sort_values('time').reset_index(drop=True)
        return arrivals
    
    def extract_job_arrivals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrait les arrivées de jobs (event_type = 0).
        """
        arrivals = df[df['event_type'] == 0].copy()
        arrivals = arrivals[['time', 'job_id', 'datetime', 'user', 'scheduling_class']]
        arrivals = arrivals.sort_values('time').reset_index(drop=True)
        return arrivals


# Script d'utilisation
if __name__ == '__main__':
    import json
    
    # Créer le loader
    loader = GoogleCluster2011Loader('data/raw/2011/job_events')
    
    # Charger toutes les arrivées (limiter à 50 fichiers pour commencer)
    arrivals = loader.load_all_arrivals(
        max_files=50,
        save_path='data/processed/2011_arrivals_raw.csv'
    )
    
    # Calculer les statistiques
    stats = loader.get_statistics(arrivals)
    
    # Afficher
    print("\n" + "="*60)
    print("STATISTIQUES DES ARRIVÉES - GOOGLE CLUSTER 2011")
    print("="*60)
    print(f"Nombre total de jobs: {stats['total_jobs']:,}")
    print(f"Période: {stats['start_date']} → {stats['end_date']}")
    print(f"Durée: {stats['duration_days']} jours ({stats['duration_hours']:.1f} heures)")
    print(f"Taux moyen: {stats['avg_arrivals_per_hour']:.2f} jobs/heure")
    print("\nDistribution horaire:")
    print(f"  Moyenne: {stats['hourly_stats']['mean']:.2f}")
    print(f"  Médiane: {stats['hourly_stats']['median']:.2f}")
    print(f"  Écart-type: {stats['hourly_stats']['std']:.2f}")
    print(f"  Min: {stats['hourly_stats']['min']}")
    print(f"  Max: {stats['hourly_stats']['max']}")
    print("="*60)
    
    # Sauvegarder les stats
    with open('data/processed/2011_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n✓ Fichiers sauvegardés:")
    print("  - data/processed/2011_arrivals_raw.csv")
    print("  - data/processed/2011_statistics.json")