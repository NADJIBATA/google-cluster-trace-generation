# src/data/preprocessor.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict

class TracePreprocessor:
  
    
    def __init__(self, delta_t: str = '1H', sequence_length: int = 24):
        """
        Args:
            delta_t: Pas de temps d'agrégation ('1H', '30T', etc.)
            sequence_length: Longueur des séquences (24 = 24h)
        """
        self.delta_t = delta_t
        self.sequence_length = sequence_length
        self.scaler = None
        
    def load_and_aggregate(self, filepath: str) -> pd.DataFrame:
        """
        Charge les données et agrège par pas de temps.
        
        Principe :
        1. Charger job_events
        2. Filtrer event_type == 'SUBMIT'
        3. Convertir timestamp en datetime
        4. Compter arrivées par intervalle delta_t
        """
        # Charger (possiblement en chunks si gros fichier)
        df = pd.read_csv(filepath, 
                        sep=',', header=None, names=['time', 'missing_info', 'job_id', 'event_type', 'user', 'scheduling_class', 'job_name', 'logical_job_name'],
                        usecols=['time', 'job_id', 'event_type'])
        
        # Filtrer SUBMIT
        df = df[df['event_type'] == 0].copy()
        
        # Conversion temporelle
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        
        # Agrégation par delta_t
        df = df.set_index('datetime')
        arrivals = df.resample(self.delta_t).size()
        arrivals = arrivals.to_frame(name='num_arrivals')
        
        return arrivals
    
    def extract_features(self, arrivals: pd.DataFrame) -> pd.DataFrame:
        """
        Extrait des features supplémentaires.
        
        Features créées :
        - hour : heure de la journée (0-23)
        - day_of_week : jour de la semaine (0-6)
        - hour_sin, hour_cos : encodage cyclique de l'heure
        - rolling_mean_3h : moyenne glissante 3h
        - rolling_std_3h : écart-type glissant 3h
        - diff_1h : différence première (taux de changement)
        """
        df = arrivals.copy()
        
        # Features temporelles
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # Encodage cyclique (évite discontinuité 23h -> 0h)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Features statistiques glissantes
        df['rolling_mean_3h'] = df['num_arrivals'].rolling(3, min_periods=1).mean()
        df['rolling_std_3h'] = df['num_arrivals'].rolling(3, min_periods=1).std()
        df['rolling_max_6h'] = df['num_arrivals'].rolling(6, min_periods=1).max()
        
        # Taux de changement
        df['diff_1h'] = df['num_arrivals'].diff()
        df['diff_pct'] = df['num_arrivals'].pct_change()
        
        # Remplir NaN (première ligne après diff)
        df = df.fillna(method='bfill').fillna(0)
        
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des séquences de longueur fixe pour l'entraînement.
        
        Méthode de fenêtre glissante :
        - Fenêtre de taille sequence_length
        - Glisse d'un pas de temps à la fois
        - Exemple : si sequence_length=24h et delta_t=1h
          -> séquence [t, t+1, ..., t+23] prédit ou reconstruit elle-même
        
        Returns:
            sequences: (num_sequences, sequence_length, num_features)
            timestamps: (num_sequences,) début de chaque séquence
        """
        features = df.values  # (num_timesteps, num_features)
        num_samples = len(features) - self.sequence_length + 1
        
        sequences = []
        timestamps = []
        
        for i in range(num_samples):
            seq = features[i:i + self.sequence_length]
            sequences.append(seq)
            timestamps.append(df.index[i])
        
        return np.array(sequences), np.array(timestamps)
    
    def normalize(self, sequences: np.ndarray, 
                  fit: bool = True) -> np.ndarray:
        """
        Normalise les séquences (StandardScaler par feature).
        
        Important :
        - fit=True sur train set uniquement
        - fit=False sur validation/test (réutilise scaler du train)
        """
        from sklearn.preprocessing import StandardScaler
        
        original_shape = sequences.shape
        # Reshape pour normaliser chaque feature indépendamment
        sequences_2d = sequences.reshape(-1, sequences.shape[-1])
        
        if fit:
            self.scaler = StandardScaler()
            sequences_normalized = self.scaler.fit_transform(sequences_2d)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            sequences_normalized = self.scaler.transform(sequences_2d)
        
        return sequences_normalized.reshape(original_shape)
    
    def process_full_pipeline(self, filepath: str, 
                             fit_scaler: bool = True) -> Dict:
        """
        Pipeline complet de prétraitement.
        """
        print(f"Loading and aggregating data from {filepath}...")
        arrivals = self.load_and_aggregate(filepath)
        
        print("Extracting features...")
        df_features = self.extract_features(arrivals)
        
        print("Creating sequences...")
        sequences, timestamps = self.create_sequences(df_features)
        
        print(f"Normalizing {sequences.shape[0]} sequences...")
        sequences_norm = self.normalize(sequences, fit=fit_scaler)
        
        return {
            'sequences': sequences_norm,
            'timestamps': timestamps,
            'raw_arrivals': arrivals,
            'feature_names': df_features.columns.tolist()
        }

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TracePreprocessor(delta_t='1H', sequence_length=24)
    filepath = "../../data/raw/2011/part-00000-of-00500.csv.gz"
    arrivals = preprocessor.load_and_aggregate(filepath)
    print(arrivals.head())
    print("Preprocessor test completed.")