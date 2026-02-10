"""
Preprocessor adapté pour Google Cluster 2011.
Compatible avec les données partitionnées et TimeSeriesBuilder.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

class TracePreprocessor:
    """
    Préprocesseur pour traces Google Cluster.
    
    Workflow:
        1. Charge la série temporelle (déjà agrégée par TimeSeriesBuilder)
        2. Extrait des features temporelles
        3. Crée des séquences avec fenêtre glissante
        4. Normalise
    """
    
    def __init__(self, sequence_length: int = 288, stride: int = 12):
        """
        Args:
            sequence_length: Longueur des séquences (ex: 288 = 24h avec Δt=5min)
            stride: Pas de la fenêtre glissante (ex: 12 = 1h avec Δt=5min)
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.scaler = None
        self.feature_names = []
        
    def load_time_series(self, filepath: str) -> pd.DataFrame:
        """
        Charge une série temporelle déjà agrégée.
        
        Formats supportés:
        - time_series_5min.csv (de TimeSeriesBuilder)
        - Tout CSV avec index datetime et colonne de valeurs
        
        Args:
            filepath: Chemin vers le fichier CSV
        
        Returns:
            DataFrame avec index datetime
        """
        print(f"📂 Chargement de {filepath}...")
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        print(f"✓ Chargé: {len(df)} intervalles")
        print(f"  Période: {df.index.min()} → {df.index.max()}")
        print(f"  Colonnes: {df.columns.tolist()}")
        
        return df
    
    def extract_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Extrait des features temporelles et statistiques.
        
        Args:
            df: DataFrame avec série temporelle
            target_col: Nom de la colonne cible (auto-détecté si None)
        
        Returns:
            DataFrame avec features enrichies
        """
        print("🔧 Extraction des features...")
        
        # Auto-détecter la colonne cible
        if target_col is None:
            if 'job_count' in df.columns:
                target_col = 'job_count'
            elif 'arrival_rate' in df.columns:
                target_col = 'arrival_rate'
            elif 'num_arrivals' in df.columns:
                target_col = 'num_arrivals'
            else:
                # Prendre la première colonne numérique
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_col = numeric_cols[0]
                else:
                    raise ValueError(f"Aucune colonne cible trouvée. Colonnes: {df.columns.tolist()}")
        
        print(f"  Colonne cible: {target_col}")
        
        result = pd.DataFrame(index=df.index)
        result['value'] = df[target_col]
        
        # ========== FEATURES TEMPORELLES ==========
        
        # Heure et jour
        result['hour'] = result.index.hour
        result['day_of_week'] = result.index.dayofweek
        result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
        
        # Encodage cyclique (évite discontinuité)
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        # ========== FEATURES STATISTIQUES GLISSANTES ==========
        
        # Moyennes mobiles (avec différentes fenêtres)
        for window in [12, 24, 48]:  # 1h, 2h, 4h avec Δt=5min
            result[f'rolling_mean_{window}'] = result['value'].rolling(
                window=window, min_periods=1
            ).mean()
            
            result[f'rolling_std_{window}'] = result['value'].rolling(
                window=window, min_periods=1
            ).std().fillna(0)
            
            result[f'rolling_max_{window}'] = result['value'].rolling(
                window=window, min_periods=1
            ).max()
            
            result[f'rolling_min_{window}'] = result['value'].rolling(
                window=window, min_periods=1
            ).min()
        
        # ========== FEATURES DE CHANGEMENT ==========
        
        # Différences (taux de changement)
        for lag in [1, 6, 12]:  # 5min, 30min, 1h
            result[f'diff_{lag}'] = result['value'].diff(lag).fillna(0)
            result[f'pct_change_{lag}'] = result['value'].pct_change(lag).fillna(0)
        
        # Remplacer les inf par 0
        result = result.replace([np.inf, -np.inf], 0)
        
        # Remplir les NaN restants
        result = result.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        print(f"✓ Features créées: {len(result.columns)}")
        print(f"  {result.columns.tolist()}")
        
        return result
    
    def create_sequences(self, df: pd.DataFrame, 
                        use_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des séquences avec fenêtre glissante.
        
        Args:
            df: DataFrame avec features
            use_features: Si False, utilise seulement 'value'
        
        Returns:
            sequences: (num_sequences, sequence_length, num_features)
            timestamps: (num_sequences,) timestamps de début
        """
        print(f"✂️  Création des séquences...")
        print(f"  Longueur: {self.sequence_length}")
        print(f"  Stride: {self.stride}")
        
        if use_features:
            features = df.values  # Toutes les colonnes
            self.feature_names = df.columns.tolist()
        else:
            features = df[['value']].values  # Seulement la valeur
            self.feature_names = ['value']
        
        num_timesteps = len(features)
        num_sequences = (num_timesteps - self.sequence_length) // self.stride + 1
        
        sequences = []
        timestamps = []
        
        for i in range(0, num_timesteps - self.sequence_length + 1, self.stride):
            seq = features[i:i + self.sequence_length]
            sequences.append(seq)
            timestamps.append(df.index[i])
        
        sequences = np.array(sequences)
        timestamps = np.array(timestamps)
        
        print(f"✓ {len(sequences)} séquences créées")
        print(f"  Shape: {sequences.shape}")
        print(f"  Features: {len(self.feature_names)}")
        
        return sequences, timestamps
    
    def normalize(self, sequences: np.ndarray, 
                  fit: bool = True) -> np.ndarray:
        """
        Normalise les séquences (StandardScaler par feature).
        
        Args:
            sequences: (num_sequences, sequence_length, num_features)
            fit: Si True, fit le scaler (train set)
        
        Returns:
            Séquences normalisées
        """
        print(f"📏 Normalisation...")
        
        original_shape = sequences.shape
        # Reshape: (num_sequences * sequence_length, num_features)
        sequences_2d = sequences.reshape(-1, sequences.shape[-1])
        
        if fit:
            self.scaler = StandardScaler()
            sequences_normalized = self.scaler.fit_transform(sequences_2d)
            print(f"  Mean: {self.scaler.mean_}")
            print(f"  Std: {np.sqrt(self.scaler.var_)}")
        else:
            if self.scaler is None:
                raise ValueError("Scaler non fitted. Appelez avec fit=True d'abord.")
            sequences_normalized = self.scaler.transform(sequences_2d)
        
        # Reshape back
        sequences_normalized = sequences_normalized.reshape(original_shape)
        
        print(f"✓ Normalisé: min={sequences_normalized.min():.2f}, max={sequences_normalized.max():.2f}")
        
        return sequences_normalized
    
    def split_sequences(self, sequences: np.ndarray, timestamps: np.ndarray,
                       train_ratio: float = 0.7, 
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       random_seed: int = 42) -> Dict:
        """
        Split les séquences en train/val/test.
        
        Args:
            sequences: Séquences à splitter
            timestamps: Timestamps correspondants
            train_ratio, val_ratio, test_ratio: Ratios de split
            random_seed: Seed pour reproductibilité
        
        Returns:
            Dict avec train, val, test
        """
        print(f"✂️  Split train/val/test...")
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        n_samples = len(sequences)
        np.random.seed(random_seed)
        indices = np.random.permutation(n_samples)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        result = {
            'train': sequences[train_idx],
            'val': sequences[val_idx],
            'test': sequences[test_idx],
            'train_timestamps': timestamps[train_idx],
            'val_timestamps': timestamps[val_idx],
            'test_timestamps': timestamps[test_idx]
        }
        
        print(f"✓ Split effectué:")
        print(f"  Train: {len(result['train'])} ({len(result['train'])/n_samples*100:.1f}%)")
        print(f"  Val:   {len(result['val'])} ({len(result['val'])/n_samples*100:.1f}%)")
        print(f"  Test:  {len(result['test'])} ({len(result['test'])/n_samples*100:.1f}%)")
        
        return result
    
    def save_processed_data(self, data: Dict, output_dir: str):
        """
        Sauvegarde les données prétraitées.
        
        Args:
            data: Dict avec train/val/test
            output_dir: Dossier de sortie
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Sauvegarde dans {output_path}...")
        
        # Sauvegarder les séquences
        np.save(output_path / 'train.npy', data['train'])
        np.save(output_path / 'val.npy', data['val'])
        np.save(output_path / 'test.npy', data['test'])
        
        print(f"  ✓ train.npy: {data['train'].shape}")
        print(f"  ✓ val.npy: {data['val'].shape}")
        print(f"  ✓ test.npy: {data['test'].shape}")
        
        # Sauvegarder les timestamps
        np.save(output_path / 'train_timestamps.npy', data['train_timestamps'])
        np.save(output_path / 'val_timestamps.npy', data['val_timestamps'])
        np.save(output_path / 'test_timestamps.npy', data['test_timestamps'])
        
        # Sauvegarder le scaler
        if self.scaler is not None:
            with open(output_path / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"  ✓ scaler.pkl")
        
        # Sauvegarder la config
        config = {
            'sequence_length': self.sequence_length,
            'stride': self.stride,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'n_train': len(data['train']),
            'n_val': len(data['val']),
            'n_test': len(data['test'])
        }
        
        import json
        with open(output_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"  ✓ config.json")
    
    def process_full_pipeline(self, 
                             time_series_file: str,
                             output_dir: str,
                             use_features: bool = True,
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15) -> Dict:
        """
        Pipeline complet de prétraitement.
        
        Args:
            time_series_file: Fichier CSV de série temporelle
            output_dir: Dossier de sortie
            use_features: Utiliser features enrichies ou seulement valeur
            train_ratio, val_ratio, test_ratio: Ratios de split
        
        Returns:
            Dict avec toutes les données
        """
        print("="*70)
        print("🔄 PIPELINE COMPLET DE PRÉTRAITEMENT")
        print("="*70)
        
        # 1. Charger
        df = self.load_time_series(time_series_file)
        
        # 2. Extraire features
        if use_features:
            df_features = self.extract_features(df)
        else:
            # Juste la valeur
            if 'job_count' in df.columns:
                df_features = df[['job_count']].rename(columns={'job_count': 'value'})
            elif 'arrival_rate' in df.columns:
                df_features = df[['arrival_rate']].rename(columns={'arrival_rate': 'value'})
            else:
                df_features = df.iloc[:, :1].rename(columns={df.columns[0]: 'value'})
        
        # 3. Créer séquences
        sequences, timestamps = self.create_sequences(df_features, use_features=use_features)
        
        # 4. Normaliser (fit sur toutes les données avant split)
        sequences_norm = self.normalize(sequences, fit=True)
        
        # 5. Split
        data = self.split_sequences(
            sequences_norm, timestamps,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        # 6. Sauvegarder
        self.save_processed_data(data, output_dir)
        
        print("\n" + "="*70)
        print("✅ PIPELINE TERMINÉ")
        print("="*70)
        
        return data


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    # Configuration
    CONFIG = {
        'time_series_file': 'data/processed/time_series/time_series_5min.csv',
        'output_dir': 'data/processed/sequences',
        
        'sequence_length': 288,  # 24h avec Δt=5min
        'stride': 12,            # 1h
        
        'use_features': False,   # True = features enrichies, False = juste valeur
        
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15
    }
    
    # Créer préprocesseur
    preprocessor = TracePreprocessor(
        sequence_length=CONFIG['sequence_length'],
        stride=CONFIG['stride']
    )
    
    # Exécuter pipeline
    data = preprocessor.process_full_pipeline(
        time_series_file=CONFIG['time_series_file'],
        output_dir=CONFIG['output_dir'],
        use_features=CONFIG['use_features'],
        train_ratio=CONFIG['train_ratio'],
        val_ratio=CONFIG['val_ratio'],
        test_ratio=CONFIG['test_ratio']
    )
    
    print(f"\n🎉 Données prêtes pour l'entraînement !")
    print(f"   Lancez: python scripts/train_lstm_vae.py")