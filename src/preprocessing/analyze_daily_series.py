#!/usr/bin/env python3
"""
Analyse des patterns d'arrivée de jobs PAR JOUR.
Crée des distributions discrètes (histogrammes) pour chaque journée type.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import json
from collections import defaultdict

# Configuration
DATA_PATH = Path("data/processed")
OUTPUT_PATH = Path("data/outputs/processed/daily")
FIGURE_PATH = Path("data/outputs/results/figures/daily")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
FIGURE_PATH.mkdir(parents=True, exist_ok=True)


class DailyPatternAnalyzer:
    """Analyse les patterns d'arrivée jour par jour."""
    
    def __init__(self, delta_t_minutes=5):
        """
        Args:
            delta_t_minutes: Pas de temps en minutes (ex: 5, 10, 15)
        """
        self.delta_t = delta_t_minutes
        self.bins_per_day = int(24 * 60 / delta_t_minutes)
        print(f"📊 Analyseur initialisé : Δt={delta_t_minutes}min, {self.bins_per_day} bins/jour")
    
    def load_time_series(self, filename='time_series_dt5min.csv'):
        """Charge la série temporelle."""
        filepath = DATA_PATH / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {filepath}")
        
        print(f"📂 Chargement de {filepath}...")
        df = pd.read_csv(filepath)
        df['time_bin'] = pd.to_datetime(df['datetime'])  # Adjust column name
        df['num_arrivals'] = df['job_count']  # Add expected column name
        
        print(f"✅ {len(df)} intervalles chargés")
        print(f"   Période : {df['time_bin'].min()} → {df['time_bin'].max()}")
        
        return df
    
    def extract_daily_distributions(self, df):
        """
        Extrait la distribution discrète pour chaque jour.
        
        Returns:
            Dict[date, np.array]: Distribution pour chaque jour
        """
        print(f"\n📅 Extraction des distributions quotidiennes...")
        
        # Ajouter la colonne date
        df['date'] = df['time_bin'].dt.date
        
        # Ajouter l'index temporel dans la journée (0 à bins_per_day-1)
        df['time_of_day_index'] = (
            df['time_bin'].dt.hour * 60 + df['time_bin'].dt.minute
        ) // self.delta_t
        
        # Grouper par jour
        daily_distributions = {}
        
        for date, group in df.groupby('date'):
            # Créer un vecteur de taille fixe pour la journée
            distribution = np.zeros(self.bins_per_day)
            
            # Remplir avec les valeurs observées
            for _, row in group.iterrows():
                idx = int(row['time_of_day_index'])
                if 0 <= idx < self.bins_per_day:
                    distribution[idx] = row['num_arrivals']
            
            daily_distributions[date] = distribution
        
        print(f"✅ {len(daily_distributions)} jours extraits")
        
        return daily_distributions
    
    def compute_daily_statistics(self, daily_distributions):
        """Calcule les statistiques pour chaque jour."""
        
        stats_per_day = {}
        
        for date, dist in daily_distributions.items():
            stats_per_day[date] = {
                'date': str(date),
                'total_jobs': int(dist.sum()),
                'mean': float(dist.mean()),
                'std': float(dist.std()),
                'min': int(dist.min()),
                'max': int(dist.max()),
                'median': float(np.median(dist)),
                'non_zero_intervals': int(np.count_nonzero(dist)),
                'zero_intervals': int(np.sum(dist == 0)),
                'peak_hour': int(np.argmax(dist) * self.delta_t / 60),
            }
        
        return stats_per_day
    
    def identify_day_types(self, df):
        """
        Identifie les types de jours (jour de semaine, weekend, etc.).
        
        Returns:
            Dict avec classification des jours
        """
        print(f"\n🏷️  Identification des types de jours...")
        
        df['date'] = df['time_bin'].dt.date
        df['day_of_week'] = df['time_bin'].dt.dayofweek
        
        day_types = defaultdict(list)
        
        for date in df['date'].unique():
            day_data = df[df['date'] == date].iloc[0]
            dow = day_data['day_of_week']
            
            if dow < 5:  # Lundi à Vendredi
                day_types['weekday'].append(date)
            else:  # Samedi, Dimanche
                day_types['weekend'].append(date)
            
            # Par jour de la semaine
            day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 
                        'friday', 'saturday', 'sunday']
            day_types[day_names[dow]].append(date)
        
        print(f"   Jours de semaine : {len(day_types['weekday'])}")
        print(f"   Jours de weekend : {len(day_types['weekend'])}")
        
        return dict(day_types)
    
    def compute_average_day_profile(self, daily_distributions, day_list=None):
        """
        Calcule le profil moyen pour un ensemble de jours.
        
        Args:
            daily_distributions: Dict des distributions quotidiennes
            day_list: Liste des dates à inclure (None = toutes)
        
        Returns:
            Tuple (mean_profile, std_profile, min_profile, max_profile)
        """
        if day_list is None:
            day_list = list(daily_distributions.keys())
        
        # Collecter les distributions
        distributions = np.array([daily_distributions[d] for d in day_list 
                                 if d in daily_distributions])
        
        if len(distributions) == 0:
            return None, None, None, None
        
        # Calculer les statistiques
        mean_profile = distributions.mean(axis=0)
        std_profile = distributions.std(axis=0)
        min_profile = distributions.min(axis=0)
        max_profile = distributions.max(axis=0)
        
        return mean_profile, std_profile, min_profile, max_profile
    
    def save_daily_distributions(self, daily_distributions, filename='daily_distributions.npz'):
        """Sauvegarde les distributions dans un fichier NPZ."""
        
        output_file = OUTPUT_PATH / filename
        
        # Convertir les dates en strings pour numpy
        dates = [str(d) for d in daily_distributions.keys()]
        distributions = np.array(list(daily_distributions.values()))
        
        np.savez_compressed(
            output_file,
            dates=dates,
            distributions=distributions,
            delta_t=self.delta_t,
            bins_per_day=self.bins_per_day
        )
        
        print(f"💾 Distributions sauvegardées : {output_file}")
    
    def save_daily_statistics(self, stats_per_day, filename='daily_statistics.json'):
        """Sauvegarde les statistiques par jour."""
        
        output_file = OUTPUT_PATH / filename
        
        # Convert date keys to strings for JSON serialization
        stats_serializable = {str(date): stats for date, stats in stats_per_day.items()}
        
        with open(output_file, 'w') as f:
            json.dump(stats_serializable, f, indent=2, default=str)
        
        print(f"💾 Statistiques sauvegardées : {output_file}")
    
    def visualize_daily_patterns(self, daily_distributions, day_types, stats_per_day):
        """Crée des visualisations des patterns quotidiens."""
        
        print(f"\n📊 Création des visualisations...")
        
        # Figure 1 : Comparaison weekday vs weekend
        self._plot_weekday_vs_weekend(daily_distributions, day_types)
        
        # Figure 2 : Profils par jour de la semaine
        self._plot_day_of_week_profiles(daily_distributions, day_types)
        
        # Figure 3 : Heatmap de tous les jours
        self._plot_daily_heatmap(daily_distributions)
        
        # Figure 4 : Distribution des statistiques
        self._plot_daily_statistics_distribution(stats_per_day)
        
        # Figure 5 : Quelques exemples de jours
        self._plot_example_days(daily_distributions, stats_per_day)
    
    def _plot_weekday_vs_weekend(self, daily_distributions, day_types):
        """Compare les profils weekday vs weekend."""
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Profil weekday
        mean_wd, std_wd, min_wd, max_wd = self.compute_average_day_profile(
            daily_distributions, day_types.get('weekday', [])
        )
        
        # Profil weekend
        mean_we, std_we, min_we, max_we = self.compute_average_day_profile(
            daily_distributions, day_types.get('weekend', [])
        )
        
        if mean_wd is not None:
            hours = np.arange(self.bins_per_day) * self.delta_t / 60
            
            # Weekday
            ax.plot(hours, mean_wd, label='Weekday (mean)', linewidth=2, color='blue')
            ax.fill_between(hours, mean_wd - std_wd, mean_wd + std_wd, 
                           alpha=0.2, color='blue', label='Weekday (±1σ)')
            
            # Weekend
            if mean_we is not None:
                ax.plot(hours, mean_we, label='Weekend (mean)', linewidth=2, color='red')
                ax.fill_between(hours, mean_we - std_we, mean_we + std_we, 
                               alpha=0.2, color='red', label='Weekend (±1σ)')
        
        ax.set_xlabel('Heure de la journée', fontsize=12)
        ax.set_ylabel('Nombre d\'arrivées', fontsize=12)
        ax.set_title('Profil Moyen : Weekday vs Weekend', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 24)
        
        plt.tight_layout()
        plt.savefig(FIGURE_PATH / 'weekday_vs_weekend.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ weekday_vs_weekend.png")
    
    def _plot_day_of_week_profiles(self, daily_distributions, day_types):
        """Profils moyens par jour de la semaine."""
        
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        axes = axes.flatten()
        
        day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        day_keys = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        hours = np.arange(self.bins_per_day) * self.delta_t / 60
        
        for idx, (name, key) in enumerate(zip(day_names, day_keys)):
            ax = axes[idx]
            
            mean_profile, std_profile, min_profile, max_profile = \
                self.compute_average_day_profile(daily_distributions, 
                                                 day_types.get(key, []))
            
            if mean_profile is not None:
                ax.plot(hours, mean_profile, linewidth=2, color='navy')
                ax.fill_between(hours, mean_profile - std_profile, 
                               mean_profile + std_profile, alpha=0.3, color='navy')
                
                n_days = len(day_types.get(key, []))
                ax.set_title(f'{name} (n={n_days})', fontsize=11, fontweight='bold')
            else:
                ax.set_title(f'{name} (pas de données)', fontsize=11)
            
            ax.set_xlabel('Heure')
            ax.set_ylabel('Arrivées')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 24)
        
        # Supprimer le dernier subplot vide
        fig.delaxes(axes[-1])
        
        plt.suptitle('Profils Moyens par Jour de la Semaine', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIGURE_PATH / 'day_of_week_profiles.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ day_of_week_profiles.png")
    
    def _plot_daily_heatmap(self, daily_distributions):
        """Heatmap de tous les jours."""
        
        # Préparer les données
        dates = sorted(daily_distributions.keys())
        data_matrix = np.array([daily_distributions[d] for d in dates])
        
        # Limiter à 100 jours max pour la lisibilité
        if len(dates) > 100:
            step = len(dates) // 100
            dates = dates[::step]
            data_matrix = data_matrix[::step]
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(dates) * 0.1)))
        
        im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        # Axes
        hours = np.arange(0, self.bins_per_day, self.bins_per_day // 24)
        hour_labels = [f'{int(h * self.delta_t / 60)}h' for h in hours]
        ax.set_xticks(hours)
        ax.set_xticklabels(hour_labels)
        
        # Dates (seulement quelques-unes pour lisibilité)
        n_dates = len(dates)
        date_step = max(1, n_dates // 20)
        date_ticks = list(range(0, n_dates, date_step))
        date_labels = [str(dates[i]) for i in date_ticks]
        ax.set_yticks(date_ticks)
        ax.set_yticklabels(date_labels, fontsize=8)
        
        ax.set_xlabel('Heure de la journée', fontsize=12)
        ax.set_ylabel('Date', fontsize=12)
        ax.set_title('Heatmap des Arrivées de Jobs (par jour)', fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Nombre d\'arrivées', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(FIGURE_PATH / 'daily_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ daily_heatmap.png")
    
    def _plot_daily_statistics_distribution(self, stats_per_day):
        """Distribution des statistiques quotidiennes."""
        
        # Extraire les métriques
        total_jobs = [s['total_jobs'] for s in stats_per_day.values()]
        means = [s['mean'] for s in stats_per_day.values()]
        maxs = [s['max'] for s in stats_per_day.values()]
        peak_hours = [s['peak_hour'] for s in stats_per_day.values()]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total jobs par jour
        ax = axes[0, 0]
        ax.hist(total_jobs, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title('Distribution : Total Jobs par Jour', fontweight='bold')
        ax.set_xlabel('Total jobs')
        ax.set_ylabel('Fréquence')
        ax.grid(True, alpha=0.3)
        
        # Moyenne d'arrivées
        ax = axes[0, 1]
        ax.hist(means, bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax.set_title('Distribution : Moyenne d\'Arrivées par Intervalle', fontweight='bold')
        ax.set_xlabel('Moyenne')
        ax.set_ylabel('Fréquence')
        ax.grid(True, alpha=0.3)
        
        # Max d'arrivées
        ax = axes[1, 0]
        ax.hist(maxs, bins=30, edgecolor='black', alpha=0.7, color='red')
        ax.set_title('Distribution : Max d\'Arrivées par Jour', fontweight='bold')
        ax.set_xlabel('Max arrivées')
        ax.set_ylabel('Fréquence')
        ax.grid(True, alpha=0.3)
        
        # Heure du pic
        ax = axes[1, 1]
        ax.hist(peak_hours, bins=24, edgecolor='black', alpha=0.7, color='green')
        ax.set_title('Distribution : Heure du Pic d\'Activité', fontweight='bold')
        ax.set_xlabel('Heure')
        ax.set_ylabel('Fréquence')
        ax.set_xlim(0, 24)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURE_PATH / 'daily_statistics_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ daily_statistics_distribution.png")
    
    def _plot_example_days(self, daily_distributions, stats_per_day):
        """Affiche quelques exemples de jours."""
        
        # Trier par total_jobs
        sorted_days = sorted(stats_per_day.items(), 
                           key=lambda x: x[1]['total_jobs'])
        
        # Sélectionner : min, médiane, max, et quelques autres
        n_days = len(sorted_days)
        indices = [0, n_days//4, n_days//2, 3*n_days//4, n_days-1]
        example_days = [sorted_days[i] for i in indices if i < len(sorted_days)]
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        hours = np.arange(self.bins_per_day) * self.delta_t / 60
        
        for idx, (date, stats) in enumerate(example_days[:6]):
            ax = axes[idx]
            
            dist = daily_distributions[date]
            
            ax.bar(hours, dist, width=self.delta_t/60*0.8, alpha=0.7)
            ax.set_title(f'{date} (Total: {stats["total_jobs"]} jobs)', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Heure')
            ax.set_ylabel('Arrivées')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 24)
        
        # Supprimer les subplots inutilisés
        for idx in range(len(example_days), 6):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Exemples de Journées (du min au max)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIGURE_PATH / 'example_days.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ example_days.png")
    
    def print_summary(self, daily_distributions, stats_per_day, day_types):
        """Affiche un résumé de l'analyse."""
        
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ DE L'ANALYSE PAR JOUR")
        print("=" * 70)
        
        print(f"\n📅 Nombre de jours analysés : {len(daily_distributions)}")
        print(f"   Jours de semaine : {len(day_types.get('weekday', []))}")
        print(f"   Jours de weekend : {len(day_types.get('weekend', []))}")
        
        # Statistiques globales
        total_jobs_all = [s['total_jobs'] for s in stats_per_day.values()]
        
        print(f"\n📊 Jobs par jour :")
        print(f"   Minimum    : {min(total_jobs_all):,}")
        print(f"   Maximum    : {max(total_jobs_all):,}")
        print(f"   Moyenne    : {np.mean(total_jobs_all):,.0f}")
        print(f"   Médiane    : {np.median(total_jobs_all):,.0f}")
        print(f"   Écart-type : {np.std(total_jobs_all):,.0f}")
        
        print("\n" + "=" * 70)


def main():
    """Fonction principale."""
    
    print("=" * 70)
    print("📊 ANALYSE DES PATTERNS D'ARRIVÉE PAR JOUR")
    print("=" * 70)
    
    # Configuration
    DELTA_T = 5  # minutes
    
    # Initialiser l'analyseur
    analyzer = DailyPatternAnalyzer(delta_t_minutes=DELTA_T)
    
    # 1. Charger la série temporelle
    print("\n📂 Étape 1/6 : Chargement des données")
    df = analyzer.load_time_series(f'time_series_dt{DELTA_T}min.csv')
    
    # 2. Extraire les distributions quotidiennes
    print("\n📊 Étape 2/6 : Extraction des distributions quotidiennes")
    daily_distributions = analyzer.extract_daily_distributions(df)
    
    # 3. Calculer les statistiques par jour
    print("\n📈 Étape 3/6 : Calcul des statistiques")
    stats_per_day = analyzer.compute_daily_statistics(daily_distributions)
    
    # 4. Identifier les types de jours
    print("\n🏷️  Étape 4/6 : Classification des jours")
    day_types = analyzer.identify_day_types(df)
    
    # 5. Sauvegarder
    print("\n💾 Étape 5/6 : Sauvegarde")
    analyzer.save_daily_distributions(daily_distributions)
    analyzer.save_daily_statistics(stats_per_day)
    
    # 6. Visualiser
    print("\n📊 Étape 6/6 : Visualisations")
    analyzer.visualize_daily_patterns(daily_distributions, day_types, stats_per_day)
    
    # Résumé
    analyzer.print_summary(daily_distributions, stats_per_day, day_types)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSE TERMINÉE")
    print("=" * 70)
    print(f"\n📁 Fichiers créés :")
    print(f"   - {OUTPUT_PATH}/daily_distributions.npz")
    print(f"   - {OUTPUT_PATH}/daily_statistics.json")
    print(f"   - {FIGURE_PATH}/*.png (5 figures)")
    print("\n📋 Prochaine étape :")
    print("   Utiliser ces distributions pour l'analyse d'incertitude")
    print("=" * 70)


if __name__ == "__main__":
    main()