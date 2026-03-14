# Scénarios Générés - Format CSV

## 📁 Fichiers

- `scenario_nominal.csv` : Scénario scenario_nominal
- `scenario_optimistic.csv` : Scénario scenario_optimistic
- `scenario_pessimistic.csv` : Scénario scenario_pessimistic
- `scenarios_random.csv` : Scénario scenarios_random
- `scenarios_stress.csv` : Scénario scenarios_stress
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
- Date de début : 2011-05-01 00:00:00
- Intervalle : 5 minutes
- Durée typique : 0.1 heures

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

### scenario_nominal
- Points: 100
- Moyenne: 945.85
- Écart-type: 125.82
### scenario_optimistic
- Points: 100
- Moyenne: 714.14
- Écart-type: 105.65
### scenario_pessimistic
- Points: 100
- Moyenne: 1250.58
- Écart-type: 421.37
### scenarios_random
- Points: 1000
- Moyenne: 934.63
- Écart-type: 202.53
### scenarios_stress
- Points: 1000
- Moyenne: 1280.95
- Écart-type: 797.61

## 🔄 Génération

Ces fichiers ont été générés à partir des modèles VAE entraînés sur les données
Google Cluster 2011. Les scénarios représentent différentes conditions de charge :

- **nominal** : Charge typique/moyenne
- **optimistic** : Charge basse
- **pessimistic** : Charge haute
- **stressed** : Charge variable avec pics
- **random** : Échantillon aléatoire

---

Généré le : 2026-02-26 15:01:32
