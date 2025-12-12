import pandas as pd
import json
import time
from kafka import KafkaProducer
from config.kafka_config import KAFKA_BOOTSTRAP_SERVER_EXTERNAL, KAFKA_TOPIC_RAW_EVENTS

# --- 1. Initialisation de Kafka ---
try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVER_EXTERNAL],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    print(f"Producteur Kafka initialisé sur {KAFKA_BOOTSTRAP_SERVER_EXTERNAL}")
except Exception as e:
    print(f"Erreur d'initialisation Kafka. Assurez-vous que le broker est lancé (docker-compose up -d). Erreur: {e}")
    exit(1)

# --- 2. Chargement des Données à Simuler ---
# Remarque: Le chemin est relatif à l'endroit où le script est exécuté (dans le dossier src/)
DATA_FILE = "../data/raw/sensor_sample_int.csv" 
try:
    # Charger uniquement les colonnes nécessaires pour économiser la mémoire
    df = pd.read_csv(DATA_FILE, usecols=["Time", "Sensor_ID", "Value"]).dropna()
    # Nous allons simuler la pression du lit (P02) et le mouvement (M04)
    df_stream = df[df['Sensor_ID'].isin(['P02', 'M04'])].reset_index(drop=True)
    print(f"Chargement de {len(df_stream)} événements pour la simulation...")
except FileNotFoundError:
    print(f"Erreur: Le fichier de données {DATA_FILE} est introuvable. Placez les CSV dans data/raw/")
    exit(1)

# --- 3. Envoi du Flux ---
print("Démarrage de la simulation du flux de capteurs...")
for index, row in df_stream.iterrows():
    # Création du message JSON basé sur le schéma défini dans kafka_config.py
    message = {
        "timestamp": int(row['Time']),  # Supposé être en ms
        "sensor_id": row['Sensor_ID'],
        "value": float(row['Value'])
    }
    
    # Envoi à Kafka
    producer.send(KAFKA_TOPIC_RAW_EVENTS, value=message)
    
    # Affichage pour le suivi
    if index % 100 == 0:
        print(f"Événement #{index}: Envoyé {row['Sensor_ID']} avec la valeur {row['Value']}")
        
    # Délai pour simuler un flux en temps réel
    time.sleep(0.01)  # Envoi rapide (100 messages/s) pour la démo

producer.flush()
print("Simulation terminée.")