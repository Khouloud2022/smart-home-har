from kafka import KafkaConsumer
from config.kafka_config import KAFKA_BOOTSTRAP_SERVER_EXTERNAL, KAFKA_TOPIC_RAW_EVENTS
import json
import time

# --- Configuration du Consumer ---
consumer = KafkaConsumer(
    KAFKA_TOPIC_RAW_EVENTS,
    bootstrap_servers=[KAFKA_BOOTSTRAP_SERVER_EXTERNAL],
    auto_offset_reset='earliest',  # Commencez à lire depuis le début du topic
    enable_auto_commit=True,
    group_id='test-consumer-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# --- Lecture des Messages ---
print(f"Démarrage de l'écoute sur le topic: {KAFKA_TOPIC_RAW_EVENTS}...")
print("Attendez que le producteur (02_kafka_producer_stream.py) envoie des messages.")

count = 0
MAX_MESSAGES = 10

try:
    # Tenter de récupérer 10 messages ou attendre 10 secondes
    for message in consumer:
        print("--- Message Reçu ---")
        print(f"Offset: {message.offset}")
        print(f"Clé: {message.key}")
        print(f"Valeur (JSON décodé): {message.value}")
        print("-------------------")
        
        count += 1
        if count >= MAX_MESSAGES:
            print(f"Limite de {MAX_MESSAGES} messages atteinte. Arrêt.")
            break
            
except Exception as e:
    print(f"Erreur lors de la consommation des messages: {e}")
    
finally:
    consumer.close()
    print("Test du Consumer terminé.")