import sys
import time
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
from config.kafka_config import KAFKA_BOOTSTRAP_SERVER_EXTERNAL, KAFKA_TOPIC_RAW_EVENTS

# --- Configuration de l'Admin ---
admin_client = KafkaAdminClient(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVER_EXTERNAL,
    client_id='topic_initializer'
)

# --- Définition du Topic ---
topic_list = [
    NewTopic(
        name=KAFKA_TOPIC_RAW_EVENTS,
        num_partitions=1,           # Une seule partition suffit pour ce mini-projet
        replication_factor=1        # Un seul broker dans notre docker-compose
    )
]

# --- Création du Topic ---
print(f"Tentative de création du topic Kafka: {KAFKA_TOPIC_RAW_EVENTS}")

try:
    # Le Kafka broker peut prendre un moment à démarrer; on attend quelques secondes
    time.sleep(5) 
    admin_client.create_topics(new_topics=topic_list, validate_only=False)
    print(f"Topic '{KAFKA_TOPIC_RAW_EVENTS}' créé avec succès.")

except TopicAlreadyExistsError:
    print(f"Topic '{KAFKA_TOPIC_RAW_EVENTS}' existe déjà. Poursuite...")

except Exception as e:
    # Peut arriver si le broker n'est pas encore prêt
    print(f"Erreur lors de la connexion au broker Kafka ({KAFKA_BOOTSTRAP_SERVER_EXTERNAL}). Assurez-vous qu'il est en cours d'exécution.")
    print(f"Détails de l'erreur: {e}")
    sys.exit(1)

finally:
    admin_client.close()