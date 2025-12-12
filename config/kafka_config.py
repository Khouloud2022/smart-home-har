# config/kafka_config.py

# --- Paramètres de Connexion au Broker Kafka ---
# Le broker est exposé par Docker sur le port 9093 de l'hôte (localhost).
# Le broker est accessible par les conteneurs (Spark) via le hostname 'kafka' sur le port 9092.
# Nous utilisons le port 'EXTERNAL' (9093) pour le Producteur qui s'exécute sur l'hôte.
# Nous utilisons le port 'PLAINTEXT' (9092) pour le Consumer (Spark) qui s'exécute dans un autre conteneur Docker.

# Utilisé par le script 02_kafka_producer_stream.py (s'exécute sur l'hôte)
KAFKA_BOOTSTRAP_SERVER_EXTERNAL = "localhost:9093"

# Utilisé par le script 03_spark_streaming_inference.py (s'exécute dans le conteneur Spark)
KAFKA_BOOTSTRAP_SERVER_INTERNAL = "kafka:9092"

# --- Topics du Projet ---
# Topic pour les événements bruts des capteurs
KAFKA_TOPIC_RAW_EVENTS = "smart_home_events"

# --- Connecteur PySpark/Kafka ---
# Package requis par Spark pour lire et écrire dans Kafka (à inclure dans spark-submit)
SPARK_KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"

# --- Schéma des Messages Kafka (pour Décodage PySpark) ---
# Définition du schéma JSON des messages envoyés par le Producteur
# Il est crucial que ce schéma corresponde exactement à ce que le Producteur envoie.
from pyspark.sql.types import StructType, StructField, StringType, LongType, FloatType

KAFKA_MESSAGE_SCHEMA = StructType([
    StructField("timestamp", LongType(), True),  # Horodatage Unix en millisecondes
    StructField("sensor_id", StringType(), True), # Ex: "P02", "M04"
    StructField("value", FloatType(), True)      # Valeur du capteur (pression, mouvement, etc.)
])