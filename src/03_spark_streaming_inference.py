from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, hour, current_timestamp
from pyspark.ml import PipelineModel
from config.kafka_config import KAFKA_BOOTSTRAP_SERVER_INTERNAL, KAFKA_TOPIC_RAW_EVENTS, KAFKA_MESSAGE_SCHEMA, SPARK_KAFKA_PACKAGE

# --- 1. Initialisation de Spark ---
# Le paramètre packages est normalement passé via spark-submit, mais on le configure ici pour l'environnement
spark = SparkSession.builder \
    .appName("HAR_Streaming_Inference") \
    .config("spark.jars.packages", SPARK_KAFKA_PACKAGE) \
    .getOrCreate()

# Ajuster le niveau de log pour ne voir que les messages importants
spark.sparkContext.setLogLevel("WARN")
print("Spark Session pour Structured Streaming démarrée.")

# --- 2. Chemins et Constantes ---
# ANCIEN CHEMIN:
# MODEL_PATH = "/opt/spark/work-dir/models/spark_har_pipeline" 

# NOUVEAU CHEMIN:
MODEL_PATH = "/home/jovyan/work/models/spark_har_pipeline"
BED_PRESSURE_SENSOR = "P02"

# --- 3. Chargement du Modèle ML ---
try:
    loaded_model = PipelineModel.load(MODEL_PATH)
    print("Pipeline ML chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle. Avez-vous exécuté 01_ml_training_batch.py ? Erreur: {e}")
    spark.stop()
    exit(1)

# --- 4. Lecture du Flux Kafka ---
# Lire le flux brut (valeurs binaires)
raw_stream_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVER_INTERNAL) \
    .option("subscribe", KAFKA_TOPIC_RAW_EVENTS) \
    .option("startingOffsets", "latest") \
    .load()

# --- 5. Préparation des Données pour l'Inférence ---
# 5a. Décoder la valeur JSON
decoded_df = raw_stream_df.select(
    from_json(col("value").cast("string"), KAFKA_MESSAGE_SCHEMA).alias("data")
).select("data.*")

# 5b. Filtrer et Préparer les Features pour le Modèle
# Nous n'avons besoin que des événements de pression du lit (P02) pour notre modèle
# et nous devons recréer la colonne 'hour_of_day' comme dans le script d'entraînement
inference_df = decoded_df.filter(col("sensor_id") == BED_PRESSURE_SENSOR) \
    .withColumn("timestamp_s", (col("timestamp") / 1000).cast("timestamp")) \
    .withColumn("hour_of_day", hour(col("timestamp_s"))) \
    .withColumnRenamed("value", "pressure_value") \
    .select("timestamp", "sensor_id", "pressure_value", "hour_of_day")

# --- 6. Application du Modèle (Inférence) ---
prediction_stream = loaded_model.transform(inference_df)

# Sélection finale et formatage pour l'affichage
output_stream = prediction_stream.withColumn(
    "current_ts", current_timestamp()
).select(
    col("current_ts").alias("Processing_Time"),
    col("timestamp").alias("Event_Time_ms"),
    col("pressure_value"),
    col("hour_of_day"),
    col("prediction").alias("Is_Sleeping_Prediction") # 0.0 ou 1.0
)

# --- 7. Écriture du Flux (Console) ---
print("Démarrage de l'inférence en temps réel. Les résultats s'afficheront ci-dessous...")
query = output_stream.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()