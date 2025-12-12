# src/01_ml_training_batch.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# --- 1. Initialisation de Spark ---
spark = SparkSession.builder.appName("HAR_Batch_Training").getOrCreate()

# --- 2. Chemins et Constantes ---
# NOUVEAUX CHEMINS pour l'image Docker jupyter/pyspark-notebook
DATA_PATH = "/home/jovyan/work/data/raw/"
MODEL_PATH = "/home/jovyan/work/models/spark_har_pipeline"
BED_PRESSURE_SENSOR = "P02"

# --- 3. Chargement des Données Brutes ---
print("Chargement des données brutes de capteurs...")
df = spark.read.csv(
    DATA_PATH + "sensor_sample_int.csv",
    header=True,
    inferSchema=True
) 
# ATTENTION: Nous utilisons les noms de colonnes inférés par Spark (timestamp, sensor_id, value)

# Filtrer pour ne garder que le capteur de pression du lit (P02)
# Correction: Utilisation de 'sensor_id' (minuscule)
df_bed = df.filter(col("sensor_id") == BED_PRESSURE_SENSOR)

# --- 4. Feature Engineering & Création du Label (is_sleeping) ---
# Correction: Utilisation de 'timestamp' au lieu de 'timestamp_ms'
# Nous supposons toujours que la colonne 'timestamp' est en millisecondes (MS), d'où la division par 1000.
df_bed = df_bed.withColumn(
    "timestamp_s",
    (col("timestamp").cast("long") / 1000).cast("timestamp")
) \
.withColumn("hour_of_day", hour(col("timestamp_s")))
# Simuler le label 'is_sleeping': 1 si Pression > 1500 (personne au lit) ET Heure est Nocturne (22h à 7h)
BED_PRESSURE_THRESHOLD = 1500.0

# Correction: Utilisation de 'value' au lieu de 'Value'
df_labeled = df_bed.withColumn(
    "is_sleeping",
    when(
        (col("value") > BED_PRESSURE_THRESHOLD) &
        ((col("hour_of_day") >= 22) | (col("hour_of_day") < 7)),
        1
    ).otherwise(0)
)

# Nous simplifions les features pour le ML: Pression et l'Heure
df_ml = df_labeled.select(
    col("value").alias("pressure_value"), # 'value' est le capteur de pression
    col("hour_of_day"),
    col("is_sleeping").alias("label")
)

# Enlever les valeurs nulles pour l'entraînement
df_ml = df_ml.na.drop()

# --- 5. Création du Pipeline ML ---
# 5a. VectorAssembler: Combine les features en un seul vecteur
assembler = VectorAssembler(
    inputCols=["pressure_value", "hour_of_day"],
    outputCol="features_raw"
)

# 5b. StandardScaler: Mise à l'échelle (facultatif mais recommandé pour la régression logistique)
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=False
)

# 5c. Algorithme: Régression Logistique
lr = LogisticRegression(featuresCol="features", labelCol="label")

# 5d. Pipeline: La chaîne complète des étapes
pipeline = Pipeline(stages=[assembler, scaler, lr])

# --- 6. Entraînement et Sauvegarde ---
print("Démarrage de l'entraînement du modèle...")
# Utiliser un petit échantillon (0.1) pour accélérer le temps d'exécution
model = pipeline.fit(df_ml.sample(fraction=0.1, seed=42))

# Sauvegarde du pipeline COMPLET
model.write().overwrite().save(MODEL_PATH)
print(f"Modèle entraîné et sauvegardé à: {MODEL_PATH}")

spark.stop()