from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# --- 1. Initialisation de Spark ---
spark = SparkSession.builder.appName("HAR_Batch_Training").getOrCreate()

# --- 2. Chemins et Constantes ---
DATA_PATH = "/home/jovyan/work/data/raw/"
MODEL_PATH = "/home/jovyan/work/models/spark_har_pipeline"

# Use sensor 5892 (bed pressure sensor from your data)
BED_PRESSURE_SENSOR = "5892"

# --- 3. Chargement des Données Brutes (avec limit pour diagnostic rapide) ---
print("="*60)
print("Chargement et analyse rapide des données...")
print("="*60)

df = spark.read.csv(
    DATA_PATH + "sensor_sample_int.csv",
    header=True,
    inferSchema=True
)

# FASTER APPROACH: Use limit() to get sample without counting entire 9.2GB file
print("\n✓ Fichier lu (sans compter tous les enregistrements)")

# Show unique sensors quickly
print("\n✓ Capteurs disponibles:")
sensors_sample = df.limit(100000).select('sensor_id').distinct().collect()
for sensor in sensors_sample:
    print(f"   - sensor_id: {sensor[0]}")

# --- 4. Filtrer pour le capteur de pression du lit ---
print(f"\n" + "="*60)
print(f"Filtrage du capteur {BED_PRESSURE_SENSOR}")
print("="*60)

df_bed = df.filter(col("sensor_id").cast("string") == BED_PRESSURE_SENSOR)

print(f"\n✓ Données filtrées pour capteur {BED_PRESSURE_SENSOR}")
print("Premiers exemples:")
df_bed.show(5)

# --- 5. Feature Engineering & Création du Label ---
print(f"\n" + "="*60)
print("Feature Engineering & création du label")
print("="*60)

# Convertir timestamp et extraire l'heure
df_bed = df_bed.withColumn(
    "timestamp_s",
    col("timestamp").cast("timestamp")
).withColumn(
    "hour_of_day", 
    hour(col("timestamp_s"))
)

# Convertir value en float
df_bed = df_bed.withColumn("value_float", col("value").cast("float"))

# Label: 1 si Pression > 500 ET Heure Nocturne (22h-7h)
PRESSURE_THRESHOLD = 500.0
df_labeled = df_bed.withColumn(
    "is_sleeping",
    when(
        (col("value_float") > PRESSURE_THRESHOLD) &
        ((col("hour_of_day") >= 22) | (col("hour_of_day") < 7)),
        1
    ).otherwise(0)
)

# Préparer pour le ML
df_ml = df_labeled.select(
    col("value_float").alias("pressure_value"),
    col("hour_of_day"),
    col("is_sleeping").alias("label")
)

# Enlever les valeurs nulles
df_ml = df_ml.na.drop()

print(f"\n✓ Données préparées")
print("Aperçu des données ML:")
df_ml.show(5)

# --- 6. Création du Pipeline ML ---
print(f"\n" + "="*60)
print("Création du Pipeline ML")
print("="*60)

assembler = VectorAssembler(
    inputCols=["pressure_value", "hour_of_day"],
    outputCol="features_raw"
)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=False
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.3
)

pipeline = Pipeline(stages=[assembler, scaler, lr])

print("\n✓ Pipeline créé avec 3 étapes:")
print("  1. VectorAssembler")
print("  2. StandardScaler")
print("  3. LogisticRegression")

# --- 7. Entraînement (avec sample pour vitesse) ---
print(f"\n" + "="*60)
print("Entraînement du modèle (sur 10% de l'échantillon)")
print("="*60)

# Sample 10% of data for faster training
df_sample = df_ml.sample(fraction=0.1, seed=42)

print("\n✓ Entraînement en cours...")
model = pipeline.fit(df_sample)
print("✓ Modèle entraîné avec succès!")

# --- 8. Sauvegarde du modèle ---
print(f"\n" + "="*60)
print("Sauvegarde du modèle")
print("="*60)

try:
    model.write().overwrite().save(MODEL_PATH)
    print(f"\n✓ Modèle sauvegardé avec succès!")
    print(f"  Chemin: {MODEL_PATH}")
except Exception as e:
    print(f"\n✗ ERREUR lors de la sauvegarde: {e}")

print(f"\n" + "="*60)
print("✓ ENTRAÎNEMENT COMPLÉTÉ!")
print("="*60)

spark.stop()