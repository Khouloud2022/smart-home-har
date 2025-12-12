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
BED_PRESSURE_SENSOR = "P02"

# --- 3. Chargement des Données Brutes ---
print("Chargement des données brutes de capteurs...")

try:
    df = spark.read.csv(
        DATA_PATH + "sensor_sample_int.csv",
        header=True,
        inferSchema=True
    )
    print(f"✓ Fichier chargé avec succès")
    print(f"  Nombre de lignes: {df.count()}")
    print(f"  Colonnes: {df.columns}")
    df.show(5)
    
except Exception as e:
    print(f"✗ ERREUR lors de la lecture du fichier: {e}")
    print(f"  Chemin attendu: {DATA_PATH}sensor_sample_int.csv")
    spark.stop()
    exit(1)

# Vérifier que la colonne sensor_id existe
if "sensor_id" not in df.columns:
    print(f"✗ ERREUR: colonne 'sensor_id' introuvable!")
    print(f"  Colonnes disponibles: {df.columns}")
    spark.stop()
    exit(1)

# --- 4. Filtrage du capteur ---
print(f"\nFiltrage du capteur {BED_PRESSURE_SENSOR}...")
df_bed = df.filter(col("sensor_id").cast("string") == BED_PRESSURE_SENSOR)

if df_bed.count() == 0:
    print(f"✗ ERREUR: Aucune donnée trouvée pour le capteur {BED_PRESSURE_SENSOR}")
    print(f"  Capteurs disponibles: {df.select('sensor_id').distinct().collect()}")
    spark.stop()
    exit(1)

print(f"✓ {df_bed.count()} lignes filtrées pour le capteur {BED_PRESSURE_SENSOR}")

# --- 5. Feature Engineering & Création du Label ---
df_bed = df_bed.withColumn(
    "timestamp_s",
    (col("timestamp").cast("long") / 1000).cast("timestamp")
).withColumn("hour_of_day", hour(col("timestamp_s")))

BED_PRESSURE_THRESHOLD = 1500.0
df_labeled = df_bed.withColumn(
    "is_sleeping",
    when(
        (col("value") > BED_PRESSURE_THRESHOLD) &
        ((col("hour_of_day") >= 22) | (col("hour_of_day") < 7)),
        1
    ).otherwise(0)
)

df_ml = df_labeled.select(
    col("value").alias("pressure_value"),
    col("hour_of_day"),
    col("is_sleeping").alias("label")
)

df_ml = df_ml.na.drop()
print(f"✓ Données préparées: {df_ml.count()} lignes")

# --- 6. Création du Pipeline ML ---
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

lr = LogisticRegression(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[assembler, scaler, lr])

# --- 7. Entraînement et Sauvegarde ---
print("\nDémarrage de l'entraînement du modèle...")
model = pipeline.fit(df_ml.sample(fraction=0.1, seed=42))

try:
    model.write().overwrite().save(MODEL_PATH)
    print(f"✓ Modèle entraîné et sauvegardé à: {MODEL_PATH}")
except Exception as e:
    print(f"✗ ERREUR lors de la sauvegarde: {e}")
    
spark.stop()