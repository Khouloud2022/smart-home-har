from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, mean, stddev, min, max, count

# --- 1. Initialisation de Spark ---
spark = SparkSession.builder.appName("HAR_EDA").getOrCreate()

# --- 2. Chemins et Constantes ---
DATA_PATH = "/home/jovyan/work/data/raw/"
BED_PRESSURE_SENSOR = "P02"

# --- 3. Chargement et Préparation des Données Brutes ---
print("Chargement des données pour l'EDA...")
df = spark.read.csv(
    DATA_PATH + "sensor_sample_int.csv",
    header=True,
    inferSchema=True
) 

# Filtrer pour le capteur P02 et créer la colonne heure
df_bed = df.filter(col("sensor_id").cast("string") == BED_PRESSURE_SENSOR) \
           .withColumn("timestamp_s", (col("timestamp").cast("long") / 1000).cast("timestamp")) \
           .withColumn("hour_of_day", hour(col("timestamp_s"))) \
           .select("value", "hour_of_day")

# --- 4. Analyse Descriptive de la Pression (Feature 1) ---
print("\n" + "="*50)
print("Analyse Descriptive de la Pression (P02):")
df_stats = df_bed.select(
    count(col("value")).alias("Count"),
    min(col("value")).alias("Min_Pressure"),
    mean(col("value")).alias("Mean_Pressure"),
    stddev(col("value")).alias("StdDev_Pressure"),
    max(col("value")).alias("Max_Pressure")
)
df_stats.show()

# --- 5. Analyse de la Distribution Temporelle (Feature 2) ---
print("\n" + "="*50)
print("Distribution des Événements par Heure (Hour_of_Day):")
df_hour_dist = df_bed.groupBy("hour_of_day").count().orderBy("hour_of_day")
df_hour_dist.show(24)
print("="*50 + "\n")

# --- 6. Sauvegarde des résultats pour l'utilisateur (CSV) ---
# Sauvegarde des statistiques
df_stats.write.mode("overwrite").csv("/home/jovyan/work/data/processed/p02_pressure_stats.csv", header=True)
# Sauvegarde de la distribution horaire
df_hour_dist.write.mode("overwrite").csv("/home/jovyan/work/data/processed/p02_hour_distribution.csv", header=True)

spark.stop()