from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# --- 1. Initialize Spark ---
spark = SparkSession.builder.appName("HAR_Batch_Training").getOrCreate()

# --- 2. Paths and Constants ---
DATA_PATH = "/home/jovyan/work/data/raw/sensor_sample_int.csv"
MODEL_PATH = "/home/jovyan/work/models/spark_har_pipeline"
BED_PRESSURE_SENSOR = "5892"

print("="*70)
print("HAR BATCH ML TRAINING WITH EVALUATION")
print("="*70)

# --- 3. Load CSV ---
print("\n[1/6] Loading sensor data...")
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
print("     ✓ CSV loaded successfully")

# --- 4. Filter and prepare data ---
print("\n[2/6] Filtering sensor data...")
df_bed = df.filter(col("sensor_id").cast("string") == BED_PRESSURE_SENSOR)
print(f"     ✓ Filtered for sensor {BED_PRESSURE_SENSOR}")

# --- 5. Feature engineering ---
print("\n[3/6] Feature engineering...")
df_features = df_bed.withColumn(
    "timestamp_s",
    col("timestamp").cast("timestamp")
).withColumn(
    "hour_of_day",
    hour(col("timestamp_s"))
).withColumn(
    "value_float",
    col("value").cast("float")
).withColumn(
    "is_sleeping",
    when(
        (col("value").cast("float") > 500.0) &
        ((hour(col("timestamp").cast("timestamp")) >= 22) | 
         (hour(col("timestamp").cast("timestamp")) < 7)),
        1.0
    ).otherwise(0.0)
)

df_ml = df_features.select(
    col("value_float").alias("pressure_value"),
    col("hour_of_day"),
    col("is_sleeping").alias("label")
).na.drop()

print("     ✓ Features created")

# Split data into train (70%) and test (30%)
print("\n[4/6] Splitting data (70% train, 30% test)...")
(df_train, df_test) = df_ml.randomSplit([0.7, 0.3], seed=42)
print(f"     ✓ Training set size: {df_train.count()} (approximate)")
print(f"     ✓ Test set size: {df_test.count()} (approximate)")

# --- 6. Create ML Pipeline ---
print("\n[5/6] Creating and training model...")

pipeline = Pipeline(stages=[
    VectorAssembler(inputCols=["pressure_value", "hour_of_day"], outputCol="features_raw"),
    StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False),
    LogisticRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.3)
])

# Train on 70% of data
model = pipeline.fit(df_train)
print("     ✓ Model trained successfully")

# --- 7. Make predictions on test set ---
print("\n[6/6] Evaluating model on test set...")
predictions = model.transform(df_test)

# --- 8. Calculate evaluation metrics ---
print("\n" + "="*70)
print("EVALUATION METRICS")
print("="*70)

# Binary Classification Metrics
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
try:
    auc = binary_evaluator.evaluate(predictions)
    print(f"\n📊 Binary Classification Metrics:")
    print(f"   AUC (Area Under ROC Curve): {auc:.4f}")
except:
    print(f"\n📊 Binary Classification Metrics:")
    print(f"   AUC: (skipped - requires probability predictions)")

# Multiclass Classification Metrics
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = multiclass_evaluator.evaluate(predictions)
print(f"\n📊 Accuracy Metrics:")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Additional metrics
multiclass_evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
precision = multiclass_evaluator_precision.evaluate(predictions)
print(f"   Weighted Precision: {precision:.4f}")

multiclass_evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
recall = multiclass_evaluator_recall.evaluate(predictions)
print(f"   Weighted Recall: {recall:.4f}")

multiclass_evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1 = multiclass_evaluator_f1.evaluate(predictions)
print(f"   F1-Score: {f1:.4f}")

# Confusion Matrix (approximate)
print(f"\n📊 Confusion Matrix (from test predictions):")
predictions.groupBy("label", "prediction").count().show()

# Label distribution
print(f"\n📊 Label Distribution in Test Set:")
predictions.groupBy("label").count().show()

# --- 9. Save model ---
print(f"\n" + "="*70)
print("SAVING MODEL")
print("="*70)

try:
    model.write().overwrite().save(MODEL_PATH)
    print(f"\n✓ Model saved to {MODEL_PATH}")
except Exception as e:
    print(f"\n✗ Error saving model: {e}")

print(f"\n" + "="*70)
print("✓ TRAINING AND EVALUATION COMPLETE!")
print("="*70 + "\n")

spark.stop()