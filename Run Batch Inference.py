# Databricks notebook source
##Inputs
# catalog = "commercial_dev__jwlt"
catalog = "commercial_prod__jwlt"
schema = "project_name_a"
table_name = "winequality_test_dataset"
model_name = "wine_quality"
alias_name = "champion"



full_table_name = f"{catalog}.{schema}.{table_name}"
model_name = f"{catalog}.{schema}.{model_name}"
print(full_table_name)

# COMMAND ----------

import mlflow.pyfunc
import mlflow
from pyspark.sql.functions import struct

# COMMAND ----------

# DBTITLE 1,Get Data
test_data = spark.table(full_table_name).drop("quality")
pdf_data = test_data.toPandas()

# COMMAND ----------

# DBTITLE 1,Load Model to Test
mlflow.set_registry_uri("databricks-uc")
model_uri = f"models:/{model_name}@{alias_name}"
# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type='double')

# COMMAND ----------

# DBTITLE 1,Run Batch Inference
# Apply the model to the new data
udf_inputs = struct(*(pdf_data.columns.tolist()))

prediction_data = test_data.withColumn(
  "prediction",
  loaded_model_udf(udf_inputs)
)

display(prediction_data)
