# Databricks notebook source
# MAGIC %run ./personalization_config

# COMMAND ----------

# MAGIC %sql
# MAGIC select tr_date, tr_description, tr_amount from transactions

# COMMAND ----------

from training_file import TrainingFile

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocessing
# MAGIC 
# MAGIC The transaction narrative and merchant description is a free form text filled in by a merchant without common guidelines or industry standards, hence requiring a data science approach to this data inconsistency problem. In this solution accelerator, we demonstrate how text classification techniques can help organizations better understand the brand hidden in any transaction narrative given a reference data set of merchants. How close is the transaction description `STARBUCKS LONDON 1233-242-43 2021` to the company "Starbucks"? 

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T

@udf(returnType=T.StringType())
def dates_udf(description):
    return str(date_pattern.sub(" ", str(description)))

tr_df = spark.sql("select * from "+table_name)
tr_df_cleaned = (
    df
        .withColumn("tr_description_clean", dates_udf(F.col("tr_description")))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), price_regex, ""))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "(\(+)|(\)+)", ""))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "&", " and "))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "[^a-zA-Z0-9]+", " "))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "\\s+", " "))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "\\s+x{2,}\\s+", " ")) 
        .withColumn("tr_description_clean", F.trim(F.col("tr_description_clean")))
)
display(tr_df_cleaned.select("tr_merchant", "tr_description", "tr_description_clean"))

# COMMAND ----------


