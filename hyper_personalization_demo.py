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

# DBTITLE 1,Clean the dataset
from pyspark.sql import functions as F
from pyspark.sql import types as T

@udf(returnType=T.StringType())
def dates_udf(description):
    return str(date_pattern.sub(" ", str(description)))

tr_df = spark.sql("select * from "+table_name)
tr_df_cleaned = (
    tr_df
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

from fasttext_mlflow import FastTextMLFlowModel

# COMMAND ----------

# DBTITLE 1,Prep the data in the right format for FastText
tr_df_fasttext = tr_df_cleaned.withColumn(
    "fasttext",
    F.concat(
        F.concat(
            F.lit("__label__"),
            F.regexp_replace(F.col("tr_merchant"), "\\s+", "-")
        ),
        F.lit(" "),
        F.col("tr_description_clean")
    )
)

# tr_df_fasttext.write.mode("overwrite").format("delta").save(getParam("transactions_fasttext"))

# COMMAND ----------



# spark.sql("CREATE TABLE IF NOT EXISTS " + table_name + " USING DELTA LOCATION '" + transaction_path + "'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imbalanced dataset
# MAGIC When it comes to card transactions data, it is very common to come across a large disparity in available data for different merchants. For example it is to be expected that "Amazon" will drive much more transactions than "MyLittleCornerShop". Let's inspect the distribution of our raw data.

# COMMAND ----------

# DBTITLE 1,Sampling
from utils.personalization_utils import  format_dict, sample_data

tr_df_sampled = sample_data(5000, 100, tr_df)
display(tr_df_sampled.groupBy("tr_merchant").count().orderBy("count"))

# COMMAND ----------

# DBTITLE 1,Split to training and validation
import pyspark.sql.functions as F
from pyspark.sql.window import Window

w =  Window.partitionBy("tr_merchant").orderBy(F.rand())
df = tr_df_sampled.withColumn("class_percentile", F.bround(F.percent_rank().over(w), 4))

df.where("class_percentile < 0.9") \
  .write \
  .mode("overwrite") \
  .format("delta") \
  .save(getParam('transactions_train_raw'))

df.where("class_percentile >= 0.9") \
  .write \
  .mode("overwrite") \
  .format("delta") \
  .save(getParam('transactions_valid_raw'))
