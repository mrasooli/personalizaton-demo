from pyspark.sql import functions as F
from pyspark.sql import types as T

@udf(returnType=T.StringType())
def dates_udf(description):
    return str(date_pattern.sub(" ", str(description)))

def clean_trans(table_name):
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
  return tr_df

