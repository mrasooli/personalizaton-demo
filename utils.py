from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import udf
import pyspark

@udf(returnType=T.StringType())
def dates_udf(description):
    return str(date_pattern.sub(" ", str(description)))

def clean_trans(table_name):
  df = pyspark.sql("select * from "+table_name)
  df_cleaned = (
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
  return df_cleaned

