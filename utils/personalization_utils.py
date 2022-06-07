from pyspark.sql import functions as F

def format_dict(label_column, value_column, in_dict):
    labels = in_dict[label_column]
    rates = in_dict[value_column]
    result = dict()
    for i in range(0, len(labels)):
        result[labels[i]] = rates[i]
    return result

def sample_data(sample_size, count_threshold, data):
    counted = data.groupBy("tr_merchant").count()
    counted = counted.where(F.col("count") >= count_threshold)
    counted = counted \
        .withColumn("sample_rate", sample_size / F.col("count")) \
        .withColumn("sample_rate", F.when(F.col("sample_rate") > 1, 1).otherwise(F.col("sample_rate")))
    sample_rates = counted.select("tr_merchant", "sample_rate").toPandas().to_dict()
    sample_rates = format_dict("tr_merchant", "sample_rate", sample_rates)
    result = data.sampleBy("tr_merchant", sample_rates)
    return result