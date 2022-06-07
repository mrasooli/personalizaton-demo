# Databricks notebook source
# MAGIC %pip install fasttext==0.9.2

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import re
from pathlib import Path

# We ensure that all objects created in that notebooks will be registered in a user specific database. 
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split('@')[0]

# Please replace this cell should you want to store data somewhere else.
database_name = '{}_merchcat'.format(re.sub('\W', '_', username))
_ = sql("CREATE DATABASE IF NOT EXISTS {}".format(database_name))

# Similar to database, we will store actual content on a given path
home_directory = '/FileStore/{}/merchcat'.format(username)
dbutils.fs.mkdirs(home_directory)

# Where we might stored temporary data on local disk
temp_directory = "/tmp/{}/merchcat".format(username)
Path(temp_directory).mkdir(parents=True, exist_ok=True)

# COMMAND ----------

import re

config = {
  'num_executors'             :  '8',
  'model_name'                :  'merchcat_{}'.format(re.sub('\.', '_', username)),
  'transactions_raw'          :  '/mnt/industry-gtm/fsi/datasets/card_transactions',
  'transactions'              :  '{}/transactions'.format(home_directory),
  'transactions_fasttext'     :  '{}/labeled_transactions'.format(home_directory),
  'transactions_model_dir'    :  '{}/fasttext'.format(home_directory),
  'transactions_train_raw'    :  '{}/labeled_transactions_train_raw'.format(home_directory),
  'transactions_train_hex'    :  '{}/labeled_transactions_train_hex'.format(home_directory),
  'transactions_valid_raw'    :  '{}/labeled_transactions_valid_raw'.format(home_directory),
  'transactions_valid_hex'    :  '{}/labeled_transactions_valid_hex'.format(home_directory),
  'num_executors'             :  '8',
  'model_name'                :  'transbed_{}'.format(re.sub('\.', '_', username)),
  'transactions_raw'          :  '/mnt/industry-gtm/fsi/datasets/card_transactions',
  'merchant_edges'            :  '{}/merchant_edges'.format(home_directory),
  'merchant_nodes'            :  '{}/merchant_nodes'.format(home_directory),
  'shopping_trips'            :  '{}/shopping_trips'.format(home_directory),
  'merchant_vectors'          :  '{}/merchant_vectors'.format(home_directory),
  'shopping_trip_size'        :  '5',
  'shopping_trip_days'        :  '2',
  'shopping_trip_number'      :  '1000'
}

# COMMAND ----------

import pandas as pd
 
# as-is, we simply retrieve dictionary key, but the reason we create a function
# is that user would be able to replace dictionary to application property file
# without impacting notebook code
def getParam(s):
  return config[s]
 
# passing configuration to scala
spark.createDataFrame(pd.DataFrame(config, index=[0])).createOrReplaceTempView('esg_config')

# COMMAND ----------

# MAGIC %scala
# MAGIC val cdf = spark.read.table("esg_config")
# MAGIC val row = cdf.head()
# MAGIC val config = cdf.schema.map(f => (f.name, row.getAs[String](f.name))).toMap
# MAGIC def getParam(s: String) = config(s)

# COMMAND ----------

def tear_down():
  import shutil
  try:
    shutil.rmtree(temp_directory)
  except:
    pass
  dbutils.fs.rm(home_directory, True)
  _ = sql("DROP DATABASE IF EXISTS {} CASCADE".format(database_name))

# COMMAND ----------

from pyspark.sql import functions as F

tr_df = (
    spark
        .read
        .format('delta')
        .load(getParam('transactions_raw'))
        .select('tr_date', 'tr_merchant', 'tr_description', 'tr_amount')
        .filter(F.expr('tr_merchant IS NOT NULL'))
)

# COMMAND ----------

tr_df.write.format("delta").mode("overwrite").save(getParam("transactions"))

# COMMAND ----------

spark.sql("USE DATABASE "+database_name) 

# COMMAND ----------

transaction_path = getParam("transactions")
table_name = 'transactions'

# COMMAND ----------

# Create the table.
spark.sql("CREATE TABLE IF NOT EXISTS " + table_name + " USING DELTA LOCATION '" + transaction_path + "'")

# COMMAND ----------

import re

nDAY = r'(?:[0-3]?\d)'  # day can be from 1 to 31 with a leading zero 

nMNTH = r'(?:11|12|10|0?[1-9])' # month can be 1 to 12 with a leading zero

nYR = r'(?:(?:19|20)\d\d)'  # I've restricted the year to being in 20th or 21st century on the basis 

nDELIM = r'(?:[\/\-\._])?' 

NUM_DATE = f"""
    (?P<num_date>
        (?:^|\D) # new bit here
        (?:
        # YYYY-MM-DD
        (?:{nYR}(?P<delim1>[\/\-\._]?){nMNTH}(?P=delim1){nDAY})
        |
        # YYYY-DD-MM
        (?:{nYR}(?P<delim2>[\/\-\._]?){nDAY}(?P=delim2){nMNTH})
        |
        # DD-MM-YYYY
        (?:{nDAY}(?P<delim3>[\/\-\._]?){nMNTH}(?P=delim3){nYR})
        |
        # MM-DD-YYYY
        (?:{nMNTH}(?P<delim4>[\/\-\._]?){nDAY}(?P=delim4){nYR})
        )
        (?:\D|$) # new bit here
    )"""

DAY = r"""
(?:
    # search 1st 2nd 3rd etc, or first second third
    (?:[23]?1st|2{1,2}nd|\d{1,2}th|2?3rd|first|second|third|fourth|fifth|sixth|seventh|eighth|nineth)
    |
    # or just a number
    (?:[0123]?\d)
)"""

MONTH = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)'

YEAR = r"""(?:(?:[12]?\d|')?\d\d)"""

DELIM = r'(?:\s*(?:[\s\.\-\\/,]|(?:of))\s*)'

YEAR_4D = r"""(?:[12]\d\d\d)"""

DATE_PATTERN = f"""(?P<wordy_date>
    # non word character or start of string
    (?:^|\W)
        (?:
            # match various combinations of year month and day 
            (?:
                # 4 digit year
                (?:{YEAR_4D}{DELIM})?
                    (?:
                    # Day - Month
                    (?:{DAY}{DELIM}{MONTH})
                    |
                    # Month - Day
                    (?:{MONTH}{DELIM}{DAY})
                    )
                # 2 or 4 digit year
                (?:{DELIM}{YEAR})?
            )
            |
            # Month - Year (2 or 3 digit)
            (?:{MONTH}{DELIM}{YEAR})
            # non delimited dates
            |
            (?:{DAY}{MONTH}{YEAR})
            |
            (?:{DAY}{MONTH}{YEAR_4D})
            |
            (?:xx{DELIM}xx{DELIM}{YEAR_4D})
        )
    # non-word character or end of string
    (?:$|\W)
)"""

TIME = r"""(?:
(?:
# first number should be 0 - 59 with optional leading zero.
[012345]?\d
# second number is the same following a colon or a dot or h
(:|\.|h)[012345]\d
)
# next we add our optional seconds number in the same format
(?::[012345]\d)?
# and finally add optional am or pm possibly with . and spaces
(?:\s*(?:a|p)\.?m\.?)?
)"""

COMBINED = f"""(?P<combined>
    (?:
        # time followed by date, or date followed by time
        {TIME}?{DATE_PATTERN}{TIME}?
        |
        # or as above but with the numeric version of the date
        {TIME}?{NUM_DATE}{TIME}?
    ) 
    # or a time on its own
    |
    (?:{TIME})
)"""

price_regex = "(((?:\\d+\.)*\\d+,\\d+)|(\\d+\.\\d+))(?:[/\\s]*)(?:(gbp|\%))"

date_pattern = re.compile(COMBINED, re.IGNORECASE | re.VERBOSE | re.UNICODE)
