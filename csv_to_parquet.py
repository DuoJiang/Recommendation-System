import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import argparse

def settings(memory):
	### setting ###
	conf = pyspark.SparkConf() \
			.setAll([('spark.app.name', 'downsampling code'),
					 ('spark.master', 'local'),
					 ('spark.executor.memory', memory),
					 ('spark.driver.memory', memory)])
	spark = SparkSession.builder \
			.config(conf=conf) \
			.getOrCreate()
	return spark

def create_schema_with_index():
	data_schema = StructType([
    StructField("user_id", StringType()),
    StructField("book_id", StringType()),
    StructField("is_read", IntegerType()),
    StructField("rating", IntegerType()),
    StructField("is_reviewed", IntegerType()) # for the data on HDFS
	])
	return data_schema


if __name__ == "__main__":
	
	### input arguments ###
	parser = argparse.ArgumentParser()
	parser.add_argument("--net_id", help="Inputing your NetID")
	parser.add_argument("--csv_path", help="Specifying the path of the csv file.")
	parser.add_argument("--parquet_path", help="Specifying the path of the parquet file.")
	parser.add_argument("--set_memory", help="Specifying the memory.")
	args = parser.parse_args()

	### setting ###
	spark = settings(args.set_memory)
	hdfs_original_path = "hdfs:///user/bm106/pub/goodreads/"
	hdfs_my_path = "hdfs:///user/"+args.net_id+"/goodreads/"
	#hdfs_path = "" # for local testing

	### 1. from csv to parquet ###
	# mypath = "hdfs:///user/kll482/goodreads/poetry_interactions.csv"
	data_schema = create_schema_with_index()
	data = spark.read.csv(hdfs_original_path+args.csv_path, header=True, schema=data_schema)

	data.write.option("schema", data_schema).parquet(hdfs_my_path+"data/"+args.parquet_path, mode="overwrite")
