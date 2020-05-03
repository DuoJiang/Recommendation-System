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

def set_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--to_net_id", help="Inputing the netID for saving models")
	parser.add_argument("--csv_path", help="Specifying the path of the csv file.")
	parser.add_argument("--parquet_path", help="Specifying the path of the parquet file.")
	parser.add_argument("--set_memory", help="Specifying the memory.")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	
	# input arguments
	args = set_arguments()

	# setting 
	spark = settings(args.set_memory)

	# path
	hdfs_teachers_path = "hdfs:///user/bm106/pub/goodreads/"
	to_hdfs_path = "hdfs:///user/"+args.net_id+"/goodreads/"
	#hdfs_path = "" # for local testing

	### 1. from csv to parquet ###
	# mypath = "hdfs:///user/kll482/goodreads/poetry_interactions.csv"
	data_schema = create_schema_with_index()
	data = spark.read.csv(hdfs_teachers_path+args.csv_path, header=True, schema=data_schema)

	data.write.option("schema", data_schema).parquet(to_hdfs_path+"data/"+args.parquet_path, mode="overwrite")
