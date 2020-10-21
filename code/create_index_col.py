import pyspark
from pyspark.sql.functions import monotonically_increasing_id, col
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql import SparkSession
import argparse


def index_func(data, col_name):
	'''
	col_name: data.select(col_name)
	'''
	indexer = (data.select(col_name).distinct()\
	  .orderBy(col_name) \
	  .withColumn(col_name+"_index", monotonically_increasing_id()))
	# indexer = (data.select(col_name).distinct() \
	# 			.rdd.map(itemgetter(0)).zipWithIndex() \
	# 			.toDF([col_name, col_name+"_index"]))
	data = data.join(indexer, [col_name])
	return data


if __name__ == "__main__":
	### setting ###
	memory = "3g"
	conf = pyspark.SparkConf() \
			.setAll([('spark.app.name', 'downsampling code'),
					 ('spark.master', 'local'),
					 ('spark.executor.memory', memory),
					 ('spark.driver.memory', memory)])
	spark = SparkSession.builder \
			.config(conf=conf) \
			.getOrCreate()

	### arguments ###
	parser = argparse.ArgumentParser()
	parser.add_argument("--read_parquet_path",
	 help="specifying the path of the parquet file you want to read.")
	parser.add_argument("--write_parquet_path",
	 help="specifying the path of the parquet file you want to write.")
	args = parser.parse_args()

	### read data ###
	data = spark.read.parquet(args.read_parquet_path)

	# create user_id_index and book_id_index (IntegerType)
	data = index_func(data=data, col_name="user_id")
	data = index_func(data=data, col_name="book_id")

	data.printSchema()

	# write parquet
	data.write.parquet(args.write_parquet_path, mode="ignore")