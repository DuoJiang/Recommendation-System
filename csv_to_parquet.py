### packages ###
from pyspark.sql import SparkSession
import argparse

if __name__ == "__main__":
	### setting ###
	spark = SparkSession.builder \
			.master("local") \
			.appName("csv to parquet") \
			.config("spark.some.config.option", "some-value") \
			.getOrCreate()

	### from csv to parquet ###
	parser = argparse.ArgumentParser()
	parser.add_argument("--csv_path", help="specifying the path of the csv file.")
	parser.add_argument("--parquet_path", help="specifying the path of the parquet file.")

	args = parser.parse_args()

	people = spark.read.csv(args.csv_path, header=True, 
	                        schema="user_id STRING, book_id STRING, is_read INT, "+\
	                         	   "rating INT, is_review INT")

	people_csv.write.parquet(args.parquet_path, mode="ignore")