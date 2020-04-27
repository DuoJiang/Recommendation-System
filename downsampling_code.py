### packages ###
from pyspark.sql import SparkSession

if __name__ == "__main__":
	### setting ###
	spark = SparkSession.builder \
			.master("local") \
			.appName("subset code") \
			.config("spark.some.config.option", "some-value") \
			.getOrCreate()

	### downsampling to 1% by users ###