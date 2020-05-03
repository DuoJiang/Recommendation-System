import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import monotonically_increasing_id, col, expr
import pyspark.sql.functions as F
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import Evaluator
import argparse
import numpy as np
from itertools import product
import time

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
    StructField("is_reviewed", IntegerType()),
    StructField("user_id_index", IntegerType()),
    StructField("book_id_index", IntegerType())
	])
	return data_schema

def stratify_sampling(data, key, item, seed=123):
	'''
	This function is to extract approximately half of the interactions per user
	Input:
	1. key: user_id
	2. item: book_id
	'''
	# step 1: shuffle the rows
	data = data.orderBy(rand(seed=seed))
	# step 2: partition by key # user_id
	window = Window.partitionBy(key).orderBy(item) # item: book_id
	# step 3: get row index for partition
	new_data = data.withColumn("index_by_"+key,
					   F.row_number() \
					   .over(window))
	# step 4: get even number on index_by_key
	even_data = new_data.where(col("index_by_"+key)%2 == 0).select(data.schema.names)
	odd_data = new_data.where(col("index_by_"+key)%2 != 0).select(data.schema.names)
	return even_data, odd_data

def customized_split_func(data, train, val_user, user, set_seed=123):
	'''
	This function is to hold out half of the interactions per user from validation to training.
	Input:
	1. data: the whole dataframe
	2. train: the training set dataframe which contains all columns
	3. val_user: the validation or testing set dataframe which only contains user_id
	4. user: name of the user columns
	'''
	# temporary validation table
	val_data_temp = data.join(val_user, on=user, how='inner').select(data.schema.names)
	# sampleBy: stratefied sampling; for each user_id, we extract half of the interaction
	val_frac = dict([(uid[0], 0.5) for uid in val_user.select(user).collect()])
	val_data = val_data_temp.sampleBy(user, val_frac, seed = set_seed)
	# using the subtract function and putting the other 50% to the training
	train_val_data = train.union(val_data_temp.subtract(val_data))
	return train_val_data, val_data

def unionAll(*dataframes):
	'''
	Thiss function is to concatenate a list of dataframes.
	refer to https://datascience.stackexchange.com/questions/11356/merging-multiple-data-frames-row-wise-in-pyspark
	'''
	return reduce(DataFrame.unionAll, dataframes)

def kfold_split(data, user, k=4):
	'''
	This function is to split the whole to training, validation, and test set.
	From the basic setting of this project, we hold out 60% of the users for the training set, and
	20% of the users for the testing set. Beyond the basic setting, we will do 4-fold cross validation.
	Input:
	1. data: the whole dataset
	2. user: name of the user columns
	'''
	# 1. initialize a dictionay to store the k-fold DataFrame
	kfold_dict = {}
	# 2. let's first hold out users for the testing set. we will create the DataFrame later.
	user_split_for_test = data.select(user).distinct().randomSplit([0.8, 0.2], seed=123)
	test_user = user_split_for_test[1]
	train_hold_out_user = user_split_for_test[0]
	# 3. use the 80% of the users to do k-fold.
	percentage = 0.8/k
	k_fold = [percentage]*k # [0.2]*4 if k==4
	kfold_user = train_hold_out_user.randomSplit(k_fold, seed=123) # kfold_user is a k-element list of DataFrame
	# 4. let's create cross-validation dataset
	for i in range(len(kfold_user)):
		# let kfold_user[i] be the val
		hold_dfs = kfold_user[:i] + kfold_user[i+1:]
		train_user = unionAll(*hold_dfs) # users in the train set
		val_user = kfold_user[i] # users in the val set 
		# join data and train by user to get the training dataframe
		train_only_data = data.join(train_user, on=user, how='inner').select(data.schema.names) # user_split_sample is a 5-element list of user_id 
		# in the validation set, leave half of the interactions per user to the training set
		validation_result = customized_split_func(data=data,
												 train=train_only_data,
												 val_user=val_user,
												 user=user)
		val_data = validation_result[1]
		train_val_data = validation_result[0] # train + half val
		# in the test set, leave half of the interactions per user to the training set
		test_result = customized_split_func(data=data,
											train=train_val_data,
											val_user=test_user,
											user=user) # test_user from the beginning step
		test_data = test_result[1]
		final_train_data = test_result[0] # train + half val + half test
		# add train, val , test to the dict for the i fold
		kfold_dict[i] = [final_train_data, val_data, test_data]
	# return the k-fold dictionay
	return kfold_dict

def train_test_split(kfold_sets):
	'''
	After finding the best configuration,
	we will need to train the train set again.
	What we need to do is to join the train_data and val_data
	in one of the kfold_sets we create early on.
	Input:
	1. kfold_sets: the k-fold data sets
	'''
	train_val_test_data = kfold_sets[0] # all k-fold sets have the same test set
	full_train = unionAll(*train_val_test_data[:2])
	full_test = train_val_test_data[2]
	return full_train, full_test

def tuning_als(kfold_sets=None, rank_list=None, regParam_list=None,
			   metrics=None, k=10, maxIter=5, seed=123,
			   user="user_id_index", item="book_id_index", rating="rating"):
	'''
	This function is to run custom cross validation and metrics\
	Input:
	1. rank_list: a list of ranks for tuning
	2. regParam_list: a list of regulization parameters for tuning
	3. train_data: train set
	4. val_data: validation set
	5. k: top k items for evaluation
	6. ranking_metrics: the function uses the ranking metrics if this is not False; 
						{precisionAt, meanAveragePrecision, ndcgAt}
	7. regression_metrics: the function uses the regression metrics if this is not False; 
							{rmse, mae, r2}
	output:
	1. best_param_dict: a dictionary of the best configuration
	2. tuning_table: a dictionary of all configurations
	'''
	if rank_list == None or regParam_list == None:
		print("Error! Please enter rank_list or regParam_list.")
		return
	if kfold_sets == None:
		print("Error! You must enter the k-fold sets")
		return
	if metrics == None:
		print("Error! You must select a metric.")
		return
	# tuning_table: for storing the hyperparameter and metrics
	tuning_table = {"rank": [],
					"regParam": [],
					"avg_metrics": []}
	# a combination of all tuning hyperparameters
	param_combination = list(product(rank_list, regParam_list))
	for i, params in enumerate(param_combination):
		print("Start " + str(i+1) + " configuration.")
		# initialize parameters, total_metrcs 
		rank, regParam = params[0], params[1]
		total_metrics = [] # list: for storing metrics in each k-fold interaction
		# storing the rank and regParam
		tuning_table["rank"].append(rank)
		tuning_table["regParam"].append(regParam)
		for k_index in range(len(kfold_sets)):
			# initialize train set, validation set
			train_data, val_data = kfold_sets[k_index][0], kfold_sets[k_index][1]
			# initializa ALS
			#print("xxxx")
			als = ALS(rank=rank, maxIter=maxIter, regParam = regParam, seed=123, 
		              coldStartStrategy="drop", userCol=user, 
		              itemCol=item, ratingCol=rating,
		              implicitPrefs=False, nonnegative=True)
			#print("aaaa")
			model = als.fit(train_data)
			#print("bbbb")
			predictions = model.transform(val_data)
			#print("cccc")
			# evaluation
			if metrics in ["rmse", "mae", "r2"]:
				# we use the regression metrics
				metrics_result = top_k_regressionmetrics(
									dataset=predictions, k=k,
									regression_metrics=metrics,
									user=user, item=item, rating=rating,
									prediction="prediction")
			elif metrics in ["precisionAt", "meanAveragePrecision", "ndcgAt"]:
				# we use the ranking metrics
				metrics_result = top_k_rankingmetrics(
									dataset=predictions, k=k,
									ranking_metrics=metrics,
									user=user, item=item, rating=rating,
									prediction="prediction")
			total_metrics.append(metrics_result)
			#print(k_index+1)
		print("Finish " + str(i+1) + " configuration.")
		# compute average metrics for k-fold cross validation
		avg_metrics = np.mean(total_metrics)
		tuning_table["avg_metrics"].append(avg_metrics)
	# find the best hyperparamters from the average metrics of k-fold
	best_param_dict = {}
	if metrics in ["rmse", "mae", "r2"]:
		# we use the regression metrics (select minimum)
		best_index = np.argmin(tuning_table["avg_metrics"])
	elif metrics in ["precisionAt", "meanAveragePrecision", "ndcgAt"]:
		# we use the ranking metrics (select maximum)
		best_index = np.argmax(tuning_table["avg_metrics"])
	# store the best configuration into the dictionary
	best_param_dict["rank"] = tuning_table["rank"][best_index]
	best_param_dict["regParam"] = tuning_table["regParam"][best_index]
	best_param_dict["avg_metrics"] = tuning_table["avg_metrics"][best_index]
	return best_param_dict, tuning_table

def top_k_rankingmetrics(dataset=None, k=10, ranking_metrics="precisionAt", user="user_id_index",
 						item="book_id_index", rating="rating", prediction="prediction"):
	'''
	This function is to compute the ranking metrics from predictions.
	Input:
	1. k: only evaluate the performance of the top k items
	2. ranking_metrics: precisionAt, meanAveragePrecision, ndcgAt 
	3. user, item, prediction: column names; string type

	refer to https://vinta.ws/code/spark-ml-cookbook-pyspark.html
	'''
	if dataset == None:
		print("Error! Please specify a dataset.")
		return 
	# prediction table
	windowSpec = Window.partitionBy(user).orderBy(col(prediction).desc())
	perUserPredictedItemsDF = dataset \
		.select(user, item, prediction, F.rank().over(windowSpec).alias('rank')) \
		.where('rank <= {}'.format(k)) \
		.groupBy(user) \
		.agg(expr('collect_list({}) as items'.format(item)))
	# actual target table
	windowSpec = Window.partitionBy(user).orderBy(col(rating).desc())
	perUserActualItemsDF = dataset \
		.select(user, item, rating, F.rank().over(windowSpec).alias('rank')) \
		.where('rank <= {}'.format(k)) \
		.groupBy(user) \
		.agg(expr('collect_list({}) as items'.format(item)))
	# join
	perUserItemsRDD = perUserPredictedItemsDF \
		.join(F.broadcast(perUserActualItemsDF), user, 'inner') \
		.rdd \
		.map(lambda row: (row[1], row[2]))
	ranking_metrics_evaluator = RankingMetrics(perUserItemsRDD)
	# get the result of the metric
	if ranking_metrics == "precisionAt":
		precision_at_k = ranking_metrics_evaluator.precisionAt(k)
		#print("precisionAt: {}".format(round(precision_at_k, 4)))
		return precision_at_k
	elif ranking_metrics == "meanAveragePrecision":
		mean_avg_precision = ranking_metrics_evaluator.meanAveragePrecision(k)
		#print("meanAveragePrecision: {}".format(round(mean_avg_precision, 4)))
		return mean_avg_precision
	elif ranking_metrics == "ndcgAt":
		ndcg_at_k = ranking_metrics_evaluator.ndcgAt(k)
		#print("meanAveragePrecision: {}".format(round(ndcg_at_k, 4)))
		return ndcg_at_k

def top_k_regressionmetrics(dataset=None, k=10, regression_metrics="rmse", user="user_id_index",
					 item="book_id_index", rating="rating", prediction="prediction"):
	'''
	This function is to compute the regression metrics from predictions
	Input:
	1. k: only evaluate the performance of the top k items
	2. regression_metrics: rmse, mae, r2 
	3. user, item, prediction: column names; string type

	refer to https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html
	'''
	if dataset == None:
		print("Error! Please specify a dataset.")
		return
	# prediction table
	windowSpec = Window.partitionBy(user).orderBy(col(prediction).desc())
	user_items_prediction_df = dataset \
		.select(user, item, prediction, rating, F.rank().over(windowSpec).alias('rank')) \
		.where('rank <= {}'.format(k))
	# regression metrics
	regression_metrics_evaluator = RegressionEvaluator(metricName=regression_metrics,
													   labelCol=rating,
                                					   predictionCol=prediction)
	result = regression_metrics_evaluator.evaluate(user_items_prediction_df)
	#print(result)
	#print("{}: {}".format(regression_metrics, round(result, 4)))
	return result # return rmse, mae, or r2

def set_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--from_net_id", help="Inputing the netID for reading data")
	parser.add_argument("--to_net_id", help="Inputing the netID for saving models")
	parser.add_argument("--parquet_path", help="Specifying the path of the parquet file you want to read.")
	parser.add_argument("--top_k", help="Only evaluating top k interations.")
	parser.add_argument("--k_fold_split", help="Doing k-fold cross validation.")
	parser.add_argument("--metrics", help="The metrics for cross validation and measurement.")
	parser.add_argument("--rank_list", help="A list of ranks for tuning.")
	parser.add_argument("--regParam_list", help="A list of regularization parameters for tuning.")
	parser.add_argument("--path_of_model", help="Save the fitted model with this path.")
	parser.add_argument("--set_memory", help="Specifying the memory.")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	
	# arguments 
	args = set_arguments()

	# initial some parameters from args
	top_k = int(args.top_k)
	my_metrics = args.metrics
	rank_list = eval(args.rank_list)
	regParam_list = eval(args.regParam_list)
	path_of_model = args.path_of_model
	k_fold_split = int(args.k_fold_split)

	# setting 
	spark = settings(args.set_memory)

	# path
	from_hdfs_path = "hdfs:///user/"+args.from_net_id+"/goodreads/"
	to_hdfs_path = "hdfs:///user/"+args.to_net_id+"/goodreads/"
	to_home_path = "/home/"+args.to_net_id+"/goodreads/"
	#hdfs_path = ""
	
	### 1. read data ###
	print("Reading the data.")
	data_schema = create_schema_with_index()
	data = spark.read.schema(data_schema).parquet(from_hdfs_path+"data/"+args.parquet_path)
	# data = spark.read.parquet("indexed_poetry.parquet", schema=data_schema)
	data.printSchema()

	### 2. get k-fold cross validation ###
	print("Creating k-fold training and validation sets.")
	kfold_sets = kfold_split(data, "user_id", k=k_fold_split)

	### 3. tuning ALS by cross validation ###
	start_time = time.time()

	print("Tuning the ALS model.")
	# cross validation tuning
	# rank_list = [5] # [5, 10, 15, 20]
	# regParam_list = [0.01] # np.logspace(start=-3, stop=2, num=6)
	tuning_result = tuning_als(kfold_sets=kfold_sets, rank_list=rank_list,
					regParam_list=regParam_list, k=top_k, maxIter=5,
				   	metrics=my_metrics)

	best_config = tuning_result[0]
	best_rank, best_regParam = best_config["rank"], best_config["regParam"]

	### 4. prediction on the test set ###
	# train on the train set again, and then make prediction on the test set
	# initialize ALS estimator
	print("Re-training on the train set and predicting on the test set.")
	als = ALS(rank=best_rank, regParam = best_regParam, maxIter=5,
			  seed=123, coldStartStrategy="drop", userCol="user_id_index", 
              itemCol="book_id_index", ratingCol="rating",
              implicitPrefs=False, nonnegative=True)
	# train test split
	train_data, test_data = train_test_split(kfold_sets=kfold_sets)
	model = als.fit(train_data)
	predictions = model.transform(test_data) # predictions is a DataFrame with prediction column
	# compute ranking metrics on the test set
	if my_metrics in ["rmse", "mae", "r2"]:
		test_metrics = top_k_regressionmetrics(dataset=predictions,
						k=top_k,
						regression_metrics=my_metrics,
						user="user_id_index",
						item="book_id_index",
						rating="rating",
						prediction="prediction")
	elif my_metrics in ["precisionAt", "meanAveragePrecision", "ndcgAt"]:
		test_metrics = top_k_rankingmetrics(dataset=predictions,
						k=top_k,
						ranking_metrics=my_metrics,
						user="user_id_index",
						item="book_id_index",
						rating="rating",
						prediction="prediction")

	end_time = time.time()
	time_statement = "It takes {0} seconds to tune and train the model.".\
						format(str(round(end_time-start_time, 2)))
	print(time_statement)
	
	### 5. save the estimator (model) ###
	# refer to https://spark.apache.org/docs/2.3.0/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression.save
	print("Saving the estimator.")
	model.write().overwrite().save(to_hdfs_path+"models/"+path_of_model)

	# record all of the hyperparameter configurations, the best configuration, testing result
	print("Recording the tuning history.")
	with open(to_home_path+"history/"+"tuning_history.txt", "a+") as file:
		write_args = (path_of_model,
				   	  str(rank_list),
				   	  str(regParam_list),
				   	  best_rank,
				   	  best_regParam,
				   	  my_metrics,
				   	  round(test_metrics, 4),
				   	  time_statement)
		file.write("Model Path: {0}\n" \
				   "Rank List: {1}\n" \
				   "RegParam List: {2}\n" \
				   "Best Rank: {3}; Best RegParam: {4}\n" \
				   "Test Result ({5}): {6}" \
				   "Note: {7}\n\n" \
				   "---------" \
				   "\n\n" \
				   .format(*write_args))
