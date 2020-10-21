def remove_low_interaction(data, user="user_id", threshold=20):
	'''
	This function is to remove those users who have low interactions
	(less than the threshold)
	'''
	# 1. initialize n_users and n_samples
	n_users = data.select(user).distinct().count()
	n_samples = data.count()
	# 2. count the interaction and filter the users 
	#user_id_frequent = data.groupBy(user).count().filter("count>"+threshold).select(user)
	# print the percentage of the user_id which is removed
	# 2. create partition by user and count with the partition
	window = Window.partitionBy(user)
	data_freq = data.withColumn("count_interaction",
								F.count(user) \
								.over(window))
	# 3. filter the users with less count and 
	data_freq = data_freq.where(col("count_interaction") > threshold)
	frequent_user_id = data_freq.select(user).distinct()
	print("I remove {0}% of the total users which has less than {1} iteractions.".\
		  format(str(round((1-frequent_user_id.count()/n_users)*100, 2)), threshold))
	# 4. remove the count_interaction column
	data_freq = data_freq.select(data.schema.names)
	print("I remove {0}% of the total rows by deleting the users which have less than {1} iteractions.".\
		  format(str(round((1-data_freq.count()/n_samples)*100, 2)), threshold))
	return data_freq

def remove_low_interaction(data, user="user_id", threshold=20):
	'''
	This function is to remove those users who have low interactions
	(less than the threshold)
	'''
	# 1. initialize n_users and n_samples
	n_users = data.select(user).distinct().count()
	n_samples = data.count()
	# 2. count the interaction and filter the users 
	user_id_frequent = data.groupBy(user).count().filter("count>"+threshold).select(user)
	# print the percentage of the user_id which is removed
	print("I remove {0}% of the total users who have less than {1} iteractions.".\
		  format(str(round((1-user_id_frequent.count()/n_users)*100, 2)), threshold))
	# 3. then delete the users with less count (by joining on the user_id which was not removed from the last step)
	data_freq = data.join(user_id_frequent, user, how='inner').select(data.schema.names)
	print("I remove {}% of the total rows by deleting the users which have less than {} iteractions.".\
		  format(str(round((1-data_freq.count()/n_samples)*100, 2)), threshold))

def downsampling(data, user="user_id", percentage=0.01):
	'''
	This function is to keep k% of the users in the data
	'''
	user_id_1_perc = data.select(user).distinct().sample(False, float(percentage), seed=123)
	downsample_data = data.join(user_id_1_perc, user, how='inner').select(data.schema.names)
	print("After downsampling, we only keep {0}% of the high-interation users. Now, wwe have {1} rows and {2} users.".\
			format(float(percentage)*100, downsample_data.count(), downsample_data.select(user).distinct().count()))
	return downsample_data

def downsampling(data, user="user_id", percentage=0.01):
	'''
	This function is to keep k% of the users in the data
	'''
	# get distinct user
	distinct_user = list(data.select(user).distinct().toPandas()[user])
	# create user_id -> group id

def index_func_no_order(data, col_name):
	'''
	col_name: data.select(col_name)
	'''
	# get distinct col
	distinct_col = list(data.select(col_name).distinct().toPandas()[col_name])
	# create distinct column -> group id
	mapping = {distinct_col[i]: i+1 for i in range(len(distinct_col))}
	# apply mapping dictionary
	mapping_expr = create_map([lit(i) for i in chain(*mapping.items())])
	index_data = data.withColumn(col_name+"_index", mapping_expr.getItem(col(col_name)))
	return index_data

def index_func_order(data, col_name):
	'''
	col_name: data.select(col_name)
	'''
	indexer = data.select(col_name).distinct()\
	  .orderBy(col_name) \
	  .withColumn(col_name+"_index",
	  			 monotonically_increasing_id() \
	  			 .cast(IntegerType()) \
	  			 ) \
	# indexer = (data.select(col_name).distinct() \
	# 			.rdd.map(itemgetter(0)).zipWithIndex() \
	# 			.toDF([col_name, col_name+"_index"]))
	data = data.join(indexer, [col_name])
	return data
