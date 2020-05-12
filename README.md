# Goodreads-Recommendation-Systems

**Course: DS-GA 1004 Big Data**

Note: We use the Dumbo Systems from NYU for HDFS and Spark. 

## Instruction

### Short Introduction

The following codes are used to handle **goodreads\_interactions.parquet** on Dumbo. However, I used **poetry\_interactions.csv** to test my code on the local.

Note:

The following python scripts, **csv\_to\_parquet.py**, **downsampling.py**, and **modeling.py**, are stored in Dumbo.

### step 0: setting

- 1. Create the folder for this project on both Dumbo and HDFS

```
$ mkdir goodreads
$ hfs -mkdir goodreads
```

- 2. Change the working directory to goodreads on Dumbo.

```
$ cd path/to/goodreads
```

- 3. Create two folders in goodreads on HDFS:
 
	- **models** for saving the fitting estimators
	- **data** for saving the data

```
$ hfs -mkdir goodreads/models goodreads/data
```

- 4. Create the history folders in goodreads on Dumbo:

	- **history** for saving the tuning history

```
$ mkdir path/to/goodreads/history
```

- 5. Upload all of the python files, **csv\_to\_parquet.py**, **downsampling.py**, and **modeling.py**, from local machine to Dumbo

```
scp *.py NetID@dumbo.hpc.nyu.edu:goodreads
```

### step 1: CSV to Parquet

Run **csv\_to\_parquet.py** to transform the csv file to a parquet file.

Inputs:

1. csv\_path
2. parquet\_path
3. to\_net\_id: Your netid, for saving data to your HDFS

```python
spark-submit csv_to_parquet.py --net_id ${NetID} --csv_path goodreads_interactions.csv --parquet_path goodreads_interactions.parquet --set_memory 10g
```

### step 2: Downsampling

Run **downsampling\_code.py** to downsample the DataFrame and create index columns for both user\_id and book\_id. It will also create a new parquet file as a subset DataFrame.

Inputs:

1. read\_parquet\_path: the path for reading the whole dataset
2. write\_parquet\_path: the path for writing out the subset data
3. thres: We delete users who have interactions less than x (thres)
4. percentage: We keep x% of the users
5. from\_net\_id: ${MyNetID}, for reading data from my HDFS
6. to\_net\_id: ${YourNetID}, for saving data to your HDFS

I keep 1% of the users in the following code:

``` 
spark-submit downsampling.py --from_net_id ${MyNetID} --to_net_id ${YourNetID} --read_parquet_path goodreads_interactions.parquet --write_parquet_path twenty_five_percent_500.parquet --thres 500 --percentage 0.25 --set_memory 30g
```

After running ** downsampling.py** with the above inputs, I got the following three notifications.

- threshold = 20

1. I remove 18.07% of the total user id who has less than 20 iteractions.

2. I remove 0.53% of the total rows by deleting the user_id who has less than 20 iteractions.

- threshold = 500
 
1. I remove 87.31% of the total users who have less than 500 iteractions.
2. After downsampling, we only keep 1.0% of the high-interation users. Now, we have 1298926 rows and 1109 users.
3. After downsampling, we only keep 10.0% of the high-interation users. Now, we have 13524365 rows and 11053 users.
4. After downsampling, we only keep 100.0% of the high-interation users. Now, we have 136531923 rows and 111222 users.




### step 3: ALS Modeling

Run **modeling_code.py** to train the ALS model. I will save the estimator into the **models** folder. Also, I will save the tuning history to a text file, **tuning_history.txt**.

Inputs:

1. parquet_path: path of the data
2. kfold_split: We do 4-fold cross validation 
2. k: We only evaluating on the top k interactions
3. metrics: 
	- RankingMetrics: **precisionAt, meanAveragePrecision, ndcgAt**
	- RegressionMetrics: **rmse, mae, r2**
4. rank_list: The list of multiple ranks for tuning
5. regParam\_list: The list of multiple regParams for tuning
6. path\_of\_model: We save the fitted estimator with this name or path
7. from\_net\_id: ${MyNetID}, for reading data from my HDFS
8. to\_net\_id: ${YourNetID}, for saving data to your HDFS

Example:

1. metrics: precisionAt
2. rank_list: [5,10,15] \(Don't type a space between elements\)
3. regParam_list: [0.01,0.1] \(Don't type a space between elements\)
4. path\_of\_model: first_model

```
spark-submit modeling.py --from_net_id ${MyNetID} --to_net_id ${YourNetID} --parquet_path one_percent_500.parquet --top_k 500 --metrics precisionAt --rank_list [10,50,100,150] --regParam_list [0.001,0.01,0.1] --path_of_model model_1perc_1_precisionAt --set_memory 30g
```

