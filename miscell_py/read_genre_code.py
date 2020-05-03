import numpy as np, pandas as pd
import json

def load_data(file_name, head = 100):
    count = 0
    data = []
    with open(file_name, "r") as file:
        for line in file:
            d = json.loads(line)
            count += 1
            data.append(d)
            
            # break if reaches the 100th line
            if (head is not None) and (count > head):
                break
    return data


if __name__ == "__main__":
	path = "/Users/garyliu/Documents/NYUClasses/BigData/Project/"
	data = load_data(path+"goodreads_interactions_poetry.json", None)

	# from list of dict to list of list with user_id, book_id, is_read, rating, is_review
	list_data = []
	for line in data:
	    list_data.append([line["user_id"], line["book_id"], int(line["is_read"]),
	                      line["rating"], int(len(line["review_text_incomplete"])>0)])
	df = pd.DataFrame(list_data, columns = ["user_id", "book_id", "is_read", "rating", "is_review"])

	## write to csv
	df.to_csv("poetry_interactions.csv", index=False)