import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
import glob
from IPython.display import display
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity


tqdm.pandas(desc="My Progress Bar")
csv_paths = glob.glob("./pretty/*.csv", recursive=True)
len(csv_paths)
# find all the metamap_output csv files in the pretty folder

metamap = pd.DataFrame()
for csv_path in tqdm(csv_paths):
    df = pd.read_csv(csv_path, low_memory=False)
    metamap = pd.concat([metamap, df], ignore_index=True)
metamap.to_csv("./metamap_output.csv", index=False)

metamap = pd.read_csv("metamap_output.csv", low_memory=False)
display(metamap)
display(metamap.preferred_name.value_counts(ascending=False, normalize=True))
display(metamap.isnull().sum())


discharge = pd.read_csv("./discharge_clean.csv", parse_dates=["storetime"])
discharge.isnull().sum()
discharge.pop("charttime")
discharge = discharge[["subject_id", "storetime"]]
metamap = pd.merge(metamap, discharge, on="subject_id", how="left")
metamap.score.value_counts().sort_index(ascending=False)

# if the negex column is 1, the score should be negative
# if the negex column is 0, the score should be positive
metamap["score"] = np.where(metamap.negex == 1, metamap.score * -1, metamap.score)
metamap.score.value_counts().sort_index(ascending=False)

reader = Reader(
    line_format="user item rating timestamp",
    rating_scale=(min(metamap.score), max(metamap.score)),
)
data = Dataset.load_from_df(metamap[["subject_id", "preferred_name", "score"]], reader)

np.mean(metamap.score)

model = SVD()
trainset = data.build_full_trainset()
model.fit(trainset)

user_features = model.pu  # User features
item_features = model.qi  # Item features
print(user_features.shape, item_features.shape)
assert user_features.shape == item_features.shape

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_features)

# Make the diagonal elements (self-similarity) zero
np.fill_diagonal(user_similarity, 0)

user_similarity


def get_most_similar_users(user_id, user_similarity_matrix, k=5):
    """
    Returns top k similar users for a given user_id.
    """
    # Get similarity scores for the specified user with all users
    similarity_scores = user_similarity_matrix[user_id]
    # Get indices of top similar users
    top_users_indices = np.argsort(similarity_scores)[::-1][:k]
    # Get top similarity scores
    top_similarity_scores = similarity_scores[top_users_indices]
    return list(zip(top_users_indices, top_similarity_scores))


get_most_similar_users(0, user_similarity, k=5)
