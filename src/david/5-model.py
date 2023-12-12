import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
import glob
from IPython.display import display
from surprise import Reader, Dataset, SVD
from surprise import KNNBasic
from surprise.model_selection import cross_validate
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import TimeSeriesSplit
from fastFM import als
from collections import defaultdict

tqdm.pandas(desc="My Progress Bar")
mm_df = pd.read_csv("metamap_output.csv", low_memory=False)

discharge = pd.read_csv("./discharge_clean.csv", parse_dates=["storetime"])
discharge.isnull().sum()
discharge.pop("charttime")
discharge = discharge[["subject_id", "storetime", "hadm_id"]]
metamap = pd.merge(mm_df, discharge, on="subject_id", how="left")
metamap["negex"] = metamap["negex"].astype(str)

# sort metamap by subject_id and storetime
metamap = metamap[~((metamap["negex"] != "0") & (metamap["negex"] != "1"))]
metamap["negex"] = metamap["negex"].astype(int)
metamap.sort_values(by=["storetime"], inplace=True)
metamap.reset_index(drop=True, inplace=True)

display(metamap)
# modify the preferred_name column to add the negex value
metamap["preferred_name"] = metamap["preferred_name"].astype(str)
metamap["preferred_name"] = np.where(
    metamap.negex == 1,
    metamap.preferred_name + " (NEGATED)",
    metamap.preferred_name,
)
metamap["preferred_name"].value_counts()

# Make dictionary with subject_id as key and list of ICD codes as values
# find set of unique combinations of subject_ids, hadm_ids from metamap
keys = metamap[["subject_id", "hadm_id"]].drop_duplicates()

icd_diag = pd.read_csv("combined_mapped_icd_diagnoses.csv", low_memory=False)
merged_icd = pd.merge(icd_diag, keys, on=["subject_id", "hadm_id"], how="right")
# groupby subject_id and hadm_id and make list of icd_codes
merged_icd = merged_icd.groupby(["subject_id", "hadm_id"]).agg(
    {"icd_code": lambda x: list(x)}
)
merged_icd.reset_index(inplace=True)
display(merged_icd)

icd_dict = dict(zip(merged_icd.subject_id, merged_icd.icd_code))
icd_dict

metamap["score"] = metamap["score"].astype(float)  # Convert to float if not already
metamap["negex"] = metamap["negex"].astype(int)  # Convert to int if not already
metamap["score"] = np.where(metamap.negex == 1, metamap.score * -1, metamap.score)
min(metamap.score), max(metamap.score)

metamap = metamap[["subject_id", "preferred_name", "score"]]
metamap.score = metamap.score.astype("float32")
# split into development and test sets; test set is the last 10% of the data
X_dev = metamap.iloc[: int(len(metamap) * 0.9)]
X_test = metamap.iloc[int(len(metamap) * 0.9) :]
print(X_dev.shape, X_test.shape)

# make TimeSeriesSplit object on X_dev
# tscv = TimeSeriesSplit(n_splits=5)
# results = []
# reader = Reader(
#     line_format="user item rating",
#     rating_scale=(min(metamap.score), max(metamap.score)),
# )

# for train_index, test_index in tscv.split(X_dev):
#     X_dev = X_dev[["subject_id", "preferred_name", "score"]]
#     filter_symptoms = X_dev.preferred_name.value_counts() > 100
#     filter_symptoms = filter_symptoms[filter_symptoms].index.tolist()
#     filter_users = X_dev.subject_id.value_counts() > 5
#     filter_users = filter_users[filter_users].index.tolist()
#     # X_dev = X_dev[
#     #     (
#     #         X_dev.preferred_name.isin(filter_symptoms)
#     #         & (X_dev.subject_id.isin(filter_users))
#     #     )
#     # ]
#     X_train, X_val = X_dev.iloc[train_index], X_dev.iloc[test_index]
#     trainset = Dataset.load_from_df(
#         X_train[["subject_id", "preferred_name", "score"]], reader
#     )
#     # dataset = Dataset.load_from_df(
#     #     X_val[["subject_id", "preferred_name", "score"]], reader
#     # )
#     testset = [tuple(row) for row in X_val.itertuples(index=False, name=None)]

#     model = KNNBasic(sim_options={"name": "cosine", "user_based": True})
#     model.fit(trainset.build_full_trainset())
#     predictions = model.test(testset, verbose=True)
#     print(predictions)
#     break

# find 5 most similar users for each user in the test set

# Example of analyzing the predictions
# for uid, iid, true_r, est, _ in predictions:
#     print(
#         f"User {uid} rated item {iid} {true_r}, and the estimated rating is {est}"
#     )


# metamap.dtypes

patientsPerSymptom = defaultdict(set)
symptomsPerPatient = defaultdict(set)

patientsPerDisease = defaultdict(set)
diseasesPerPatient = defaultdict(set)
diseaseNames = defaultdict(str)

for index, row in tqdm(icd_diag.iterrows()):
    disease = row["icd_code"]
    diseaseNames[disease] = row["long_title"]


for index, row in tqdm(metamap.iterrows()):
    symptom = row["preferred_name"]
    patient = row["subject_id"]
    patientsPerSymptom[symptom].add(patient)
    symptomsPerPatient[patient].add(symptom)

for patient, diseases in tqdm(icd_dict.items()):
    for disease in diseases:
        patientsPerDisease[disease].add(patient)
        diseasesPerPatient[patient].add(disease)


# def get_most_similar_users(user_id, user_similarity_matrix, k=5):
#     """
#     Returns top k similar users for a given user_id.
#     """
#     # Get similarity scores for the specified user with all users
#     similarity_scores = user_similarity_matrix[user_id]
#     # Get indices of top similar users
#     top_users_indices = np.argsort(similarity_scores)[::-1][:k]
#     # Get top similarity scores
#     top_similarity_scores = similarity_scores[top_users_indices]
#     return list(zip(top_users_indices, top_similarity_scores))


def Jaccard(s1, s2):
    """
    Returns Jaccard similarity between two sets.
    More computationally efficient implementation.
    """
    # Count the elements in the intersection
    intersection_count = sum(1 for element in s1 if element in s2)

    # The union count is the sum of the sizes of individual sets minus the intersection count
    union_count = len(s1) + len(s2) - intersection_count

    # To avoid division by zero
    if union_count == 0:
        return 0

    return intersection_count / union_count


import heapq


def mostSimilarFast(i, usersPerItem, itemsPerUser, k=5):
    patients = usersPerItem[i]

    # Efficiently build candidate items set using set comprehension
    candidateItems = {i2 for p in patients for i2 in itemsPerUser[p] if i2 != i}

    # Using a min-heap to find top-k similar items
    min_heap = []

    for i2 in candidateItems:
        sim = Jaccard(patients, usersPerItem[i2])
        if len(min_heap) < k:
            heapq.heappush(min_heap, (sim, i2))
        else:
            # Only add to heap if sim is greater than the smallest similarity in the heap
            if sim > min_heap[0][0]:
                heapq.heappushpop(min_heap, (sim, i2))

        # Early stopping if we find 'k' items with similarity 1
        if sim == 1 and len(min_heap) == k:
            break

    # Extract items from the heap and sort them in descending order of similarity
    return sorted([(i2, sim) for sim, i2 in min_heap], key=lambda x: x[1], reverse=True)


# test_sim_pats = mostSimilarFast(10034345, symptomsPerPatient, patientsPerSymptom)
# for p, s in test_sim_pats:
#     possible_diseases = icd_dict[p]
#     print(possible_diseases)
#     for d in possible_diseases:
#         print(d)
#         print(diseaseNames[d])
#         print(mostSimilarFast(d, patientsPerDisease, diseasesPerPatient))
# print(mostSimilarFast(p, patientsPerDisease, diseasesPerPatient))


from collections import Counter

# get 1000 patients from X_test
# for each patient, calculate the most similar patients based on symptoms
# for each similar patient, calculate the most similar diseases based on diseases
# aggregate the similarity scores for each disease
# select the top 5 most similar diseases
# print the details of the top 5 diseases
# print the actual diseases for the patient
# X_test = X_test[["subject_id", "preferred_name", "score"]]
# patients = X_test.subject_id.unique()


# # Find similar patients based on symptoms
# correct = 0
# incorrect = 0
# for patient in tqdm(patients):
#     test_sim_pats = mostSimilarFast(patient, symptomsPerPatient, patientsPerSymptom)

#     # Initialize a Counter to tally disease frequencies
#     disease_frequency = Counter()

#     # Iterate over similar patients and their diseases
#     for p, _ in test_sim_pats:
#         possible_diseases = icd_dict[p]
#         for d in possible_diseases:
#             disease_frequency[d] += 1

#     # Select the top 5 most common diseases
#     top_diseases = disease_frequency.most_common(None)
#     results = {}  # disease: aggregate frequency
#     # Print the details of the top 5 diseases
#     for disease, freq in top_diseases:
#         # print(f"Disease Code: {disease}")
#         # print(f"Disease Name: {diseaseNames[disease]}")
#         # print(f"Frequency: {freq}")
#         # Print similar diseases
#         similar_diseases = mostSimilarFast(
#             disease, patientsPerDisease, diseasesPerPatient
#         )
#         # print("Similar Diseases:")

#         for similar_disease, similarity in similar_diseases:
#             try:
#                 results[similar_disease] += similarity
#             except KeyError:
#                 results[similar_disease] = similarity
#     results = sorted(results.items(), key=lambda x: x[1], reverse=True)
#     for disease, similarity in results[:1]:
#         print(f"Disease Code: {disease}")
#         print(f"Disease Name: {diseaseNames[disease]}")
#         print(f"Aggregated Similarity: {similarity}")
#         if disease in icd_dict[patient]:
#             correct += 1
#         else:
#             incorrect += 1
# print(correct, incorrect)


import joblib
from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed


# Assuming X_test is a DataFrame loaded or defined previously
X_test = X_test[["subject_id", "preferred_name", "score"]]
patients = X_test.subject_id.unique()


# function to be parallelized for each patient
def process_patient(patient):
    test_sim_pats = mostSimilarFast(patient, symptomsPerPatient, patientsPerSymptom)
    disease_frequency = Counter()

    for p, _ in test_sim_pats:
        possible_diseases = icd_dict[p]
        for d in possible_diseases:
            disease_frequency[d] += 1

    top_diseases = disease_frequency.most_common(None)
    results = {}

    for disease, freq in top_diseases:
        similar_diseases = mostSimilarFast(
            disease, patientsPerDisease, diseasesPerPatient
        )

        for similar_disease, similarity in similar_diseases:
            results[similar_disease] = results.get(similar_disease, 0) + similarity

    results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # Return the top disease and its similarity score
    top_disease, similarity = results[0] if results else (None, 0)
    return patient, top_disease, similarity, top_disease in icd_dict[patient]


# Parallel processing
results = Parallel(n_jobs=-1, verbose=100, backend ='multiprocessing')(
    delayed(process_patient)(patient) for patient in tqdm(patients)
)

correct = 0
incorrect = 0

# Process the results
for patient, disease, similarity, is_correct in results:
    if disease is not None:
        print(f"Patient: {patient}")
        print(f"Disease Code: {disease}")
        print(f"Disease Name: {diseaseNames[disease]}")
        print(f"Aggregated Similarity: {similarity}")
        if is_correct:
            correct += 1
        else:
            incorrect += 1

print(correct, incorrect)


# # for train_index, test_index in tscv.split(X_dev):
# # print("TRAIN:", train_index, "TEST:", test_index)
# # X_train, X_val = X_dev.iloc[train_index], X_dev.iloc[test_index]
# # print(X_train.shape, X_val.shape)
# X_dev = Dataset.load_from_df(X_dev[["subject_id", "preferred_name", "score"]], reader)
# # X_val = Dataset.load_from_df(
# #     X_val[["subject_id", "preferred_name", "score"]], reader
# # )
# X_temp = X_dev
# print()
# model = KNNBasic(sim_options={"name": "cosine", "user_based": True})

# for i, row in tqdm(X_test.iterrows()):
#     X_temp.add_rating(row["subject_id"], row["preferred_name"], row["score"])
#     trainset = X_temp.build_full_trainset()
#     model.fit(trainset)
#     new_user = model.trainset.to_inner_uid(row["subject_id"])
#     similar_users = model.get_neighbors(new_user, k=5)
#     print(similar_users)


# # Now use construct_testset
# testset = trainset.construct_testset(raw_testset)

# predictions = model.test(testset=testset)
# similar_users = {
#     uid: model.get_neighbors(uid, k=5) for uid, iid, _, _, _ in predictions
# }
# print(similar_users)

# model = SVD()
# model.fit(X_train.build_full_trainset())

# # get top 5 most similar users
# user_features = model.pu  # User features
# item_features = model.qi  # Item features
# print(user_features.shape, item_features.shape)
# # cast to float32
# user_features = user_features.astype("float32")
# item_features = item_features.astype("float32")

# # Calculate cosine similarity between users
# user_similarity = cosine_similarity(user_features)
# # Make the diagonal elements (self-similarity) zero
# np.fill_diagonal(user_similarity, 0)
# user_similarity
