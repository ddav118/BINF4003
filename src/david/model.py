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
import csv
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

# modify the preferred_name column to add the negex value
metamap["preferred_name"] = metamap["preferred_name"].astype(str)
metamap["preferred_name"] = np.where(
    metamap.negex == 1,
    metamap.preferred_name + " (NEGATED)",
    metamap.preferred_name,
)

metamap = metamap[["subject_id", "hadm_id", "preferred_name", "score"]]
metamap.score = metamap.score.astype("float32")

# split into development and test sets; test set is the last 10% of the data
X_dev = metamap.iloc[: int(len(metamap) * 0.99)]
X_test = metamap.iloc[int(len(metamap) * 0.99) :]
print(X_dev.shape, X_test.shape)

# subject_id as key and list of ICD codes as values
# set of unique combinations of subject_ids, hadm_ids from metamap
keys = X_dev[["subject_id", "hadm_id"]].drop_duplicates()

icd_diag = pd.read_csv("combined_mapped_icd_diagnoses.csv", low_memory=False)
merged_icd = pd.merge(icd_diag, keys, on=["subject_id", "hadm_id"], how="right")
# groupby subject_id and hadm_id and make list of icd_codes
merged_icd = merged_icd.groupby(["subject_id", "hadm_id"]).agg(
    {"icd_code": lambda x: list(x)}
)
merged_icd.reset_index(inplace=True)
icd_dict = dict(zip(merged_icd.subject_id, merged_icd.icd_code))


merged_icd_test = pd.merge(icd_diag, X_test, on=["subject_id", "hadm_id"], how="right")
merged_icd_test = merged_icd_test.groupby(["subject_id", "hadm_id"]).agg(
    {"icd_code": lambda x: list(x)}
)
merged_icd_test.reset_index(inplace=True)
icd_dict_test = dict(zip(merged_icd_test.subject_id, merged_icd_test.icd_code))

patientsPerSymptom = defaultdict(set)
symptomsPerPatient = defaultdict(set)

patientsPerDisease = defaultdict(set)
diseasesPerPatient = defaultdict(set)
diseaseNames = defaultdict(str)

for index, row in tqdm(icd_diag.iterrows(), total=icd_diag.shape[0]):
    disease = row["icd_code"]
    diseaseNames[disease] = row["long_title"]


for index, row in tqdm(X_dev.iterrows(), total=X_dev.shape[0]):
    symptom = row["preferred_name"]
    patient = row["subject_id"]
    patientsPerSymptom[symptom].add(patient)
    symptomsPerPatient[patient].add(symptom)

for patient, diseases in tqdm(icd_dict.items(), total=len(icd_dict)):
    for disease in diseases:
        patientsPerDisease[disease].add(patient)
        diseasesPerPatient[patient].add(disease)


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


from collections import Counter
import joblib
from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed


X_test = X_test[["subject_id", "hadm_id", "preferred_name", "score"]]
patients = X_test.subject_id.unique()

# get patients and symptoms from X_test
pat_sym = X_test.groupby("subject_id").agg({"preferred_name": lambda x: list(x)})
pat_sym.reset_index(inplace=True)
merged_icd_test = pd.merge(
    icd_diag,
    X_test[["subject_id", "hadm_id"]],
    on=["subject_id", "hadm_id"],
    how="right",
)
# groupby subject_id and hadm_id and make list of icd_codes
merged_icd_test = merged_icd_test.groupby(["subject_id", "hadm_id"]).agg(
    {"icd_code": lambda x: list(x)}
)

merged_icd_test.reset_index(inplace=True)
merged_icd_test = pd.merge(merged_icd_test, pat_sym, on="subject_id")
merged_icd_test.pop("hadm_id")
merged_icd_test = merged_icd_test[["subject_id", "preferred_name"]].values


# function to be parallelized for each patient
def process_patient(patient, symptoms):
    # Adding test patient to model, will be removed later
    for symptom in symptoms:
        patientsPerSymptom[symptom].add(patient)
        symptomsPerPatient[patient].add(symptom)

    test_sim_pats = mostSimilarFast(
        patient, symptomsPerPatient, patientsPerSymptom, k=10
    )
    disease_frequency = Counter()

    for p, _ in test_sim_pats:
        possible_diseases = icd_dict[p]
        for d in possible_diseases:
            disease_frequency[d] += 1

    top_diseases = disease_frequency.most_common(None)
    result = defaultdict(float)

    for disease, _ in top_diseases:
        similar_diseases = mostSimilarFast(
            disease, patientsPerDisease, diseasesPerPatient
        ) # Find all similar diseases from k=10 most similar patients

        for similar_disease, similarity in similar_diseases:
            result[similar_disease] += similarity # Aggregate the similarity

    # take out the test patient from the model
    for symptom in symptoms:
        try:
            patientsPerSymptom[symptom].remove(patient) #prevent data leakage
        except KeyError:
            continue
    symptomsPerPatient[patient] = set()
    with open(f"/home/ddavilag/mimic/results/{patient}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Predicted Disease", "Jaccard Similarity Sum"]
        )  # Writing header

        for key, value in result.items():
            writer.writerow([key, value])


# Parallel processing
Parallel(n_jobs=-1, verbose=100, backend="multiprocessing")(
    delayed(process_patient)(patient, symptoms)
    for (patient, symptoms) in merged_icd_test
)