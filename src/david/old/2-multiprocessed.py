import time
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pymetamap import MetaMap
import regex as re
import os
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import glob


# Download the set of stop words the first time
# nltk.download("punkt")
# nltk.download("stopwords")
# stop_words = set(stopwords.words("english"))
tqdm.pandas(desc="My Progress Bar")


def metamap_lookup(corpus, patient_id):
    if os.path.exists(f"./data/{patient_id}.txt"):
        return
    try:
        concepts, errors = mm.extract_concepts(
            [corpus],
            mm_data_version="USAbase",
            word_sense_disambiguation=True,
            allow_acronym_variants=True,
            # prefer_multiple_concepts=True,
            # relaxed_model=True,
            restrict_to_sts=["dsyn", "sosy"],  # "fndg"
        )
        # Write the combined results to a file
        with open(f"./data/{patient_id}.txt", "w") as f:
            for concept in concepts:
                f.write(str(concept))
                f.write("\n")
    except Exception as e:
        return


def replace_multiple_whitespace(text):
    try:
        return re.sub(r"\s+", " ", text).strip()
    except:
        return text


def replace_digits(text):
    try:
        return re.sub(r"\d", "", text)
    except:
        return text


def preprocess_stopwords(text):
    try:
        stop_words = set(stopwords.words("english"))
        negation_words = {
            "no",
            "not",
            "nor",
            "none",
            "never",
            "cannot",
            "isn't",
            "wasn't",
            "shouldn't",
            "wouldn't",
            "couldn't",
            "won't",
            "don't",
            "doesn't",
            "didn't",
            "haven't",
            "hasn't",
            "hadn't",
        }

        # Tokenize the text
        words = word_tokenize(text)

        # Filter the words
        filtered_words = [
            word
            for word in words
            if word.lower() not in stop_words or word.lower() in negation_words
        ]

        return " ".join(filtered_words)
    except:
        return text


if __name__ == "__main__":
    # Path: multiprocessed.py
    mm = MetaMap.get_instance("/home/ddavilag/Desktop/mimic/public_mm/bin/metamap")
    # EMAIL = "dmd2225@cumc.columbia.edu"
    # APIKEY = "19895da9-5d2d-41da-80a3-26a6e3b55fdf"
    # inst = Submission(EMAIL, apikey=APIKEY)
    # inst.set_mm_ksource("2022AA")
    df = pd.read_csv("discharge_cleaner.csv", low_memory=False)
    df.corpus.dropna(inplace=True)

    print("Number of available cores: ", os.cpu_count())
    files_processed = glob.glob("./data/*.txt")

    print("Number of processed files: ", len(files_processed))
    manual_ann = glob.glob("/home/ddavilag/Desktop/mimic/manual_ann/*.txt")
    manual_ann = [int(os.path.basename(x).split(".")[0]) for x in manual_ann]
    metamap_manual_ann = []
    for file in manual_ann:
        res = glob.glob("/home/ddavilag/Desktop/mimic/data/" + str(file) + ".txt")
        if len(res) > 0:
            metamap_manual_ann.append(int(file))
    diff = set(manual_ann) - set(metamap_manual_ann)
    print("Number of manual annotations missing: ", len(manual_ann))
    # get only the patient id from the file name
    pids = [int(os.path.basename(f).split(".")[0]) for f in files_processed]
    print(pids[:3])
    # NOTE: did the commented out stuff already on discharge_clean.csv
    # df["text"] = df["corpus"].str.replace(":", "")
    # df["text"] = df["text"].str.replace("_", "")
    # df["text"] = df["text"].str.replace("\n", " ")
    # df["text"] = df["text"].progress_apply(replace_multiple_whitespace)
    # df.text.dropna(inplace=True)
    # df["text"] = df["text"].progress_apply(replace_digits)
    # df.text.dropna(inplace=True)
    # df["text"] = df["text"].progress_apply(preprocess_stopwords)
    # df.text.dropna(inplace=True)
    # df.to_csv("discharge_cleaner.csv", index=False)
    # print("cleaner csv saved")

    text_corpus = df["text"].values
    patient_ids = df["subject_id"].values
    # patient_dict = zip(patient_ids, text_corpus)

    # for patient_id, text in tqdm(zip(patient_ids, text_corpus), total=len(patient_ids)):
    #     if patient_id in pids:
    #         continue
    #     metamap_lookup(inst, text, patient_id)
    #     time.sleep(6)
    manual_missing = df[df["subject_id"].isin(diff)]
    print("Number of manual annotations missing: ", len(manual_missing))
    text_missing = manual_missing["text"].values
    patient_ids_missing = manual_missing["subject_id"].values
    Parallel(n_jobs=8, verbose=100)(
        delayed(metamap_lookup)(text, patient_id)
        for patient_id, text in zip(patient_ids_missing, text_missing)
    )

    Parallel(n_jobs=-1, verbose=100)(
        delayed(metamap_lookup)(text, patient_id)
        for patient_id, text in zip(patient_ids, text_corpus)
    )
