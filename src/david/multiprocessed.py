import time
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pymetamap import MetaMap
import re
import os
from tqdm import tqdm
import glob

metamap_base_dir = "/home/ddavilag/mimic/public_mm/"
metamap_bin_dir = "bin/metamap20"

mm = MetaMap.get_instance(metamap_base_dir + metamap_bin_dir)
# Download the set of stop words the first time
# nltk.download("punkt")
# nltk.download("stopwords")
# stop_words = set(stopwords.words("english")
# https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/ClinicalText.pdf
# https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/OutOfMemory.pdf


tqdm.pandas(desc="My Progress Bar")


def process_clinical_note(patientID, corpus, test=False):
    """
    Process a clinical note using MetaMap.
    """
    try:
        if not corpus:
            print(f"Patient {patientID} failed, empty corpus")
            return
        concepts, error = mm.extract_concepts(
            sentences=corpus,
            word_sense_disambiguation=True,
            restrict_to_sts=["sosy"],
            # allow_acronym_variants=True,
            #term_processing=True,
            prune=35,
            #ignore_word_order=True,
            prefer_multiple_concepts=True,
            composite_phrase=4,
        )
        
        if concepts == []:
            print(f"Patient {patientID} failed, no concepts found + {error}")
            with open("./no_matches.txt", "a") as f:
                f.write(f"{patientID}\n")
            return
        with open(f"./data/{patientID}.txt", "w") as f:
            for concept in concepts:
                f.write(str(concept))
                f.write("\n")
    except Exception as e:
        print(f"Patient {patientID} failed, error: {e}")
        return
    print(f"Patient {patientID} processed")


def replace_multiple_whitespace(text):
    return re.sub(r"\d", "", text)  # re.sub(r"\s+", " ", text).strip()


def preprocess_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]
    return " ".join(filtered_sentence)


def to_metamap():
    # df = pd.read_csv("discharge_cleanest.csv", low_memory=False)
    df = pd.read_csv("discharge_cleaner.csv", low_memory=False)
    df.dropna(inplace=True)

    print("Number of available cores: ", os.cpu_count())
    files_processed = glob.glob("./data/*.txt")
    print("Number of processed files: ", len(files_processed))
    # get only the patient id from the file name
    pids = [int(os.path.basename(f).split(".")[0]) for f in files_processed]
    print(pids[:3])

    def clean_metamap_input(text):
        text = text.lower()
        text = re.sub(r"\d+", "n", text)
        text = re.sub(r"\s+", " ", text).strip()
        # clean newline characters
        text = text.replace("\n", " ")
        # # replace all non alphanumeric characters
        # text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        # replace all underscores
        text = text.replace("_", "")
        return text

    df["corpus"] = df["corpus"].progress_apply(clean_metamap_input)

    with open("./no_matches.txt", "r") as f:
        no_matches = f.readlines()
    no_matches = [int(i.strip()) for i in no_matches]
    print(no_matches[:3])
    # remove the pids with no matches from the dataframe
    df = df[~df["subject_id"].isin(no_matches)]

    df = df[~df["subject_id"].isin(pids)]
    print(len(df))
    text_corpus = df["corpus"].values
    patient_ids = df["subject_id"].values
    text_corpus = [i.split(".") for i in text_corpus]
    text_corpus = [x[:-1] for x in text_corpus if x != ['']]

    
    print(text_corpus[0])
    print(patient_ids[0])
    len(text_corpus), len(patient_ids)
    #text_corpus = text_corpus[text_corpus.apply(len) > 0]

    process_clinical_note(patient_ids[0], text_corpus[0], test=True)
    Parallel(n_jobs=-1, verbose=100, backend="multiprocessing")(
        delayed(process_clinical_note)(patient_id, text)
        for patient_id, text in zip(patient_ids, text_corpus)
    )
