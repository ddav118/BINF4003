import time
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from skr_web_api import Submission
import regex as re
import os
#import nltk
from tqdm import tqdm
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
import glob


# Download the set of stop words the first time
# nltk.download("punkt")
# nltk.download("stopwords")
# stop_words = set(stopwords.words("english"))
tqdm.pandas(desc="My Progress Bar")


def metamap_lookup_chunk(inst, chunk):
    # Process a chunk of text
    inst.init_mm_interactive(
        chunk, ksource="2022AA", args="--negex -y -s -I -J dsyn,sosy"
    )
    output = inst.submit()
    if output.status_code != 200:
        print(output)
        print("error")
        print(output.status_code)
        return
    return output.content.decode()


def metamap_lookup(inst, corpus, patient_id, chunk_size=10000):
    # check if json file already exists
    # if os.path.exists(f"./data/{patient_id}.json"):
    #    return
    try:
        # inst.init_generic_batch(
        #     "metamap", "--lexicon db -Z 2022AA --silent -y -s -I -J dsyn,sosy"
        # )
        # inst.set_batch_file("USERINPUT", corpus)
        # output = inst.submit()
        # results = output.content.decode()
        # if output.status_code != 200:
        #     return
        # print(results)
        # time.sleep(6)
        # corpus = corpus.encode("ascii", "ignore").decode("ascii")
        if len(corpus) > 9999:
            # Break the corpus into chunks of size 'chunk_size'
            chunks = [
                corpus[i : i + chunk_size] for i in range(0, len(corpus), chunk_size)
            ]

            # Run metamap on each chunk
            chunked_results = [metamap_lookup_chunk(inst, chunk) for chunk in chunks]

            # Concatenate the processed chunks back together
            results = "".join(chunked_results)
        else:
            inst.init_mm_interactive(
                corpus, ksource="2022AA", args="--negex -y -s -I -J dsyn,sosy"
            )
            output = inst.submit()
            if output.status_code != 200:
                print(output)
                print("error")
                print(output.status_code)
                return
            results = output.content.decode()

        # Write the combined results to a file
        with open(f"./metamap_output/{patient_id}.json", "w") as f:
            f.write(results)
    except Exception as e:
        return


def replace_multiple_whitespace(text):
    return re.sub(r"\d", "", text)  # re.sub(r"\s+", " ", text).strip()


def preprocess_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]
    return " ".join(filtered_sentence)


if __name__ == "__main__":
    # Path: multiprocessed.py
    EMAIL = "dmd2225@cumc.columbia.edu"
    APIKEY = "187a5701-83da-4679-9975-c46c1398d525"
    inst = Submission(EMAIL, apikey=APIKEY)
    df = pd.read_csv("discharge_cleaner.csv", low_memory=False)

    print("Number of available cores: ", os.cpu_count())
    print(os.path.join(os.getcwd(),"metamap_output", "*.json"))
    files_processed = glob.glob(os.path.join(os.getcwd(),"metamap_output", "*.json"), recursive=True)

    print("Number of processed files: ", len(files_processed))
    # get only the patient id from the file name
    pids = [int(os.path.basename(f).split(".")[0]) for f in files_processed]
    print(pids[:3])
    # NOTE: did the commented out stuff already on discharge_clean.csv
    # df["text"] = df["text"].str.replace(":", "")
    # df["text"] = df["text"].str.replace("_", "")
    # df["text"] = df["text"].str.replace("\n", " ")
    # df["text"] = df["text"].progress_apply(replace_multiple_whitespace)
    # df["text"] = df["text"].progress_apply(preprocess_stopwords)
    # df.to_csv("discharge_cleaner.csv", index=False)
    # print("cleaner csv saved")

    text_corpus = df["text"].values
    patient_ids = df["subject_id"].values
    # patient_dict = zip(patient_ids, text_corpus)

    for patient_id, text in tqdm(zip(patient_ids, text_corpus), total=len(patient_ids)):
        print(patient_id, text)
        break
        if patient_id in pids:
            continue
        metamap_lookup(inst, text, patient_id)
        time.sleep(6)

    # results = Parallel(n_jobs=-1, verbose=100)(
    #     delayed(metamap_lookup)(inst, text, patient_id)
    #     for patient_id, text in zip(patient_ids, text_corpus)
    # )
