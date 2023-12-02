import pandas as pd
import numpy as np
import re
import os
import glob
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

tqdm.pandas()

# original: discharge = pd.read_csv("discharge_clean.csv", low_memory=False)
discharge = pd.read_csv("discharge_cleanest.csv", low_memory=False)
print(len(discharge))
discharge.isnull().sum()
discharge.dropna(inplace=True)
print(discharge.corpus.values[0])
df = pd.read_csv("discharge_clean.csv", low_memory=False)
print(df.text.values[0])
discharge["corpus"] = discharge["corpus"].progress_apply(lambda x: x.lower())
# discharge["text"] = discharge["text"].progress_apply(lambda x: " ".join(re.findall(r"\w+", x)))
# discharge["text"] = discharge["text"].progress_apply(
#             lambda x: re.sub(r"\d+", "n", x)
#         )
stopword_list = []

stopword_list = stopwords.words("english")
stopword_list += [
    "history of present illness",
    "brief hospital course",
    "admission",
    "birth",
    "date",
    "discharge",
    "service",
    "sex",
    "patient",
    "name",
    "hospital",
    "last",
    "first",
    "course",
    "past",
    "day",
    "one",
    "family",
    "chief",
    "complaint",
    "doctor",
    "dr",
    "drs",
    "nurse",
    "name",
    "unit",
    "date",
    "birth",
    "sex",
    "service",
    "medicine",
    "team",
    "followup",
    "instructions",
    "allergies",
    "adverse drug reactions",
    "attending",
    "emergency room",
    "call",
]

# Removing negated words
negated_words = [
    "don't",
    "aren't",
    "couldn't",
    "didn't",
    "doesn't",
    "hadn't",
    "hasn't",
    "haven't",
    "isn't",
    "mightn't",
    "mustn't",
    "needn't",
    "shan't",
    "shouldn't",
    "wasn't",
    "weren't",
    "won't",
    "wouldn't",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "when",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "ain",
    "aren",
    "couldn",
    "didn",
    "doesn",
    "hadn",
    "hasn",
    "haven",
    "isn",
    "mustn",
    "needn",
    "shan",
    "shouldn",
    "wasn",
    "weren",
    "won",
    "wouldn",
    "not",
    "no",
    "nor",
]

# Filter out the negated words
stopword_list = [word for word in stopword_list if word not in negated_words]


def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in set(stopword_list)])


discharge["corpus"] = discharge["corpus"].progress_apply(remove_stopwords)
discharge["corpus"][0]
# for every row, split the text by period
# remove spaces at the beginning and end of each string, and multiple spaces

discharge["corpus"] = discharge["corpus"].progress_apply(
    lambda x: [re.sub(r"^\s+|\s+$", "", s) for s in x.split(".") if s != ""]
)
discharge["corpus"] = discharge["corpus"].progress_apply(
    lambda x: [re.sub(r"\s+", " ", s) for s in x]
)
discharge["corpus"]

# remove the empty strings and commas and leading/trailing spaces
discharge["corpus"] = discharge["corpus"].progress_apply(
    lambda x: [s for s in x if s != ""]
)
discharge["corpus"] = discharge["corpus"].progress_apply(
    lambda x: [s.strip() for s in x]
)
discharge["corpus"][0]
# remove punctuation
discharge["corpus"] = discharge["corpus"].progress_apply(
    lambda x: [re.sub(r"[^\w\s]", "", s) for s in x]
)
discharge["corpus"] = discharge["corpus"].progress_apply(
    lambda x: [s.strip() for s in x]
)

# remove multiple spaces
discharge["corpus"] = discharge["corpus"].progress_apply(
    lambda x: [re.sub(r"\s+", " ", s) for s in x]
)


# join the list of strings in each row with a period
discharge["corpus"] = discharge["corpus"].progress_apply(lambda x: ". ".join(x))
print(discharge.isnull().sum())

# replace the phrase "history of present illness" and "brief course" with empty string
discharge["corpus"] = discharge["corpus"].progress_apply(
    lambda x: re.sub(r"history of present illness", "", x)
)
discharge["corpus"] = discharge["corpus"].progress_apply(
    lambda x: re.sub(r"brief hospital course", "", x)
)

# take out all underscores
discharge["corpus"] = discharge["corpus"].progress_apply(lambda x: x.replace("_", ""))
# fix all spaces
discharge["corpus"] = discharge["corpus"].progress_apply(lambda x: x.replace("  ", " "))

# take out all numeric characters 0-9
discharge["corpus"] = discharge["corpus"].progress_apply(lambda x: re.sub(r"\d", "", x))


import re

def grammar_fix(text):
    # Basic grammar corrections (capitalization and punctuation)
    # capitalize the very first letter and strip leading/trailing spaces
    text = text.capitalize().strip()

    # Split sentences and perform capitalization and space normalization in one step
    cleaned_sentences = [
        ' '.join(sentence.capitalize().split()) for sentence in re.split(r"(?<=[.!?]) +", text) if sentence
    ]
    # Join the sentences
    return " ".join(cleaned_sentences)

    


discharge["corpus"] = discharge["corpus"].progress_apply(
    lambda x: grammar_fix(x) if x != "" else np.nan
)
display(discharge.isnull().sum())
discharge.corpus.values[0]
discharge.dropna(inplace=True)

discharge.drop(columns=["text"], inplace=True)
discharge.to_csv("discharge_cleanest.csv", index=False)
discharge

directory_path = "/home/ddavilag/mimic/data"

# Use glob to find all .txt files in the specified directory
txt_files = glob.glob(f"{directory_path}/*.txt")
len(txt_files)
