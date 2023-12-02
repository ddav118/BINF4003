import os
from IPython.display import display
import re
from sys import displayhook
import pandas as pd
from joblib import Parallel, delayed


# Define a function to parse ConceptMMI lines
def parse_conceptmmi(line):
    conceptmmi = {}
    # Match the pattern ConceptMMI(...) and capture the inside
    if match := re.search(r"ConceptMMI\((.*?)\)$", line.strip()):
        # Split the captured group by comma, considering quoted and bracketed content
        fields = re.findall(
            r'(?:[^,"|\[(]|"(?:\\.|[^"])*"|\[(?:\\.|[^\]])*\])+', match.group(1)
        )
        for field in fields:
            # Split by '=', but only the first occurrence
            parts = field.split("=", 1)
            if len(parts) == 2:
                key, value = parts
                key = key.strip()
                # Strip extra quotes and brackets from the value
                value = value.strip(r'"[]\'')
                conceptmmi[key] = value
    return conceptmmi


# Define a function to process a single file
def process_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    concepts = [parse_conceptmmi(line) for line in lines]
    for concept in concepts:
        if "semtypes" in concept:
            # only get the sosy semantic type
            concept["semtypes"] = [
                i for i in concept["semtypes"].split(",") if i == "'[sosy]'"
            ]
        if "trigger" in concept:
            concept["negex"] = concept["trigger"][-1]
    # print(concepts[0])
    # remove all unnecessary quotes from keys and values
    for concept in concepts:
        for key, value in concept.items():
            if isinstance(value, str):
                concept[key] = value.strip("'")

    results_df = pd.DataFrame(concepts)

    keys_of_interest = ["preferred_name", "cui", "score", "pos_info", "negex"]
    results_df = results_df[keys_of_interest]

    # Save the DataFrame to a CSV file with a unique name
    file_name = os.path.basename(file_path)
    output_file = "./pretty/" + os.path.splitext(file_name)[0] + ".csv"
    results_df.loc[:, "subject_id"] = file_name.split(".")[0]
    # reorder the columns
    results_df = results_df[
        ["subject_id", "cui", "preferred_name", "score", "pos_info", "negex"]
    ]
    results_df = results_df[results_df["preferred_name"] != "Discharge"]
    results_df.score = results_df.score.astype(float)
    results_df = (
        results_df.groupby(["cui", "negex"])
        .agg(
            {
                "subject_id": "first",
                "preferred_name": ["first"],
                "cui": "first",
                "score": ["sum"],
                "negex": "first",
                "pos_info": lambda x: ",".join(x),
            }
        )
        .sort_values(by=("score", "sum"), ascending=False)
    )
    # rename the columns back to normal
    results_df.columns = results_df.columns.get_level_values(0)

    results_df.to_csv(output_file, index=False)


def prettify():
    # List of file paths to be processed
    import glob

    file_paths = glob.glob("./data/*.txt", recursive=True)
    csv_paths = glob.glob("./pretty/*.csv", recursive=True)
    print(len(file_paths), len(csv_paths), sep="\n")

    # with open(file_paths[1], "r") as f:
    #     for line in f:
    #         print(line)
    #     print()

    # Number of CPU cores to use for parallel processing
    num_cores = os.cpu_count()

    # Process the files in parallel using joblib
    Parallel(n_jobs=10)(delayed(process_file)(file_path) for file_path in file_paths)


prettify()

csv_paths = glob.glob("./pretty/*.csv", recursive=True)
len(csv_paths)
# find all the metamap_output csv files in the pretty folder

metamap = pd.DataFrame()
for csv_path in tqdm(csv_paths):
    df = pd.read_csv(csv_path, low_memory=False)
    metamap = pd.concat([metamap, df], ignore_index=True)
metamap.to_csv("./metamap_output.csv", index=False)
