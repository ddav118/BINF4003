# Search for a Term in UMLS and return CUIs https://documentation.uts.nlm.nih.gov/rest/rest-api-cookbook/python-scripts.html#search-for-a-term
apikey = "187a5701-83da-4679-9975-c46c1398d525"
# This script will return CUI information for a single search term.
# Optional query parameters are commented out below.

import requests
import argparse


def results_list(input):
    # parser = argparse.ArgumentParser(description='process user given parameters')
    # parser.add_argument("-k", "--apikey", required = True, dest = "apikey", help = "enter api key from your UTS Profile")
    # parser.add_argument("-v", "--version", required =  False, dest="version", default = "current", help = "enter version example-2021AA")
    # parser.add_argument("-s", "--input", required =  True, dest="input", help = "enter a search term, using hyphens between words, like diabetic-foot")

    # args = parser.parse_args()
    version = "2020AA"

    uri = "https://uts-ws.nlm.nih.gov"
    content_endpoint = "/rest/search/" + version
    full_url = uri + content_endpoint
    page = 0

    try:
        while True:
            page += 1
            query = {"input": input, "apiKey": apikey, "pageNumber": page}
            # query['includeObsolete'] = 'true'
            # query['includeSuppressible'] = 'true'
            # query['returnIdType'] = "sourceConcept"
            # query['sabs'] = "SNOMEDCT_US"
            r = requests.get(full_url, params=query)
            r.raise_for_status()
            print(r.url)
            r.encoding = "utf-8"
            outputs = r.json()

            items = (([outputs["result"]])[0])["results"]

            if len(items) == 0:
                if page == 1:
                    print("No results found." + "\n")
                    break
                else:
                    break

            print("Results for page " + str(page) + "\n")

            for result in items:
                print("UI: " + result["ui"])
                print("URI: " + result["uri"])
                print("Name: " + result["name"])
                print("Source Vocabulary: " + result["rootSource"])
                print("\n")

        print("*********")

    except Exception as except_error:
        print(except_error)


input_concepts = [
    "abdominal pain",
    "chronic abdominal pain",
    "epigastric pain",
    "RUO pain",
    "LUO pain",
    "crampy pain",
    "vomiting",
    "diarrhea",
    "nausea",
    "fevers",
    "chills",
    "poor PO intake",
    "weight loss",
    "abdominal distention",
    "epigastric pain",
]
input_concepts = [
    "422587007",
    "55300003",
    "301715003",
    "62315008",
    "79922009",
    "301717006",
    "43724002",
    "386661006",
    "422400008",
    "111985007",
    "426977000",
    "64379006",
    "444746004",
    "LA17242-1",  # "LA17242-1",
]
vocab = [
    "SNOMEDCT_US",
    "SNOMEDCT_US",
    "SNOMEDCT_US",
    "SNOMEDCT_US",
    "SNOMEDCT_US",
    "SNOMEDCT_US",
    "SNOMEDCT_US",
    "SNOMEDCT_US",
    "SNOMEDCT_US",
    "SNOMEDCT_US",
    "SNOMEDCT_US",
    "SNOMEDCT_US",
    "SNOMEDCT_US",
    "LNC",
]


inputs = zip(input_concepts, vocab)

# results_list(input)

import requests

res_list = []
for input, vocab in inputs:
    # url = f"https://uts-ws.nlm.nih.gov/rest/crosswalk/2020AA/source/{vocab}/{input}?apiKey={apikey}"
    url = f"https://uts-ws.nlm.nih.gov/rest/search/2020AA?apiKey=187a5701-83da-4679-9975-c46c1398d525&string={input}&sabs={vocab}"
    # url = f"https://uts-ws.nlm.nih.gov/rest/search/current?apiKey=187a5701-83da-4679-9975-c46c1398d525&string={input}&searchType=exact"

    response = requests.request("GET", url)
    results = response.json()
    if len(results) == 0:
        url = f"https://uts-ws.nlm.nih.gov/rest/search/2020AA?apiKey=187a5701-83da-4679-9975-c46c1398d525&string={input}"
        response = requests.request("GET", url)
        results = response.json()
    print(
        f"Original Concept_code: {input}\n",
        f"Vocab: {vocab}\n",
        f'Mapped CUI: {results["result"]["results"][0]["ui"]}\n',
        f'Mapped Concept_name: {results["result"]["results"][0]["name"]}\n',
    )
    # print(response.text)
    # print(results)
    with open("manual_cui_list.csv", "a") as f:
        f.write(
            f"{input},{vocab},{results['result']['results'][0]['ui']},{results['result']['results'][0]['name']}\n"
        )
