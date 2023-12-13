import re
import pandas as pd
import os

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

for f in os.listdir('raw metamap output'):
    with open(f'raw metamap output/{f}') as file:
        output = open(f'cleaned-metamap-csv/{f[:-4]}.csv', 'w')
        lines = file.readlines()
        concepts = [parse_conceptmmi(line) for line in lines]
        output.write('cui, text, negex\n')
        for concept in concepts:
            if "cui" in concept:
                cui = concept["cui"]
            if "trigger" in concept:
                trigger_info = concept["trigger"].split(', "')
                for item in trigger_info:
                    lst = item.split(',"')
                    for annotation in lst:
                        negex = annotation[-1]
                        temp = re.findall(r'(?<=\d-")[^"]+', annotation)
                        output.write(f'{cui},{temp[0]},{negex}\n')
    
