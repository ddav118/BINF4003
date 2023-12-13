import os
import csv
from icdmappings import Mapper

file_path = '/Users/Orpheus/Desktop/ponyo/'

icd_10_dict = {}

with open(file_path + 'd_icd_diagnoses.csv') as d_icd_diagnoses:
    csvreader = csv.reader(d_icd_diagnoses)

    for contents in csvreader:
        if contents[1] == '10':
            icd_10_dict[contents[0]]=contents[2]

manual_mappings = {} 

with open(file_path + 'manual_mappings.csv') as f:
    csvreader = csv.reader(f)
    header = []
    header = next(csvreader)

    for contents in csvreader:
        manual_mappings[contents[0]] = contents[2]

data = []
mapper = Mapper()

with open(file_path + 'note_icd_diagnoses.csv') as diagnoses_icd:
    csvreader = csv.reader(diagnoses_icd)
    header = []
    header = next(csvreader)
        
    data.append(header)
 
    for contents in csvreader:
        if contents[5] == '9':
            icd_code = mapper.map(contents[4], source = 'icd9', target = 'icd10')
            if icd_code == 'NoDx' or icd_code == None:
                icd_code = manual_mappings[contents[4]]
            contents[4] = icd_code
            contents[5] = '10'
            contents[-1] = icd_10_dict[icd_code]
        data.append(contents)


with open(file_path + '/combined_mapped_icd_diagnoses.csv', 'w', newline = '') as combined_icd_diagnoses:
    csvwriter = csv.writer(combined_icd_diagnoses)
    csvwriter.writerows(data)

print("All done!")


