import os
import csv

file_path = '/Users/Orpheus/Desktop/ponyo/'

icd_10_dict = {}

with open(file_path + 'd_icd_diagnoses.csv') as d_icd_diagnoses:
    csvreader = csv.reader(d_icd_diagnoses)

    for contents in csvreader:
        if contents[1] == '10':
            icd_10_dict[contents[0]]=contents[0]

manual_mappings = {} 

with open(file_path + 'manual_mappings.csv') as f:
    csvreader = csv.reader(f)
    header = []
    header = next(csvreader)

    for contents in csvreader:
        if contents[2] not in icd_10_dict:
            print(contents)
