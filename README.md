# BINF 4008: Symbolic Methods - Final Project

## Repository Structure

```bash
├── README.md
├── .gitignore
├── data
│   ├── {patient_id}.txt                 - pyMetaMap output for each patient
│   ├── {patient_id}.csv                 - parse the gross pyMetaMap output into clean csv files for each patient
│
├── src
│   ├── cindy
│   ├── david
│   │   ├── 1-multiprocessed.py          - cleans input text, runs MetaMap on input text, and writes output data/{patient_id}.txt files
│   │   ├── 2-clean_mm_output.py         - cleans the pyMetaMap output into csv format
│   │   ├── 3-convert.py                 - converts manual annotation concept_id to UMLS CUI's - used for Midpoint Presentation
│   │   ├── 4-compare.ipynb              - compares MetaMap outputs against manual annotations - used for Midpoint Presentation
│   │   ├── 5-model.py                   - collaborative filtering model implementation. Computational Bottleneck for evaluation of model.
│   │   ├── utils
│   │   │   ├── fix_files.py             - removes empty data/{patient_id}.txt files
│   │   │   ├── pymetamap_test.ipynb     - gets list of data/{patient_id}.txt files
│   ├── elizabeth
|
├── BRAT_config
│   ├── annotation.conf                  - defines annotation types
│   ├── visual.conf                      - defines visual attributes for annotation types
│   ├── tools.conf                       - defines tools for annotation types
│   ├── template.conf                    - defines template for annotation types

```

### Project Members: Elizabeth Chang, Hsi-Yu Chen, David Davila-Garcia

*equal contribution
