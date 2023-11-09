# BINF 4008: Symbolic Methods - Final Project

## Repository Structure

```bash
├── README.md
├── .gitignore
├── data
│   ├── {patient_id}.txt       - MetaMap output for each patient
│
├── src
│   ├── cindy
│   ├── david
│   │   ├── project.ipynb      - data cleaning, MetaMap example, and defines manual annotation set (n=100)
│   │   ├── multiprocessed.py  - cleans input text, runs MetaMap on input text, and writes output data/{patient_id}.txt files
│   │   ├── convert.py         - converts manual annotation concept_id to UMLS CUI
│   │   ├── compare.ipynb      - compares MetaMap outputs against manual annotations
│   ├── elizabeth
|
├── BRAT_config
│   ├── annotation.conf        - defines annotation types
│   ├── visual.conf            - defines visual attributes for annotation types
│   ├── tools.conf             - defines tools for annotation types
│   ├── template.conf          - defines template for annotation types

```

### Project Members: Elizabeth Chang, Hsi-Yu Chen, David Davila-Garcia

*equal contribution
