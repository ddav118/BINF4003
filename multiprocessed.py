from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from skr_web_api import Submission
import regex as re
import glob
from tqdm import tqdm

EMAIL = 'dmd2225@cumc.columbia.edu'
APIKEY = '19895da9-5d2d-41da-80a3-26a6e3b55fdf'
inst = Submission(EMAIL, APIKEY)

def metamap_lookup(inst):
    ...
    
if __name__ == '__main__':
    # Path: multiprocessed.py
    df = pd.read_csv('discharge_clean.csv')
    