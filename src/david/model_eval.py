import pandas as pd
import numpy as np
import glob
import os
from IPython.display import display
from tqdm import tqdm
from collections import Counter

results = glob.glob('results/*.csv')
print(len(results))
diseaseNames = pd.read_csv('diseaseNames.csv', header=None)
diseaseNames.dropna(inplace=True)
diseaseNames.columns = ['Disease', 'Disease Name']
print(len(diseaseNames))

icd_df_test = pd.read_csv('icd_df_test.csv')
display(icd_df_test)

import numpy as np

def calculate_auprc(precisions, recalls):
    # Ensure the thresholds are sorted (and corresponding precision and recall)
    sorted_indices = np.argsort(recalls)
    sorted_recall = recalls[sorted_indices]
    sorted_precision = precisions[sorted_indices]

    # Calculate AUPRC using the trapezoidal rule
    auprc = np.trapz(sorted_precision, sorted_recall)

    return auprc


thresholds = np.arange(0.05, 1.05, 0.05)
precisions = []
recalls = []
accuracies = []
disease_by_threshold = {}
for threshold in tqdm(thresholds):
    result ={
            'True Positive': 0,
            'False Positive': 0,
            'True Negative': 0,
            'False Negative': 0
        }
    disease_by_threshold[threshold] = {}
    t_pos_diseases = []
    t_neg_diseases = []
    f_pos_diseases = []
    f_neg_diseases = []
    for csv_file in results:
        
        df = pd.read_csv("/home/ddavilag/mimic/"+csv_file)
        #min-max normalization of similarity
        df['Jaccard Similarity Sum'] = (df['Jaccard Similarity Sum'] - df['Jaccard Similarity Sum'].min()) / (df['Jaccard Similarity Sum'].max() - df['Jaccard Similarity Sum'].min())
        df.sort_values(by=['Jaccard Similarity Sum'], ascending=False, inplace=True)
        df = pd.merge(df, diseaseNames, left_on='Predicted Disease', right_on='Disease', how='left')
        #get the file name
        file_name = os.path.basename(csv_file)
        # print(file_name)
        df.dropna(inplace=True)
        df.pop('Disease')
        
        
        patient_true = icd_df_test[icd_df_test['subject_id']==int(file_name[:-4])]
        diseases_true = patient_true['Disease_List'].values[0]
        diseases_true = diseases_true.replace('[', '')
        diseases_true = diseases_true.replace(']', '')
        diseases_true = diseases_true.replace("'", '')
        diseases_true = diseases_true.replace(" ", '')
        diseases_true = diseases_true.replace("\"", '')
        diseases_true = diseases_true.split(',')
        diseases_true = pd.DataFrame(diseases_true, columns=['Disease_List'])
        
        diseases_true = pd.merge(diseases_true, diseaseNames, left_on='Disease_List', right_on='Disease', how='left')
        diseases_true.dropna(inplace=True)
        # display(diseases_true)
        # display(patient_true)
        # display(df)
        
        
        for index, row in df.iterrows():
            
            if row['Jaccard Similarity Sum'] < threshold:
                if row['Disease Name'] in diseases_true['Disease Name'].values:
                    result['False Negative'] += 1
                    f_neg_diseases.append(row['Disease Name'])
                else:
                    result['True Negative'] += 1
                    t_neg_diseases.append(row['Disease Name'])
            else:
                if row['Disease Name'] in diseases_true['Disease Name'].values:     
                    result['True Positive'] += 1
                    t_pos_diseases.append(row['Disease Name'])
                else:
                    result['False Positive'] += 1
                    f_pos_diseases.append(row['Disease Name'])
    result['Threshold'] = threshold
    # handle zero division
    if result['True Positive'] + result['False Positive'] == 0:
        precision = 0
    else:
        precision = result['True Positive'] / (result['True Positive'] + result['False Positive'])
    # handle zero division
    if result['True Positive'] + result['False Negative'] == 0:
        recall = 0
    else:
        recall = result['True Positive'] / (result['True Positive'] + result['False Negative'])
    disease_by_threshold[threshold]['True Positive'] = t_pos_diseases
    disease_by_threshold[threshold]['False Positive'] = f_pos_diseases
    disease_by_threshold[threshold]['True Negative'] = t_neg_diseases
    disease_by_threshold[threshold]['False Negative'] = f_neg_diseases
    precisions.append(precision)
    recalls.append(recall)
    accuracies.append((result['True Positive'] + result['True Negative']) / (result['True Positive'] + result['True Negative'] + result['False Positive'] + result['False Negative']))
auprc = calculate_auprc(np.array(precisions), np.array(recalls))
print(auprc)
metrics = {
    'Threshold': thresholds,
    'Precision': precisions,
    'Recall': recalls,
    'Accuracy': accuracies
}
metrics = pd.DataFrame(metrics)
display(metrics)

best_threshold = disease_by_threshold[0.45]
best_threshold_tp = pd.DataFrame(best_threshold['True Positive'], columns=['True Positive'])
best_threshold_fp = pd.DataFrame(best_threshold['False Positive'], columns=['False Positive'])
best_threshold_tn = pd.DataFrame(best_threshold['True Negative'], columns=['True Negative'])
best_threshold_fn = pd.DataFrame(best_threshold['False Negative'], columns=['False Negative'])
best_threshold_df = pd.concat([best_threshold_tp, best_threshold_fp, best_threshold_tn, best_threshold_fn], axis=1)
display(best_threshold_df['False Positive'].dropna().value_counts(normalize=True))
display(best_threshold_df['False Negative'].dropna().value_counts(normalize=True))
display(best_threshold_df['True Positive'].dropna().value_counts(normalize=True))
display(best_threshold_df['True Negative'].dropna().value_counts(normalize=True))
