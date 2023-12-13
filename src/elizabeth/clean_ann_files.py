import pandas as pd
import os

def clean_ann_file(fname):
    '''
    clean .ann files from metamap and convert into dataframe that contains:
    annotaion_num, annotation text, negex, cui
    '''
    dict = {}
    with open(fname) as f:
        for line in f:
            elements = []
            elements = line.strip().split('\t')
            #set empty list values = [annotation text, negex, cui]
            values = ['','',''] 
            if elements[0].startswith('T'): 
                values[0] = elements[-1]
                if elements[1].startswith("Symptom"):
                    values[1]=0 #symptom = 0
                elif elements[1].startswith("Negated_Symptom"):
                    values[1]=1 #negated symptom = 1
                dict[elements[0]] = values

            elif elements[0].startswith('#'):
                annotation = elements[1].split(' ')[-1]
                notes = elements[-1].split()
                string = '' 
                for i in range(len(notes)):
                    if notes[i].startswith('C') and notes[i][1:8].isdigit():
                        dict[annotation][2] = notes[i] #add cui from annotator notes
                    else:
                        string += notes[i] + ' '
                if len(notes) != 1:
                    dict[annotation][0] = string #replace annotation text in list if annotator notes contains more detailed annotaion text

    cols = ['annotation text', 'negex', 'cui']   
    df = pd.DataFrame.from_dict(dict, orient = 'index', columns=cols)
    return df

def ann_to_csv(input_dir, output_dir):
    '''
    use clean_ann_file and convert .ann to .csv
    '''
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
         if fname.endswith('.ann'):
            f = os.path.join(input_dir, fname)
            df = clean_ann_file(f)
            df.to_csv(os.path.join(output_dir, fname[:-3]+'csv'))



