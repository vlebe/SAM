import pandas as pd
import os
from glob import glob
import platform

def load_all_ipus(folder_path:str="data/transcr", load_words:bool=False):
    """Load all csv and concatenate
    """
    file_list = glob(os.path.join(folder_path, f"*_merge{'_words' if load_words else ''}.csv"))
    # Load all csv files
    data = []
    for file in file_list:
        df = pd.read_csv(file, na_values=['']) # one speaker name is 'NA'

        if platform.system() == 'Windows':
            df['dyad'] = file.split('\\')[-1].split('_')[0]
        else :
            df['dyad'] = file.split('/')[-1].split('_')[0]

        data.append(df)
            
    data = pd.concat(data, axis=0).reset_index(drop=True)
    plabels = [col for col in data.columns if not any([col.startswith(c) 
            for c in ['dyad', 'ipu_id','speaker','start','stop','text', 'duration']])]
    return data

def filter_after_jokes(df_ipu:pd.DataFrame):
    """First few ipus are useless / common to all conversations"""
    jokes_end = df_ipu[df_ipu.text.apply(lambda x: False if isinstance(x, float) else (
                ('il y avait un âne' in x) or ("qui parle ça c'est cool" in x)))].groupby('dyad').agg(
                {'ipu_id':'max'}).to_dict()['ipu_id']
    return df_ipu[df_ipu.apply(lambda x: x.ipu_id > jokes_end.get(x.dyad,0), axis = 1)], jokes_end

if __name__ == "__main__" :
    data = load_all_ipus('data/transcr')
    data, _ = filter_after_jokes(data)

    # Ce fichier contient les labels ainsi que les transcriptions (données texte)

    # Preprocessing
    data['text'].fillna('non', inplace=True)
    data = data[(data["dyad"] != "AAOR") & (data["dyad"] != "JLLJ") & (data["dyad"] != "JRBG")]

    # Remove data with audio length < 0.15s 
    data = data[data["stop"]-data["start"] > 0.2].reset_index(drop=True)
    plabels = [col for col in data.columns if not any([col.startswith(c) 
            for c in ['dyad', 'ipu_id','speaker','start','stop','text', 'duration']])]
    print(data[plabels].sum(axis=0) / data.shape[0])
    data.drop(["ipu_id"], inplace=True, axis=1)
    data["id"] = data.index
    print(data.shape)

    data.to_csv("data.csv", index=False)