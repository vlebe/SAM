from speach import elan
import pandas as pd

def read_eaf(file_path:str) :
    eaf = elan.read_eaf(file_path)
    dial = pd.DataFrame(eaf.to_csv_rows(), columns=['tier', '?', 'start ','stop ','duration ','text '])
    return dial