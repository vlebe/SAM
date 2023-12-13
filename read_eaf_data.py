from speach import elan
import pandas as pd

eaf = elan.read_eaf('data/transcr/AAOR_merge.eaf')
dial = pd.DataFrame(eaf.to_csv_rows(), columns=['tier', '?', 'start ','stop ','duration ','text '])

print(dial)