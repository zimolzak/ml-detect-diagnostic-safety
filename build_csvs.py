import pandas as pd
from helpers import repair_icd

LS = """icd10-dizzy.csv
icd10-headache.csv
icd10-stroke-risk.csv
icd10-stroke.csv
icd9-dizzy.csv
icd9-headache.csv
icd9-stroke-risk.csv
icd9-stroke.csv
"""

FILENAMES = LS.split()

if __name__ == '__main__':
    for name in FILENAMES:
        df = pd.read_csv(name)
        if df.columns[0] == 'ICD-10-CM Code':
            df['icd-repaired'] = df['ICD-10-CM Code'].apply(repair_icd)
        elif '9' in df.columns[0]:
            pass
        df.to_csv('tidy-' + name)
