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

        # Figure out which is our 2nd column (column containing the name of the top-level grouping of ICD codes).
        if '10' in name:
            if 'risk' in name:
                super_column = 'Risk factors description'
            else:
                super_column = 'ICD10 description'
            df['icd-repaired'] = df['ICD-10-CM Code'].apply(repair_icd)

        else:  # assume ICD-9
            if 'risk' in name:
                super_column = 'Description'
            else:
                super_column = "'ICD-9-CM CODE DESCRIPTION'"  # Yes, it really has single quotes in its name.
            df['icd-repaired'] = df['ICD-9-CM CODE'].apply(repair_icd)

        df_out = df[['icd-repaired', super_column]]
        df_out.to_csv('tidy-' + name)
