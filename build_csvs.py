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

FILE_LIST = LS.split()

if __name__ == '__main__':
    for filename in FILE_LIST:
        df = pd.read_csv(filename)

        # Figure out which is our 2nd column (column containing the name of the top-level grouping of ICD codes).
        if '10' in filename:
            if 'risk' in filename:
                super_column_name = 'Risk factors description'
            else:
                super_column_name = 'ICD10 description'
            df['icd'] = df['ICD-10-CM Code'].apply(repair_icd)

        else:  # assume ICD-9
            if 'risk' in filename:
                super_column_name = 'Description'
            else:
                super_column_name = "'ICD-9-CM CODE DESCRIPTION'"  # Yes, it really has single quotes in its name.
            df['icd'] = df['ICD-9-CM CODE'].apply(repair_icd)

        df['group'] = df[super_column_name]
        df_out = df[['icd', 'group']]
        df_out.to_csv('tidy-' + filename)
